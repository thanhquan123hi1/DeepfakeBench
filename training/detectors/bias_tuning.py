import re
import math
import logging
from dataclasses import dataclass, field
from typing import Union, List, Set, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Valid parameter types
VALID_PARAMETER_TYPES = [
    'frozen',
    'full',
    'all_bias',
    'all_bias_subspace',
    'linear_bias',
    'ln_bias',
    'attention_bias',
    'qkv_bias',
    'q_bias',
    'k_bias',
    'v_bias',
    'attn_proj_bias',
    'mlp_bias',
    'mlp_fc1_bias',
    'mlp_fc2_bias',
]

# Valid layer range shortcuts
VALID_LAYER_RANGES = ['all', 'early', 'middle', 'late']

# Mapping from parameter_type to active categories
CATEGORY_MAPPING: Dict[str, Set[str]] = {
    'frozen': set(),
    'full': {'all'},
    'all_bias': {
        'q_bias', 'k_bias', 'v_bias', 'attn_proj_bias',
        'mlp_fc1_bias', 'mlp_fc2_bias', 'ln_bias'
    },
    'all_bias_subspace': {
        'q_bias', 'k_bias', 'v_bias', 'attn_proj_bias',
        'mlp_fc1_bias', 'mlp_fc2_bias', 'ln_bias'
    },
    'linear_bias': {
        'q_bias', 'k_bias', 'v_bias', 'attn_proj_bias',
        'mlp_fc1_bias', 'mlp_fc2_bias'
    },
    'ln_bias': {'ln_bias'},
    'attention_bias': {'q_bias', 'k_bias', 'v_bias', 'attn_proj_bias'},
    'qkv_bias': {'q_bias', 'k_bias', 'v_bias'},
    'q_bias': {'q_bias'},
    'k_bias': {'k_bias'},
    'v_bias': {'v_bias'},
    'attn_proj_bias': {'attn_proj_bias'},
    'mlp_bias': {'mlp_fc1_bias', 'mlp_fc2_bias'},
    'mlp_fc1_bias': {'mlp_fc1_bias'},
    'mlp_fc2_bias': {'mlp_fc2_bias'},
}


@dataclass
class TuningConfig:
    """
    Configuration for CLIP/ViT parameter-efficient bias fine-tuning.
    """
    parameter_type: str = "all_bias"
    layer_range: Union[str, List[int], Tuple[int, ...]] = "all"
    train_classifier: bool = True
    include_pre_post_ln: bool = True  # For pre_layrnorm and post_layernorm

    def __post_init__(self):
        # Normalize and validate parameter_type
        self.parameter_type = self.parameter_type.lower()
        if self.parameter_type not in VALID_PARAMETER_TYPES:
            raise ValueError(
                f"Unknown parameter_type: '{self.parameter_type}'. "
                f"Must be one of: {VALID_PARAMETER_TYPES}"
            )

        # Normalize layer_range if string
        if isinstance(self.layer_range, str):
            self.layer_range = self.layer_range.lower()

    @classmethod
    def from_string(cls, strategy_str: str, train_classifier: bool = True) -> "TuningConfig":
        """
        Create TuningConfig from shorthand strategy strings.
        Examples:
          - "frozen" -> parameter_type="frozen", layer_range="all"
          - "all_bias" -> parameter_type="all_bias", layer_range="all"
          - "v_bias" -> parameter_type="v_bias", layer_range="all"
          - "bias_early" -> parameter_type="all_bias", layer_range="early"
          - "bias_middle" -> parameter_type="all_bias", layer_range="middle"
          - "bias_late" -> parameter_type="all_bias", layer_range="late"
          - "v_bias+late" -> parameter_type="v_bias", layer_range="late"
          - "mlp_bias_late" -> parameter_type="mlp_bias", layer_range="late"
          - "qkv_bias+middle" -> parameter_type="qkv_bias", layer_range="middle"
          - "linear_bias+16-23" -> parameter_type="linear_bias", layer_range="16-23"
        """
        s = strategy_str.strip().lower()

        # Handle layer-wise bias shortcuts: bias_early, bias_middle, bias_late
        if s == "bias_early":
            return cls(parameter_type="all_bias", layer_range="early", train_classifier=train_classifier)
        elif s == "bias_middle":
            return cls(parameter_type="all_bias", layer_range="middle", train_classifier=train_classifier)
        elif s == "bias_late":
            return cls(parameter_type="all_bias", layer_range="late", train_classifier=train_classifier)

        # Check for delimiter '+' or '/'
        delimiter = None
        if '+' in s:
            delimiter = '+'
        elif '/' in s:
            delimiter = '/'

        if delimiter is not None:
            parts = s.split(delimiter, 1)
            param_type = parts[0].strip()
            layer_rng = parts[1].strip()
            return cls(parameter_type=param_type, layer_range=layer_rng, train_classifier=train_classifier)

        # Check for pattern like <param_type>_<early|middle|late>
        for suffix in ['_early', '_middle', '_late']:
            if s.endswith(suffix):
                prefix = s[:-len(suffix)]
                if prefix in VALID_PARAMETER_TYPES:
                    rng = suffix[1:]
                    return cls(parameter_type=prefix, layer_range=rng, train_classifier=train_classifier)

        # Standard parameter type with all layers
        if s in VALID_PARAMETER_TYPES:
            return cls(parameter_type=s, layer_range="all", train_classifier=train_classifier)

        raise ValueError(
            f"Unrecognized tuning strategy string: '{strategy_str}'. "
            f"Valid base types: {VALID_PARAMETER_TYPES}, "
            f"or combinations like 'v_bias+late', 'bias_early', etc."
        )


def get_encoder_layers(backbone: nn.Module) -> Tuple[Optional[nn.ModuleList], int]:
    """
    Dynamically extract transformer block layers from the backbone without hardcoding layer count.
    Supports HuggingFace CLIP, OpenAI CLIP, and timm ViT architectures.
    """
    # 1. HuggingFace CLIP Vision Transformer: backbone.encoder.layers
    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
        layers = backbone.encoder.layers
        return layers, len(layers)

    # 2. OpenAI CLIP: backbone.transformer.resblocks
    if hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'resblocks'):
        layers = backbone.transformer.resblocks
        return layers, len(layers)

    # 3. timm ViT: backbone.blocks
    if hasattr(backbone, 'blocks'):
        layers = backbone.blocks
        return layers, len(layers)

    # 4. Fallback: backbone.layers
    if hasattr(backbone, 'layers'):
        layers = backbone.layers
        return layers, len(layers)

    return None, 0


def resolve_layer_range(layer_range: Union[str, List[int], Tuple[int, ...], Set[int]], total_layers: int) -> Set[int]:
    """
    Compute set of active layer indices based on total_layers and layer_range specification.
    """
    if total_layers <= 0:
        return set()

    if isinstance(layer_range, (list, tuple, set)):
        return set(int(i) for i in layer_range if 0 <= int(i) < total_layers)

    if isinstance(layer_range, str):
        s = layer_range.strip().lower()
        if s == "all":
            return set(range(total_layers))

        third = total_layers // 3
        if s == "early":
            return set(range(0, third))
        elif s == "middle":
            return set(range(third, 2 * third))
        elif s == "late":
            return set(range(2 * third, total_layers))

        # Handle range strings like "0-7" or "16-23" or comma-separated "0,1,2"
        if '-' in s:
            start_str, end_str = s.split('-', 1)
            start, end = int(start_str.strip()), int(end_str.strip())
            return set(range(start, min(end + 1, total_layers)))
        elif ',' in s:
            return set(int(x.strip()) for x in s.split(',') if 0 <= int(x.strip()) < total_layers)
        elif s.isdigit():
            idx = int(s)
            return {idx} if 0 <= idx < total_layers else set()

    raise ValueError(f"Unable to resolve layer range '{layer_range}' for total_layers={total_layers}")


def extract_layer_index(param_name: str) -> Optional[int]:
    """
    Extract the transformer block layer index from parameter name using regex patterns.
    """
    # Patterns for different ViT implementations
    patterns = [
        r'(?:encoder\.layers|layers|resblocks|blocks)\.(\d+)\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, param_name)
        if match:
            return int(match.group(1))
    return None


def categorize_parameter(param_name: str) -> str:
    """
    Categorize a parameter into one of the bias categories or 'weight' / 'other'.
    """
    # Check if parameter is a weight or embedding matrix
    if param_name.endswith('.weight') or 'weight' in param_name or 'embedding' in param_name:
        return 'weight'

    # Check for LayerNorm bias (beta)
    if 'layer_norm' in param_name or 'layernorm' in param_name or 'norm' in param_name or 'ln_' in param_name:
        if param_name.endswith('.bias') or 'bias' in param_name:
            return 'ln_bias'

    # Attention Q, K, V biases
    if ('q_proj' in param_name or '.q.' in param_name or 'query' in param_name) and param_name.endswith('.bias'):
        return 'q_bias'
    if ('k_proj' in param_name or '.k.' in param_name or 'key' in param_name) and param_name.endswith('.bias'):
        return 'k_bias'
    if ('v_proj' in param_name or '.v.' in param_name or 'value' in param_name) and param_name.endswith('.bias'):
        return 'v_bias'

    # Attention out projection bias
    if ('out_proj' in param_name or 'attn.proj' in param_name) and param_name.endswith('.bias'):
        return 'attn_proj_bias'

    # Combined in_proj_bias (used in PyTorch MultiheadAttention / OpenAI CLIP)
    if 'in_proj_bias' in param_name:
        return 'in_proj_bias'

    # MLP biases
    if ('mlp.fc1' in param_name or 'c_fc' in param_name or 'mlp.0' in param_name or 'linear1' in param_name) and param_name.endswith('.bias'):
        return 'mlp_fc1_bias'
    if ('mlp.fc2' in param_name or 'c_proj' in param_name or 'mlp.2' in param_name or 'linear2' in param_name) and param_name.endswith('.bias'):
        return 'mlp_fc2_bias'

    if param_name.endswith('.bias'):
        return 'other_bias'

    return 'other'


def apply_in_proj_mask_hook(param: nn.Parameter, active_segments: Set[str], D: int) -> None:
    """
    Registers a backward hook on packed in_proj_bias to zero out gradients for non-selected Q/K/V segments.
    active_segments: subset of {'q', 'k', 'v'}
    D: feature dimension for single projection (in_proj_bias shape is 3*D)
    """
    def hook(grad):
        if grad is None:
            return None
        mask = torch.zeros_like(grad)
        if 'q' in active_segments:
            mask[0:D] = 1.0
        if 'k' in active_segments:
            mask[D:2*D] = 1.0
        if 'v' in active_segments:
            mask[2*D:3*D] = 1.0
        return grad * mask

    param.register_hook(hook)


def apply_bias_tuning(
    model: nn.Module,
    backbone: nn.Module,
    classifier_head: nn.Module,
    tuning_config: TuningConfig
) -> Dict[str, Any]:
    """
    Configures requires_grad for all parameters in the backbone and classifier head
    according to the specified TuningConfig.
    
    Returns tuning summary metadata dictionary.
    """
    # 1. Resolve encoder layers and active layer range
    _, total_layers = get_encoder_layers(backbone)
    active_layers = resolve_layer_range(tuning_config.layer_range, total_layers)
    active_categories = CATEGORY_MAPPING[tuning_config.parameter_type]

    logger.info(
        f"Applying bias tuning: strategy='{tuning_config.parameter_type}', "
        f"layer_range='{tuning_config.layer_range}' (active layers: {sorted(list(active_layers))}, total: {total_layers})"
    )

    # Special case: Full fine-tuning
    if tuning_config.parameter_type == 'full':
        for p in backbone.parameters():
            p.requires_grad = True
        for p in classifier_head.parameters():
            p.requires_grad = tuning_config.train_classifier
        return print_trainable_parameters(model, tuning_config)

    # 2. Freeze entire backbone initially
    for p in backbone.parameters():
        p.requires_grad = False

    # 3. Configure backbone parameters
    for name, param in backbone.named_parameters():
        category = categorize_parameter(name)
        layer_idx = extract_layer_index(name)

        # Weights are strictly frozen
        if category == 'weight':
            param.requires_grad = False
            continue

        # Check if parameter is outside encoder layers (e.g. pre_layrnorm, post_layernorm)
        if layer_idx is None:
            if category == 'ln_bias' and tuning_config.include_pre_post_ln:
                # Pre/post LN bias is only trainable when tuning ln_bias or all_bias with range='all'
                if tuning_config.layer_range == 'all' and 'ln_bias' in active_categories:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
            continue

        # Parameter is inside an encoder layer
        if layer_idx not in active_layers:
            param.requires_grad = False
            continue

        # Handle packed in_proj_bias if present
        if category == 'in_proj_bias':
            D = param.numel() // 3
            needed_segments = set()
            if 'q_bias' in active_categories:
                needed_segments.add('q')
            if 'k_bias' in active_categories:
                needed_segments.add('k')
            if 'v_bias' in active_categories:
                needed_segments.add('v')

            if len(needed_segments) == 0:
                param.requires_grad = False
            elif len(needed_segments) == 3:
                param.requires_grad = True
            else:
                param.requires_grad = True
                apply_in_proj_mask_hook(param, needed_segments, D)
            continue

        # Handle standard separate bias parameters
        if category in active_categories:
            param.requires_grad = True
        elif category == 'other_bias' and tuning_config.parameter_type in ('all_bias', 'all_bias_subspace'):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 4. Configure classifier head
    for p in classifier_head.parameters():
        p.requires_grad = tuning_config.train_classifier

    # 5. Generate and return metadata
    metadata = print_trainable_parameters(model, tuning_config)
    return metadata


def print_trainable_parameters(model: nn.Module, tuning_config: Optional[TuningConfig] = None) -> Dict[str, Any]:
    """
    Computes, formats and prints detailed trainable parameter statistics.
    Returns metadata dict for experiment tracking.
    """
    trainable_backbone_names = []
    trainable_classifier_names = []

    # Category counts
    category_counts = {
        'Attention Q bias': 0,
        'Attention K bias': 0,
        'Attention V bias': 0,
        'Attention proj bias': 0,
        'MLP bias': 0,
        'LayerNorm beta': 0,
        'Classifier': 0,
        'Other': 0,
    }

    trainable_backbone_params = 0
    trainable_classifier_params = 0
    total_model_params = 0

    # Determine classifier parameter ids
    classifier_param_ids = set()
    if hasattr(model, 'head'):
        classifier_param_ids = set(id(p) for p in model.head.parameters())

    for name, param in model.named_parameters():
        n_elem = param.numel()
        total_model_params += n_elem

        if not param.requires_grad:
            continue

        is_classifier = id(param) in classifier_param_ids or 'head' in name

        if is_classifier:
            trainable_classifier_params += n_elem
            trainable_classifier_names.append(name)
            category_counts['Classifier'] += n_elem
        else:
            category = categorize_parameter(name)

            # Handle packed in_proj_bias fractional counting if hook applied
            if category == 'in_proj_bias' and tuning_config is not None:
                active_cats = CATEGORY_MAPPING[tuning_config.parameter_type]
                D = n_elem // 3
                active_segs = 0
                if 'q_bias' in active_cats:
                    category_counts['Attention Q bias'] += D
                    active_segs += 1
                if 'k_bias' in active_cats:
                    category_counts['Attention K bias'] += D
                    active_segs += 1
                if 'v_bias' in active_cats:
                    category_counts['Attention V bias'] += D
                    active_segs += 1
                effective_count = active_segs * D
                trainable_backbone_params += effective_count
                trainable_backbone_names.append(f"{name} (masked: {active_segs}/3 segments)")
            else:
                trainable_backbone_params += n_elem
                trainable_backbone_names.append(name)

                if category == 'q_bias':
                    category_counts['Attention Q bias'] += n_elem
                elif category == 'k_bias':
                    category_counts['Attention K bias'] += n_elem
                elif category == 'v_bias':
                    category_counts['Attention V bias'] += n_elem
                elif category == 'attn_proj_bias':
                    category_counts['Attention proj bias'] += n_elem
                elif category in ('mlp_fc1_bias', 'mlp_fc2_bias'):
                    category_counts['MLP bias'] += n_elem
                elif category == 'ln_bias':
                    category_counts['LayerNorm beta'] += n_elem
                else:
                    category_counts['Other'] += n_elem

    trainable_total_params = trainable_backbone_params + trainable_classifier_params
    trainable_ratio = (trainable_total_params / total_model_params) if total_model_params > 0 else 0.0

    # Format output according to prompt specifications
    print("=" * 60)
    print("Trainable parameters:")
    for n in trainable_backbone_names + trainable_classifier_names:
        print(f"  {n}")
    print("-" * 60)
    print(f"Trainable backbone params: {trainable_backbone_params:,}")
    print(f"Trainable classifier params: {trainable_classifier_params:,}")
    print(f"Total trainable params: {trainable_total_params:,}")
    print(f"Total model params: {total_model_params:,}")
    print(f"Trainable ratio: {trainable_ratio * 100:.4f} %")
    print("-" * 60)
    print(f"Attention Q bias: {category_counts['Attention Q bias']:,}")
    print(f"Attention K bias: {category_counts['Attention K bias']:,}")
    print(f"Attention V bias: {category_counts['Attention V bias']:,}")
    print(f"Attention proj bias: {category_counts['Attention proj bias']:,}")
    print(f"MLP bias: {category_counts['MLP bias']:,}")
    print(f"LayerNorm beta: {category_counts['LayerNorm beta']:,}")
    print(f"Classifier: {category_counts['Classifier']:,}")
    print("=" * 60)

    strategy_name = tuning_config.parameter_type if tuning_config else "custom"
    layer_rng_name = str(tuning_config.layer_range) if tuning_config else "all"

    metadata = {
        "tuning_strategy": strategy_name,
        "layer_range": layer_rng_name,
        "trainable_backbone_params": trainable_backbone_params,
        "trainable_classifier_params": trainable_classifier_params,
        "trainable_total_params": trainable_total_params,
        "total_model_params": total_model_params,
        "trainable_ratio": round(trainable_ratio, 6),
        "category_breakdown": category_counts,
    }
    return metadata


def verify_tuning_configuration(
    model: nn.Module,
    backbone: nn.Module,
    classifier_head: nn.Module,
    tuning_config: TuningConfig,
    run_grad_check: bool = True
) -> bool:
    """
    Rigorously validates that ONLY intended parameters have requires_grad=True,
    and optionally executes a dummy forward/backward step to verify gradient isolation.
    """
    _, total_layers = get_encoder_layers(backbone)
    active_layers = resolve_layer_range(tuning_config.layer_range, total_layers)
    active_categories = CATEGORY_MAPPING[tuning_config.parameter_type]

    # --- Check 1: ALL backbone weight matrices MUST be frozen ---
    for name, param in backbone.named_parameters():
        category = categorize_parameter(name)
        if category == 'weight':
            assert not param.requires_grad, (
                f"[Sanity Check Failed] Backbone weight matrix '{name}' must have requires_grad=False!"
            )

    # --- Check 2: ALL LayerNorm weights (gamma) MUST be frozen ---
    for name, param in backbone.named_parameters():
        if ('layer_norm' in name or 'layernorm' in name or 'norm' in name) and param.ndim == 1:
            if name.endswith('.weight'):
                assert not param.requires_grad, (
                    f"[Sanity Check Failed] LayerNorm gamma '{name}' must have requires_grad=False!"
                )

    # --- Check 3: Parameters in inactive layers MUST be frozen ---
    for name, param in backbone.named_parameters():
        layer_idx = extract_layer_index(name)
        if layer_idx is not None and layer_idx not in active_layers:
            assert not param.requires_grad, (
                f"[Sanity Check Failed] Parameter '{name}' is in inactive layer {layer_idx} but requires_grad=True!"
            )

    # --- Check 4: Specific strategy assertions ---
    if tuning_config.parameter_type == 'frozen':
        for name, param in backbone.named_parameters():
            assert not param.requires_grad, (
                f"[Sanity Check Failed] Strategy 'frozen' requires all backbone params frozen, but '{name}' is trainable!"
            )

    elif tuning_config.parameter_type == 'ln_bias':
        for name, param in backbone.named_parameters():
            cat = categorize_parameter(name)
            if cat in ('q_bias', 'k_bias', 'v_bias', 'attn_proj_bias', 'mlp_fc1_bias', 'mlp_fc2_bias'):
                assert not param.requires_grad, (
                    f"[Sanity Check Failed] Strategy 'ln_bias' requires linear bias '{name}' to be frozen!"
                )

    elif tuning_config.parameter_type == 'linear_bias':
        for name, param in backbone.named_parameters():
            cat = categorize_parameter(name)
            if cat == 'ln_bias':
                assert not param.requires_grad, (
                    f"[Sanity Check Failed] Strategy 'linear_bias' requires LayerNorm bias '{name}' to be frozen!"
                )

    elif tuning_config.parameter_type == 'v_bias':
        for name, param in backbone.named_parameters():
            cat = categorize_parameter(name)
            if cat in ('q_bias', 'k_bias', 'attn_proj_bias', 'mlp_fc1_bias', 'mlp_fc2_bias', 'ln_bias'):
                assert not param.requires_grad, (
                    f"[Sanity Check Failed] Strategy 'v_bias' requires parameter '{name}' ({cat}) to be frozen!"
                )

    # --- Check 5: Dynamic gradient verification (Dummy forward/backward) ---
    if run_grad_check:
        was_training = model.training
        model.train()

        # Determine device
        device = next(model.parameters()).device
        dummy_img = torch.randn(2, 3, 224, 224, device=device)
        dummy_data = {'image': dummy_img, 'label': torch.tensor([0, 1], device=device)}

        model.zero_grad()
        pred_dict = model(dummy_data)
        loss_dict = model.get_losses(dummy_data, pred_dict)
        loss = loss_dict['overall']
        loss.backward()

        # Check gradients
        for name, param in backbone.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, (
                    f"[Sanity Check Failed] Frozen parameter '{name}' received non-None gradient!"
                )
            else:
                assert param.grad is not None, (
                    f"[Sanity Check Failed] Trainable parameter '{name}' has grad=None after backward!"
                )
                cat = categorize_parameter(name)
                # Verify that active bias actually receives non-zero gradient
                grad_norm = param.grad.abs().sum().item()
                assert grad_norm > 0.0, (
                    f"[Sanity Check Failed] Trainable parameter '{name}' received 0 gradient!"
                )

        model.zero_grad()
        if not was_training:
            model.eval()

    logger.info("✅ Sanity check passed: Parameter selection and gradient isolation fully verified.")
    return True
