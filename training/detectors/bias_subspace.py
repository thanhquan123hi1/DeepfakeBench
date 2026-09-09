# ==============================================================================
# bias_subspace.py: Manipulation-Invariant Bias Subspace (MIBS) Regularization
# ==============================================================================
# Provides canonical parameter ordering, gradient flattening, subspace estimation (SVD / mean),
# O(Pr) projection regularization, and strict artifact serialization/validation.
# ==============================================================================

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Union

import torch
import torch.nn as nn

from .bias_tuning import categorize_parameter, extract_layer_index, TuningConfig

logger = logging.getLogger(__name__)


@dataclass
class BiasParameterSpec:
    """
    Canonical specification of a single bias parameter in the backbone.
    Tracks name, tensor shape, element count, and slice offsets in the flattened vector.
    """
    name: str
    shape: Tuple[int, ...]
    numel: int
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BiasParameterSpec":
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            numel=d["numel"],
            start=d["start"],
            end=d["end"],
        )


def get_ordered_bias_specs(
    backbone: nn.Module,
    tuning_config: Optional[TuningConfig] = None
) -> List[BiasParameterSpec]:
    """
    Extracts canonical ordered list of trainable bias parameter specifications from backbone.
    Reuses the exact categorization and filtering logic of 'all_bias'.
    
    Returns:
        List of BiasParameterSpec sorted by canonical backbone traversal order.
    """
    if tuning_config is None:
        tuning_config = TuningConfig(parameter_type="all_bias", layer_range="all")

    specs: List[BiasParameterSpec] = []
    current_offset = 0

    for name, param in backbone.named_parameters():
        category = categorize_parameter(name)
        layer_idx = extract_layer_index(name)

        # Exclude all weights
        if category == 'weight':
            continue

        # Check if pre/post layernorm bias
        if layer_idx is None:
            if category == 'ln_bias' and tuning_config.include_pre_post_ln:
                if tuning_config.layer_range == 'all':
                    numel = param.numel()
                    specs.append(BiasParameterSpec(
                        name=name,
                        shape=tuple(param.shape),
                        numel=numel,
                        start=current_offset,
                        end=current_offset + numel
                    ))
                    current_offset += numel
            continue

        # Encoder layer bias parameters
        # For all_bias with range='all', include all linear biases and layernorm biases
        if category in {
            'q_bias', 'k_bias', 'v_bias', 'attn_proj_bias',
            'mlp_fc1_bias', 'mlp_fc2_bias', 'ln_bias', 'other_bias'
        }:
            numel = param.numel()
            specs.append(BiasParameterSpec(
                name=name,
                shape=tuple(param.shape),
                numel=numel,
                start=current_offset,
                end=current_offset + numel
            ))
            current_offset += numel

    logger.debug(f"Resolved {len(specs)} canonical bias parameters with total P={current_offset}")
    return specs


def flatten_bias_tensors(
    source: Union[nn.Module, Dict[str, torch.Tensor]],
    specs: List[BiasParameterSpec],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Flattens bias parameters into a contiguous 1D tensor [P] according to canonical specs.
    Source can be an nn.Module (backbone) or a dict mapping name -> Tensor.
    """
    parts = []
    is_module = isinstance(source, nn.Module)
    param_dict = dict(source.named_parameters()) if is_module else source

    for spec in specs:
        if spec.name not in param_dict:
            raise KeyError(f"Parameter '{spec.name}' not found in source tensors")
        p = param_dict[spec.name]
        if device is not None and p.device != device:
            p = p.to(device)
        parts.append(p.reshape(-1))

    return torch.cat(parts, dim=0)


def restore_bias_tensors(
    flat_vector: torch.Tensor,
    specs: List[BiasParameterSpec]
) -> Dict[str, torch.Tensor]:
    """
    Restores individual bias tensors from a flattened 1D vector [P] according to specs.
    """
    total_p = specs[-1].end if specs else 0
    if flat_vector.numel() != total_p:
        raise ValueError(f"Vector size {flat_vector.numel()} does not match total specs size {total_p}")

    restored = {}
    for spec in specs:
        slice_tensor = flat_vector[spec.start:spec.end]
        restored[spec.name] = slice_tensor.view(spec.shape)

    return restored


def flatten_bias_gradients(
    backbone: nn.Module,
    specs: List[BiasParameterSpec],
    check_none: bool = True
) -> torch.Tensor:
    """
    Extracts and flattens gradients of backbone bias parameters into a 1D tensor [P].
    Fails loudly if any expected parameter gradient is None.
    """
    param_dict = dict(backbone.named_parameters())
    grad_parts = []

    for spec in specs:
        if spec.name not in param_dict:
            raise KeyError(f"Parameter '{spec.name}' not found in backbone")
        param = param_dict[spec.name]
        if param.grad is None:
            if check_none:
                raise ValueError(
                    f"Expected gradient for bias parameter '{spec.name}', but got None. "
                    "Ensure all bias parameters are participating in loss.backward()."
                )
            else:
                grad_parts.append(torch.zeros(spec.numel, device=param.device, dtype=param.dtype))
        else:
            grad_parts.append(param.grad.reshape(-1))

    return torch.cat(grad_parts, dim=0)


def snapshot_initial_bias(
    backbone: nn.Module,
    specs: List[BiasParameterSpec]
) -> Dict[str, torch.Tensor]:
    """
    Creates an immutable snapshot of initial pretrained CLIP bias parameters (b0).
    All tensors are detached, cloned, and set to requires_grad=False.
    """
    param_dict = dict(backbone.named_parameters())
    b0: Dict[str, torch.Tensor] = {}

    for spec in specs:
        if spec.name not in param_dict:
            raise KeyError(f"Parameter '{spec.name}' not found in backbone")
        param = param_dict[spec.name]
        b0[spec.name] = param.detach().clone().requires_grad_(False)

    return b0


def compute_subspace_loss(
    delta_b: torch.Tensor,
    U: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Manipulation-Invariant Bias Subspace (MIBS) regularization loss.
    
    Complexity: O(P * r)
    NEVER instantiates P x P projection matrix or identity matrix.
    
    Formula:
        coeff = U.T @ delta_b            # [r]
        projection = U @ coeff          # [P]
        residual = delta_b - projection # [P]
        loss_subspace = residual.pow(2).mean()
        
    Args:
        delta_b: 1D tensor of shape [P], representing current bias update b - b0.
        U: 2D tensor of shape [P, r] with orthonormal columns (U.T @ U = I_r).
        eps: Small epsilon to prevent division by zero in energy ratios.
        
    Returns:
        (loss_subspace, diagnostics_dict)
    """
    P = delta_b.shape[0]
    if U.shape[0] != P:
        raise ValueError(f"Dimension mismatch: delta_b has length {P}, but U has {U.shape[0]} rows")

    # Efficient O(Pr) projection
    coeff = torch.matmul(U.t(), delta_b)           # [r]
    projection = torch.matmul(U, coeff)            # [P]
    residual = delta_b - projection                # [P]

    # Subspace loss: mean squared residual across P parameters
    loss_subspace = residual.pow(2).mean()

    # Diagnostics
    with torch.no_grad():
        delta_norm = delta_b.norm(2).item()
        proj_norm = projection.norm(2).item()
        res_norm = residual.norm(2).item()
        delta_sq = delta_norm ** 2
        shared_energy = (coeff.pow(2).sum()).item()
        shared_energy_ratio = float(shared_energy / (delta_sq + eps))
        # Clamp to [0, 1] for numerical stability
        shared_energy_ratio = max(0.0, min(1.0, shared_energy_ratio))
        outside_energy_ratio = 1.0 - shared_energy_ratio

    diagnostics = {
        "loss_subspace": loss_subspace.item(),
        "delta_norm": delta_norm,
        "shared_norm": proj_norm,
        "outside_norm": res_norm,
        "shared_energy_ratio": shared_energy_ratio,
        "outside_energy_ratio": outside_energy_ratio,
    }

    return loss_subspace, diagnostics


def compute_gradient_cosine_matrix(
    gradients_dict: Dict[str, torch.Tensor],
    eps: float = 1e-8
) -> Tuple[torch.Tensor, List[str]]:
    """
    Computes pairwise cosine similarity matrix between manipulation method mean gradients.
    """
    methods = list(gradients_dict.keys())
    M = len(methods)
    cosine_matrix = torch.zeros(M, M, dtype=torch.float32)

    for i, m1 in enumerate(methods):
        g1 = gradients_dict[m1]
        norm1 = g1.norm(2) + eps
        for j, m2 in enumerate(methods):
            if i == j:
                cosine_matrix[i, j] = 1.0
            elif j > i:
                g2 = gradients_dict[m2]
                norm2 = g2.norm(2) + eps
                cos_sim = (torch.dot(g1, g2) / (norm1 * norm2)).item()
                cosine_matrix[i, j] = cos_sim
                cosine_matrix[j, i] = cos_sim

    return cosine_matrix, methods


def build_subspace_svd(
    gradients_dict: Dict[str, torch.Tensor],
    rank: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Builds orthonormal shared bias subspace U using thin SVD on normalized method gradients.
    
    Args:
        gradients_dict: Mapping method_name -> normalized mean gradient vector [P].
        rank: Number of top singular directions to retain (r <= M).
        
    Returns:
        (U_shared [P, r], singular_values [M], explained_energy float)
    """
    methods = list(gradients_dict.keys())
    M = len(methods)
    if rank > M:
        raise ValueError(f"Requested rank {rank} exceeds number of manipulation methods {M}")
    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")

    # Stack column vectors: G is [P, M]
    cols = [gradients_dict[m].detach().cpu().to(torch.float32).reshape(-1, 1) for m in methods]
    G = torch.cat(cols, dim=1)  # [P, M]

    # Thin SVD: G = U @ diag(S) @ V.T
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)

    # Retain top r directions
    U_shared = U[:, :rank]

    # Explained energy ratio
    total_energy = (S ** 2).sum().item()
    retained_energy = (S[:rank] ** 2).sum().item()
    explained_energy = float(retained_energy / (total_energy + 1e-12))

    # Verify orthonormality: U.T @ U ≈ I_r
    gram = U_shared.t() @ U_shared
    eye = torch.eye(rank, dtype=U_shared.dtype)
    if not torch.allclose(gram, eye, atol=1e-4):
        logger.warning(f"SVD columns imperfectly orthonormal: max diff = {(gram - eye).abs().max().item()}")

    return U_shared, S, explained_energy


def build_subspace_mean(
    gradients_dict: Dict[str, torch.Tensor],
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Baseline single-direction subspace (r=1) computed as normalized sum of method gradients.
    """
    methods = list(gradients_dict.keys())
    sum_grad = torch.zeros_like(next(iter(gradients_dict.values()))).cpu().to(torch.float32)
    for m in methods:
        sum_grad += gradients_dict[m].detach().cpu().to(torch.float32)

    norm = sum_grad.norm(2) + eps
    d = sum_grad / norm
    U = d.unsqueeze(1)  # [P, 1]
    singular_values = torch.tensor([1.0], dtype=torch.float32)
    explained_energy = 1.0

    return U, singular_values, explained_energy


@dataclass
class SubspaceArtifact:
    """
    Complete serializable container for estimated bias subspace U and diagnostic metadata.
    """
    U: torch.Tensor
    rank: int
    subspace_method: str
    backbone_name: str
    tuning_strategy: str
    parameter_specs: List[Dict[str, Any]]
    total_bias_params: int
    methods: List[str]
    gradient_cosine_matrix: torch.Tensor
    singular_values: torch.Tensor
    explained_energy: float
    batches_per_method: int
    seed: int
    mean_gradients: Optional[Dict[str, torch.Tensor]] = None

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "U": self.U.cpu(),
            "rank": self.rank,
            "subspace_method": self.subspace_method,
            "backbone_name": self.backbone_name,
            "tuning_strategy": self.tuning_strategy,
            "parameter_specs": self.parameter_specs,
            "total_bias_params": self.total_bias_params,
            "methods": self.methods,
            "gradient_cosine_matrix": self.gradient_cosine_matrix.cpu(),
            "singular_values": self.singular_values.cpu(),
            "explained_energy": self.explained_energy,
            "batches_per_method": self.batches_per_method,
            "seed": self.seed,
        }
        if self.mean_gradients:
            data["mean_gradients"] = {k: v.cpu() for k, v in self.mean_gradients.items()}
        torch.save(data, filepath)
        logger.info(f"Subspace artifact saved successfully to {filepath}")

    @classmethod
    def load(
        cls,
        filepath: str,
        validate_specs: Optional[List[BiasParameterSpec]] = None,
        atol: float = 1e-4
    ) -> "SubspaceArtifact":
        """
        Loads subspace artifact from disk with strict validation.
        If validate_specs is provided, verifies parameter names, order, shapes, P, and U orthonormality.
        Fails loudly on any discrepancy.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Subspace artifact file not found: {filepath}")

        data = torch.load(filepath, map_location="cpu")

        # Validate required fields
        required_fields = [
            "U", "rank", "subspace_method", "backbone_name",
            "tuning_strategy", "parameter_specs", "total_bias_params"
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Subspace artifact missing required field: '{field_name}' in {filepath}")

        artifact = cls(
            U=data["U"],
            rank=data["rank"],
            subspace_method=data["subspace_method"],
            backbone_name=data["backbone_name"],
            tuning_strategy=data["tuning_strategy"],
            parameter_specs=data["parameter_specs"],
            total_bias_params=data["total_bias_params"],
            methods=data.get("methods", []),
            gradient_cosine_matrix=data.get("gradient_cosine_matrix", torch.empty(0)),
            singular_values=data.get("singular_values", torch.empty(0)),
            explained_energy=data.get("explained_energy", 1.0),
            batches_per_method=data.get("batches_per_method", 0),
            seed=data.get("seed", 0),
            mean_gradients=data.get("mean_gradients", None)
        )

        # Strict validation against current model specs if requested
        if validate_specs is not None:
            artifact.validate(validate_specs, atol=atol)

        return artifact

    def validate(self, current_specs: List[BiasParameterSpec], atol: float = 1e-4) -> None:
        """
        Strict verification of the loaded subspace against the current model's bias parameter specifications.
        """
        # 1. Check total count
        if len(self.parameter_specs) != len(current_specs):
            raise ValueError(
                f"Subspace validation failed: parameter count mismatch. "
                f"Artifact has {len(self.parameter_specs)} parameters, but current model has {len(current_specs)}"
            )

        # 2. Check each parameter name, order, shape, and numel
        for i, (art_spec, cur_spec) in enumerate(zip(self.parameter_specs, current_specs)):
            if art_spec["name"] != cur_spec.name:
                raise ValueError(
                    f"Subspace validation failed at index {i}: "
                    f"name mismatch ('{art_spec['name']}' != '{cur_spec.name}')"
                )
            if tuple(art_spec["shape"]) != cur_spec.shape:
                raise ValueError(
                    f"Subspace validation failed for '{cur_spec.name}': "
                    f"shape mismatch ({art_spec['shape']} != {cur_spec.shape})"
                )
            if art_spec["numel"] != cur_spec.numel:
                raise ValueError(
                    f"Subspace validation failed for '{cur_spec.name}': "
                    f"numel mismatch ({art_spec['numel']} != {cur_spec.numel})"
                )

        # 3. Check total P
        current_P = current_specs[-1].end if current_specs else 0
        if self.total_bias_params != current_P:
            raise ValueError(
                f"Subspace validation failed: total P mismatch. "
                f"Artifact has {self.total_bias_params}, current model has {current_P}"
            )

        # 4. Check U rows
        if self.U.shape[0] != current_P:
            raise ValueError(
                f"Subspace validation failed: U has {self.U.shape[0]} rows, but model requires P={current_P}"
            )

        # 5. Check U columns
        if self.U.shape[1] != self.rank:
            raise ValueError(
                f"Subspace validation failed: U has {self.U.shape[1]} columns, but rank is {self.rank}"
            )

        # 6. Check orthonormality: U.T @ U ≈ I_r
        gram = self.U.t() @ self.U
        eye = torch.eye(self.rank, dtype=self.U.dtype)
        diff = (gram - eye).abs().max().item()
        if diff > atol:
            raise ValueError(
                f"Subspace validation failed: columns of U are not orthonormal. "
                f"Max absolute deviation |U.T @ U - I| = {diff:.6f} > atol={atol}"
            )

        logger.info(
            f"Subspace validation PASSED: P={current_P}, rank={self.rank}, "
            f"orthonormality error={diff:.2e}"
        )
