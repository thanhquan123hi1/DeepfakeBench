import os
import sys
import torch
import yaml

# Ensure project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "training")
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from training.detectors import DETECTOR
from training.detectors.bias_tuning import (
    TuningConfig,
    apply_bias_tuning,
    print_trainable_parameters,
    verify_tuning_configuration,
)
from training.detectors.clip_bias_detector import CLIPBiasDetector


def run_all_tests():
    print("============================================================")
    print("        STARTING COMPREHENSIVE BIAS TUNING TESTS            ")
    print("============================================================")

    # 1. Initialize detector via DETECTOR registry
    print("\n--- [1] Initializing CLIPBiasDetector via Registry ---")
    config = {
        'model_name': 'clip_bias',
        'tuning': 'all_bias',
        'normalize_features': True,
    }
    model = DETECTOR['clip_bias'](config)
    print("CLIPBiasDetector initialized successfully!")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Test all 16 requested strategies + combined
    strategies_to_test = [
        "frozen",
        "all_bias",
        "linear_bias",
        "ln_bias",
        "attention_bias",
        "qkv_bias",
        "q_bias",
        "k_bias",
        "v_bias",
        "attn_proj_bias",
        "mlp_bias",
        "mlp_fc1_bias",
        "mlp_fc2_bias",
        "bias_early",
        "bias_middle",
        "bias_late",
        "v_bias+late",
        "mlp_bias+early",
        "linear_bias+middle",
    ]

    print("\n--- [2] Testing All Tuning Strategies (Sanity & Gradient Check) ---")
    for strat in strategies_to_test:
        print(f"\n>> Testing strategy: '{strat}'")
        model.set_tuning_strategy(strat, verify=False)
        # Verify statically and with dummy gradient check
        verify_tuning_configuration(
            model=model,
            backbone=model.backbone,
            classifier_head=model.head,
            tuning_config=model.tuning_config,
            run_grad_check=True
        )
        meta = model.tuning_metadata
        print(f"   Trainable backbone: {meta['trainable_backbone_params']:,} | Classifier: {meta['trainable_classifier_params']:,} | Ratio: {meta['trainable_ratio']*100:.4f}%")

    # 3. Print full parameter tables for required reporting strategies
    reporting_strategies = [
        "all_bias",
        "linear_bias",
        "ln_bias",
        "v_bias",
        "mlp_bias",
        "bias_late",
    ]

    print("\n============================================================")
    print("   DETAILED TRAINABLE PARAMETERS FOR REQUIRED STRATEGIES    ")
    print("============================================================")
    for strat in reporting_strategies:
        print(f"\n\n################### STRATEGY: {strat} ###################")
        model.set_tuning_strategy(strat, verify=False)
        print_trainable_parameters(model, model.tuning_config)

    print("\n============================================================")
    print("   ALL TESTS PASSED SUCCESSFULLY! ZERO WEIGHT LEAKAGE!      ")
    print("============================================================")


if __name__ == '__main__':
    run_all_tests()
