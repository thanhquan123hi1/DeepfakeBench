# CLIP Bias Tuning: Parameter-Efficient Fine-Tuning for Deepfake Detection

[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers: 4.29+](https://img.shields.io/badge/Transformers-4.29%2B-yellow.svg)](https://huggingface.co/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **Research Question:** *Which bias parameters in CLIP ViT are responsible for generalizable deepfake detection?*

This repository implements a **Parameter-Efficient Fine-Tuning (PEFT)** benchmark framework for deepfake detection based on **CLIP ViT-L/14**. It enables fine-grained, systematic ablation of individual bias parameter types and layer depths across 24 transformer blocks—without modifying the underlying architecture, data pipeline, loss formulation, or evaluation protocol.

---

## 📌 Key Highlights

- **Fine-Grained Bias Dissection:** Selectively train Q, K, V, Attention Projection, MLP (fc1/fc2), or LayerNorm $\beta$ independently.
- **Layer-wise Depth Ablation:** Seamlessly configure depth ranges (`early`, `middle`, `late`, or arbitrary block intervals) without hardcoded layer counts.
- **Type × Depth Combinability:** Freely combine parameter type and layer range (e.g., `v_bias + late`, `mlp_bias + early`, `linear_bias + middle`).
- **Gradient Isolation & Masking:** Built-in gradient masking hooks for architectures using combined `in_proj_bias`, guaranteeing zero gradient leakage to unselected components.
- **Strict Weight Freezing:** All weight matrices, patch embeddings, and LayerNorm $\gamma$ (scale) parameters are strictly frozen.
- **Zero Drift Compatibility:** 100% plug-and-play with the unified training, validation, and cross-dataset testing pipeline of DeepfakeBench.
- **Automated Experiment Metadata:** Automatically exports `tuning_metadata.json` capturing trainable parameter counts and ratios for automated paper table generation.

---

## 📊 Supported Tuning Strategies

All strategies keep all backbone weight matrices and LayerNorm weights strictly frozen. The classification head ($1024 \times 2$) remains trainable by default.

| Strategy | Active Components | Backbone Params | Trainable % (Total) |
| :--- | :--- | :---: | :---: |
| `frozen` | Only Classifier Head | **0** | **0.0007%** |
| `v_bias` | Value Projection Bias | **24,576** | **0.0088%** |
| `q_bias` | Query Projection Bias | **24,576** | **0.0088%** |
| `k_bias` | Key Projection Bias | **24,576** | **0.0088%** |
| `attn_proj_bias` | Attention Output Projection Bias | **24,576** | **0.0088%** |
| `qkv_bias` | Q + K + V Biases | **73,728** | **0.0250%** |
| `attention_bias` | Q + K + V + Output Projection Biases | **98,304** | **0.0331%** |
| `ln_bias` | LayerNorm $\beta$ (Norm1, Norm2, Pre/Post LN) | **51,200** | **0.0176%** |
| `mlp_fc1_bias` | First Linear Layer Bias in MLP ($4096 \times 24$) | **98,304** | **0.0331%** |
| `mlp_fc2_bias` | Second Linear Layer Bias in MLP ($1024 \times 24$) | **24,576** | **0.0088%** |
| `mlp_bias` | Both fc1 and fc2 Biases in MLP | **122,880** | **0.0412%** |
| `linear_bias` | All Linear Biases (Attention + MLP) | **221,184** | **0.0736%** |
| `all_bias` | All Biases across the entire backbone | **272,384** | **0.0905%** |
| `bias_early` | All Biases in Blocks 0–7 (First 1/3) | **90,112** | **0.0304%** |
| `bias_middle` | All Biases in Blocks 8–15 (Middle 1/3) | **90,112** | **0.0304%** |
| `bias_late` | All Biases in Blocks 16–23 (Last 1/3) | **90,112** | **0.0304%** |
| `v_bias+late` | V Bias in Blocks 16–23 | **8,192** | **0.0034%** |
| `mlp_bias+early` | MLP Biases in Blocks 0–7 | **40,960** | **0.0142%** |
| `linear_bias+middle`| Linear Biases in Blocks 8–15 | **73,728** | **0.0250%** |

*Note: Total model parameters for CLIP ViT-L/14 with binary head = 303,181,826.*

---

## 📁 Repository Structure

```text
.
├── training/
│   ├── detectors/
│   │   ├── bias_tuning.py           # Core PEFT engine, layer resolution, parameter classifier, sanity check
│   │   ├── clip_bias_detector.py    # Detector implementation registered as 'clip_bias'
│   │   ├── gend_detector.py         # GenD baseline detector
│   │   ├── gend_effort_detector.py  # GenD-Effort hybrid detector
│   │   └── ...                      # Other benchmark detectors
│   ├── config/
│   │   ├── detector/
│   │   │   ├── clip_bias.yaml       # Configuration YAML for clip_bias
│   │   │   └── ...
│   │   ├── train_config.yaml        # Global training dataset and label config
│   │   └── test_config.yaml         # Global testing configuration
│   ├── trainer/
│   │   └── trainer.py               # Training loop, DDP, metrics, metadata persistence
│   ├── train.py                     # Main training entry point (with --tuning / --layer_range CLI)
│   └── test.py                      # Multi-dataset evaluation script
├── test_bias_tuning.py              # Automated verification & gradient check suite
├── train.sh                         # Single GPU run script
├── mtrain.sh                        # Multi-GPU run script (torchrun)
└── test.sh / mtest.sh               # Evaluation scripts
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/thanhquan123hi1/DeepfakeBench.git
cd DeepfakeBench

# Install requirements
bash install.sh
```

### 2. Run Sanity & Gradient Isolation Verification

Verify all 19 strategies, asserting that no backbone weights or LayerNorm scales are updated:

```bash
python test_bias_tuning.py
```

### 3. Training with Specific Bias Strategies

You can run experiments using either CLI arguments or by configuring [`training/config/detector/clip_bias.yaml`](training/config/detector/clip_bias.yaml):

```bash
# A. Value-bias only (V-bias)
python training/train.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --tuning v_bias

# B. LayerNorm Beta only
python training/train.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --tuning ln_bias

# C. All Linear biases
python training/train.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --tuning linear_bias

# D. Depth ablation: Late blocks (blocks 16-23)
python training/train.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --tuning bias_late

# E. Combined Type × Depth (V-bias in late blocks)
python training/train.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --tuning v_bias \
  --layer_range late
```

### 4. Multi-GPU Distributed Training (DDP)

Run on 2 GPUs via `torchrun`:

```bash
torchrun --nproc_per_node=2 training/train.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --tuning v_bias \
  --test_dataset "Celeb-DF-v2" \
  --ddp
```

### 5. Evaluation / Testing

Evaluate a trained checkpoint across cross-dataset benchmarks:

```bash
python training/test.py \
  --detector_path ./training/config/detector/clip_bias.yaml \
  --test_dataset "Celeb-DF-v2" "FaceShifter" "DeeperForensics-1.0" \
  --weights_path ./logs/training/clip_bias_run/ckpt_best.pth
```

---

## 📈 Experiment Tracking & Metadata

Whenever a training run starts or a checkpoint is saved, a `tuning_metadata.json` file is automatically recorded:

```json
{
  "tuning_strategy": "v_bias",
  "layer_range": "late",
  "trainable_backbone_params": 8192,
  "trainable_classifier_params": 2050,
  "trainable_total_params": 10242,
  "total_model_params": 303181826,
  "trainable_ratio": 0.000034,
  "category_breakdown": {
    "Attention Q bias": 0,
    "Attention K bias": 0,
    "Attention V bias": 8192,
    "Attention proj bias": 0,
    "MLP bias": 0,
    "LayerNorm beta": 0,
    "Classifier": 2050
  }
}
```

This format makes it trivial to aggregate experimental runs into comparison tables:

| Method / Tuning | Params | Trainable % | FF++ (c23) | CDF-v2 | DFDC | Avg AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Full Fine-Tuning | 303.18M | 100.00% | - | - | - | - |
| Linear Probe (frozen) | 2.05K | 0.0007% | - | - | - | - |
| `all_bias` | 274.43K | 0.0905% | - | - | - | - |
| `linear_bias` | 223.23K | 0.0736% | - | - | - | - |
| `ln_bias` | 53.25K | 0.0176% | - | - | - | - |
| `v_bias` | 26.63K | 0.0088% | - | - | - | - |
| `v_bias + late` | 10.24K | 0.0034% | - | - | - | - |

---

## 📄 License & Attribution

- This benchmark is built upon the unified framework of [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) (NeurIPS 2023 Datasets & Benchmarks Track).
- Code is released under the **CC BY-NC 4.0** license for academic and non-commercial research purposes.
