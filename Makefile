# ==============================================================================
# Makefile: Automated CLI Workflow for CLIP Bias PEFT Deepfake Detection
# ==============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Environment configuration
export PATH := /opt/conda/bin:$(PATH)
export LD_LIBRARY_PATH := /opt/conda/lib:$(LD_LIBRARY_PATH)

# Default experiment arguments
GPUS           ?= 2
STRATEGY       ?= all_bias
CONFIG         ?= ./training/config/detector/clip_bias.yaml
TRAIN_DATA     ?= FaceForensics++
TEST_DATA      ?= Celeb-DF-v2
LAYER_RANGE    ?= 
WEIGHTS        ?= 
PORT           ?= $(shell python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()') 

# Extra args builder
EXTRA_ARGS := 
ifneq ($(LAYER_RANGE),)
	EXTRA_ARGS += --layer_range $(LAYER_RANGE)
endif

# ------------------------------------------------------------------------------
# Help Menu
# ------------------------------------------------------------------------------
.PHONY: help
help:
	@echo "========================================================================"
	@echo "  CLIP Bias PEFT Deepfake Detection - Automation Suite"
	@echo "========================================================================"
	@echo "Available Targets:"
	@echo ""
	@echo "  make train           Train single GPU with STRATEGY (default: all_bias)"
	@echo "                       Usage: make train STRATEGY=v_bias"
	@echo ""
	@echo "  make train-ddp       Train on multi-GPU DDP (default: 2 GPUs)"
	@echo "                       Usage: make train-ddp STRATEGY=v_bias GPUS=2"
	@echo "                       Usage: make train-ddp STRATEGY=v_bias LAYER_RANGE=late"
	@echo ""
	@echo "  Shorthand Multi-GPU Targets (2x GPUs DDP):"
	@echo "    make train-full         - Full Fine-Tuning (100% backbone)"
	@echo "    make train-all-bias     - All bias parameters (0.09%)"
	@echo "    make train-v-bias       - Value bias only (0.0088%)"
	@echo "    make train-ln-bias      - LayerNorm beta only (0.0176%)"
	@echo "    make train-linear-bias  - All Linear biases (0.0736%)"
	@echo "    make train-late-bias    - Late blocks 16-23 (0.0304%)"
	@echo "    make train-v-late       - Value bias in blocks 16-23 (0.0034%)"
	@echo ""
	@echo "  Testing / Evaluation:"
	@echo "    make test          Evaluate model on test dataset"
	@echo "                       Usage: make test WEIGHTS=/path/to/best.pth"
	@echo "    make test-ddp      Evaluate with 2-GPU torchrun"
	@echo ""
	@echo "  Verification & Utilities:"
	@echo "    make verify        Run gradient isolation unit test on all 19 strategies"
	@echo "    make clean         Remove python cache and temporary files"
	@echo "========================================================================"

# ------------------------------------------------------------------------------
# Training Targets
# ------------------------------------------------------------------------------
.PHONY: train
train:
	@echo ">>> Running Single-GPU training with strategy='$(STRATEGY)'..."
	python3 training/train.py \
		--detector_path $(CONFIG) \
		--train_dataset "$(TRAIN_DATA)" \
		--test_dataset "$(TEST_DATA)" \
		--tuning "$(STRATEGY)" \
		$(EXTRA_ARGS)

.PHONY: train-ddp
train-ddp:
	@echo ">>> Running $(GPUS)-GPU DDP training with strategy='$(STRATEGY)' on port $(PORT)..."
	torchrun --nproc_per_node=$(GPUS) --master_port=$(PORT) training/train.py \
		--detector_path $(CONFIG) \
		--train_dataset "$(TRAIN_DATA)" \
		--test_dataset "$(TEST_DATA)" \
		--tuning "$(STRATEGY)" \
		--ddp \
		$(EXTRA_ARGS)

# ------------------------------------------------------------------------------
# Shorthand 2-GPU DDP Training Targets
# ------------------------------------------------------------------------------
.PHONY: train-full
train-full:
	$(MAKE) train-ddp STRATEGY=full GPUS=2

.PHONY: train-all-bias
train-all-bias:
	$(MAKE) train-ddp STRATEGY=all_bias GPUS=2

.PHONY: train-v-bias
train-v-bias:
	$(MAKE) train-ddp STRATEGY=v_bias GPUS=2

.PHONY: train-ln-bias
train-ln-bias:
	$(MAKE) train-ddp STRATEGY=ln_bias GPUS=2

.PHONY: train-linear-bias
train-linear-bias:
	$(MAKE) train-ddp STRATEGY=linear_bias GPUS=2

.PHONY: train-late-bias
train-late-bias:
	$(MAKE) train-ddp STRATEGY=bias_late GPUS=2

.PHONY: train-v-late
train-v-late:
	$(MAKE) train-ddp STRATEGY=v_bias LAYER_RANGE=late GPUS=2

# ------------------------------------------------------------------------------
# Testing Targets
# ------------------------------------------------------------------------------
.PHONY: test
test:
	@if [ -z "$(WEIGHTS)" ]; then \
		echo "Error: WEIGHTS parameter required. Example: make test WEIGHTS=./logs/.../ckpt_best.pth"; \
		exit 1; \
	fi
	@echo ">>> Running testing with weights='$(WEIGHTS)'..."
	python3 training/test.py \
		--detector_path $(CONFIG) \
		--test_dataset "$(TEST_DATA)" \
		--weights_path "$(WEIGHTS)"

.PHONY: test-ddp
test-ddp:
	@if [ -z "$(WEIGHTS)" ]; then \
		echo "Error: WEIGHTS parameter required. Example: make test-ddp WEIGHTS=./logs/.../ckpt_best.pth"; \
		exit 1; \
	fi
	@echo ">>> Running $(GPUS)-GPU DDP testing with weights='$(WEIGHTS)' on port $(PORT)..."
	torchrun --nproc_per_node=$(GPUS) --master_port=$(PORT) training/test.py \
		--detector_path $(CONFIG) \
		--test_dataset "$(TEST_DATA)" \
		--weights_path "$(WEIGHTS)"

# ------------------------------------------------------------------------------
# Verification & Diagnostics
# ------------------------------------------------------------------------------
.PHONY: verify
verify:
	@echo ">>> Running comprehensive PEFT verification suite..."
	python3 test_bias_tuning.py

.PHONY: clean
clean:
	@echo ">>> Cleaning python caches and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo ">>> Clean done."
