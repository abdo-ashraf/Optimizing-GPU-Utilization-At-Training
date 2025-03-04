# Makefile for GPU Optimization Experiments

# Mandatory variables (no default values)
ifndef STEPS
    $(error STEPS is not set)
endif

ifndef BATCH_SIZE
    $(error BATCH_SIZE is not set)
endif

ifndef PREFIX
    $(error PREFIX is not set)
endif

# Normalize PREFIX to avoid double slashes in paths
PREFIX_CLEAN = $(shell echo $(PREFIX) | sed 's:/*$$::')
DIRPATH = $(PREFIX_CLEAN)/$(BATCH_SIZE)B_$(STEPS)N_experiment/
RESULTS_FILE = $(DIRPATH)$(BATCH_SIZE)B_$(STEPS)N_results.csv
PLOTSPREFIX = $(DIRPATH)$(BATCH_SIZE)B_$(STEPS)N_

# Python interpreter
PYTHON = python3

# Define phony targets (targets that aren't actual files)
.PHONY: all clean plots help baseline tf32 bf16 torch_compile flash fused \
        results init_results 8bit reset

# Default target when just typing "make"
help:
	@echo "GPU Optimization Experiments Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make baseline         - Run baseline (no optimization)"
	@echo "  make tf32             - Run TensorFloat32 optimization"
	@echo "  make bf16             - Run BrainFloat16 optimization"
	@echo "  make torch_compile    - Run torch.compile optimization"
	@echo "  make flash            - Run FlashAttention optimization"
	@echo "  make fused            - Run fused optimizer optimization"
	@echo "  make 8bit             - Run 8-bit optimizer optimization"
	@echo "  make all              - Run all optimizations sequentially"
	@echo "  make plots            - Generate comparison plots"
	@echo "  make reset            - Reset results file and plots"
	@echo "  make clean            - Remove generated files"
	@echo "  make init_results     - Initialize CSV file at given path"
	@echo ""
	@echo "Options:"
	@echo "  STEPS=n               - Set number of steps (mandatory)"
	@echo "  BATCH_SIZE=b          - Set Batch size (mandatory)"
	@echo "  PREFIX=p              - Set output prefix (mandatory)"
	@echo ""
	@echo "Examples:"
	@echo "  make baseline STEPS=30 BATCH_SIZE=64 PREFIX=./results"
	@echo "  make all STEPS=50 BATCH_SIZE=128 PREFIX=/tmp/output"

# Initialize RESULTS_FILE file with proper structure
init_results:
	@echo "Initializing $(RESULTS_FILE) with proper structure..."
	@$(PYTHON) -c "import os; os.makedirs('$(DIRPATH)', exist_ok=True)"
	@$(PYTHON) -c "import pandas as pd; results = pd.DataFrame({'step': range($(STEPS))}); results.to_csv('$(RESULTS_FILE)', index=False); print('Results file created with $(STEPS) steps.')"

# Individual optimization targets
baseline:
	@echo "Running baseline (no optimization)..."
	@$(PYTHON) Scripts/no_optimization.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE) --results_path '$(RESULTS_FILE)'

tf32:
	@echo "Running TensorFloat32 optimization..."
	@$(PYTHON) Scripts/tensorFloat32.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE) --results_path '$(RESULTS_FILE)'

bf16:
	@echo "Running BrainFloat16 optimization..."
	@$(PYTHON) Scripts/brainFloat16.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE) --results_path '$(RESULTS_FILE)'

torch_compile:
	@echo "Running torch.compile optimization..."
	@$(PYTHON) Scripts/torch_compile.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE) --results_path '$(RESULTS_FILE)'

flash:
	@echo "Running FlashAttention optimization..."
	@$(PYTHON) Scripts/flash_attention.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE) --results_path '$(RESULTS_FILE)'

fused:
	@echo "Running fused optimizer optimization..."
	@$(PYTHON) Scripts/fused_optimizer.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE) --results_path '$(RESULTS_FILE)'

8bit:
	@echo "Running 8-bit optimizer optimization..."
	@$(PYTHON) Scripts/8-bit_optimizer.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE) --results_path '$(RESULTS_FILE)'

# Generate plot from results
plots:
	@echo "Generating comparison plots with prefix of $(PLOTSPREFIX)..."
	@if [ -f "$(RESULTS_FILE)" ]; then \
		$(PYTHON) Scripts/plotting.py --results_path '$(RESULTS_FILE)' --prefix '$(PLOTSPREFIX)'; \
		echo "Plots have been generated."; \
	else \
		echo "Error: $(RESULTS_FILE) not found. Run optimizations first."; \
		exit 1; \
	fi

# Run all optimizations sequentially
all: init_results baseline tf32 bf16 torch_compile flash fused 8bit plots

# Reset results file
reset:
	@echo "Resetting results.csv file and plots..."
	@if [ -d "$(DIRPATH)" ]; then \
		rm -f '$(RESULTS_FILE)'; \
		echo "Results file reset."; \
	else \
		echo "No results file to reset."; \
	fi
	@make init_results

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	@if [ -d "$(DIRPATH)" ]; then \
		rm -rf "$(DIRPATH)"; \
		echo "Cleaned up $(DIRPATH)."; \
	else \
		echo "No directory to clean."; \
	fi
