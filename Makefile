# Makefile for GPU Optimization Experiments

# Default number of steps for each experiment
STEPS ?= 50
BATCH_SIZE ?= 256

# Python interpreter
PYTHON = python3

# Output file names
RESULTS_FILE = results.csv
PLOTSPREFIX = ""

# Define phony targets (targets that aren"t actual files)
.PHONY: all clean plots help baseline tf32 bf16 torch_compile flash fused \
        results init_results 8bit

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
	@echo "  make all              - Run all optimizations sequentially"
	@echo "  make plots             - Generate comparison plot"
	@echo "  make reset            - Reset results file"
	@echo "  make clean            - Remove generated files"
	@echo "  make init_results     - Initialize results.csv file at RESULTS_FILE given path"
	@echo ""
	@echo "Options:"
	@echo "  STEPS=n               - Set number of steps (default: $(STEPS))"
	@echo ""
	@echo "Examples:"
	@echo "  make baseline STEPS=30"
	@echo "  make all STEPS=50"
	@echo "  make compare_pairs"

# Initialize results.csv file with proper structure
init_results:
	@echo "Initializing results.csv with proper structure..."
	@$(PYTHON) -c "import pandas as pd; results = pd.DataFrame({'step': range($(STEPS))}); results.to_csv('./$(RESULTS_FILE)', index=False); print('Results file created with step column of $(STEPS) steps.')"

# Individual optimization targets
baseline:
	@echo "Running baseline (no optimization)..."
	@$(PYTHON) Scripts/no_optimization.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE)

tf32:
	@echo "Running TensorFloat32 optimization..."
	@$(PYTHON) Scripts/tensorFloat32.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE)

bf16:
	@echo "Running BrainFloat16 optimization..."
	@$(PYTHON) Scripts/brainFloat16.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE)

torch_compile:
	@echo "Running torch.compile optimization..."
	@$(PYTHON) Scripts/torch_compile.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE)

flash:
	@echo "Running FlashAttention optimization..."
	@$(PYTHON) Scripts/flash_attention.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE)

fused:
	@echo "Running fused optimizer optimization..."
	@$(PYTHON) Scripts/fused_optimizer.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE)

8bit:
	@echo "Running 8-bit optimizer optimization..."
	@$(PYTHON) Scripts/8-bit_optimizer.py --number_of_steps $(STEPS) --batch_size $(BATCH_SIZE)

# Run all optimizations sequentially
all: init_results baseline tf32 bf16 torch_compile flash fused 8bit

# Generate plot from results
plots:
	@echo "Generating comparison plots..."
	@if [ -f results.csv ]; then \
		$(PYTHON) Scripts/plotting.py --prefix $(PLOTSPREFIX); \
		echo "Plots have been generated: optimization_performance_comparison.png, performance_heatmap.png, and mean_speedup_comparison.png"; \
	else \
		echo "Error: results.csv not found. Run optimizations first."; \
		exit 1; \
	fi

# Reset results file
reset:
	@echo "Resetting results.csv file and plots..."
	@if [ -f $(RESULTS_FILE) ]; then \
		rm -f $(RESULTS_FILE); \
		rm -f optimization_performance_comparison.png; \
		rm -f performance_heatmap.png; \
		rm -f mean_speedup_comparison.png; \
		echo "Results file and plots reset."; \
	else \
		echo "No results file to reset."; \
	fi
	@make init_results

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f $(RESULTS_FILE)
	rm -f optimization_performance_comparison.png
	rm -f performance_heatmap.png
	rm -f mean_speedup_comparison.png
	rm -f benchmark_*_steps_*.png
	@echo "Cleaned up generated files."
	@make init_results