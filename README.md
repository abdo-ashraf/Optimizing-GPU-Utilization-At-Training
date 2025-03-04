# Optimizing GPU Utilization in Deep Learning Training

## 🚀 Project Overview

This repository provides a comprehensive exploration of GPU optimization techniques for PyTorch models, focusing on improving training efficiency and performance. By implementing and comparing various optimization strategies, the project offers practical insights into enhancing deep learning training workflows.

## 📋 Table of Contents

1. [Hardware Specifications](#-hardware-specifications)
2. [Project Structure](#-project-structure)
3. [Installation](#-installation)
4. [Usage](#-usage)
5. [Optimization Techniques](#-optimizations-implemented)
6. [Performance Considerations](#-performance-considerations)
7. [Experimental Results](#-experimental-results)
8. [References](#-references)
9. [License](#-license)
10. [Contributing](#-contributing)

## 💻 Hardware Specifications

**Experimental Environment:**
- **GPU**: NVIDIA RTX 3050 Ti (4GB VRAM)
- **CPU**: Intel Core i5-11400H
- **RAM**: 16GB

## 🗂 Project Structure

### Notebooks
- `Optimizing_GPU_Utilization.ipynb`: contains a full explanation for each optimization implemented.

### Key Scripts
- `no_optimization.py`: Baseline implementation without optimizations
- `tensorFloat32.py`: TensorFloat-32 (TF32) precision optimization
- `brainFloat16.py`: BFloat16 precision optimization
- `torch_compile.py`: Torch JIT compilation optimization
- `flash_attention.py`: FlashAttention implementation
- `fused_optimizer.py`: Fused optimizer optimization
- `8-bit_optimizer.py`: 8-bit Adam optimizer for reduced memory usage

### Utility Components
- `Utils/`: Contains model and data setup utilities
- `Makefile`: Automation script for running experiments
- `requirements.txt`: Project dependencies

## 🔧 Installation

### Prerequisites
- Python 3.12+
- CUDA-enabled GPU
- pip package manager

### Dependencies Installation
```bash
pip install -r requirements.txt
```

## 🚀 Usage

This project includes a Makefile that simplifies running experiments and generating comparisons.

### Mandatory Parameters
When running experiments, you must specify three mandatory parameters:

- `STEPS=n`: Number of training steps to perform
- `BATCH_SIZE=b`: Size of each training batch
- `PREFIX=path`: Output directory for results and plots

### Running Individual Optimization Techniques
```bash
make baseline STEPS=50 BATCH_SIZE=256 PREFIX=./out         # No optimization
make tf32 STEPS=50 BATCH_SIZE=256 PREFIX=./out             # TensorFloat32
make bf16 STEPS=50 BATCH_SIZE=256 PREFIX=./out             # BrainFloat16
make torch_compile STEPS=50 BATCH_SIZE=256 PREFIX=./out    # Torch Compile
make flash STEPS=50 BATCH_SIZE=256 PREFIX=./out            # FlashAttention
make fused STEPS=50 BATCH_SIZE=256 PREFIX=./out            # Fused Optimizer
make 8bit STEPS=50 BATCH_SIZE=256 PREFIX=./out             # 8-bit Optimizer
```

### Generate Comparison Plots

After running one or more experiments:

```bash
make plots STEPS=50 BATCH_SIZE=256 PREFIX=./out
```

### Running All Optimizations and Generate Comparison Plots
```bash
make all STEPS=50 BATCH_SIZE=256 PREFIX=./out
```

### Additional Commands

```bash
make help
make reset            # Reset results file and plots
make clean            # Remove generated files
make init_results     # Initialize results.csv file at `RESULTS_FILE` given path
```

## 🔬 Optimizations Implemented

1. **No Optimization**: Baseline implementation
2. **TensorFloat-32 (TF32)**: 
   - Improved precision for matrix multiplications
   - Balanced performance and accuracy
3. **BrainFloat16 (BF16)**:
   - Reduced memory usage
   - Faster training on supported hardware
4. **Torch Compile**:
   - Just-in-time (JIT) compilation
   - Reduced overhead
5. **FlashAttention**:
   - Efficient attention mechanism
   - Improved performance for transformer models
6. **Fused Optimizer**:
   - Reduced GPU kernel launches
   - Enhanced training efficiency
7. **8-bit Optimizer**:
   - Reduced memory footprint
   - Potential training speed improvement

## 📊 Performance Considerations

- Choose optimization techniques based on your specific hardware and model architecture
- Some techniques may have compilation overhead
- Performance gains vary depending on model complexity and hardware

## 📈 Experimental Results

### Mean Relative Speedup Comparison

The following plot shows the mean relative speedup comparison for different optimization techniques compared to the baseline (no optimization). These results were generated using a batch size of 256 and 150 training steps. This plot helps in visualizing the performance gains achieved by each optimization method.

![Mean Relative Speedup Comparison](https://github.com/user-attachments/assets/87bfcbde-ae20-45ab-bbce-b03166a1ae4b)

By combining BF16, Torch compile, FlashAttention, and Fused Optimizer, I was able to reduce the average iteration time from 472.88 ms (no optimization) to 159.66 ms, making it ~3× faster! (Excluding compilation steps)

## 📚 References

- [Andrej Karpathy: Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=sBZPAn3O0jxxV0y3)
- [NVIDIA Ampere Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [PyTorch Documentation on set_float32_matmul_precision](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)
- [PyTorch Documentation on Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Documentation on torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [PyTorch Documentation on scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Online softmax Paper](https://arxiv.org/abs/1805.02867)

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

## 🤝 Contributing

Contributions are welcome! Please submit pull requests or open issues to discuss potential improvements.
