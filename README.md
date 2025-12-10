# CSDS 451 Final Major Project Proposal and Report  
## Dynamic FlashAttention++ – Power-Aware GPU Scheduling for Transformer Attention

**Course:** CSDS 451 – Designing High-Performance Systems for AI  
**Student:** Sanket, Varun  
**Instructor:** Prof. Sanmukh Kuppannagari  
**GPU Used:** NVIDIA H100 NVL MIG 1g.12GB (Case HPC)

---

# Project Proposal

## Project Overview

Dynamic FlashAttention++ is a power-aware, real-time GPU scheduling and optimization framework designed for Transformer attention mechanisms. This project aims to revolutionize the way large Transformer models like GPT and BERT utilize GPUs by integrating FlashAttention’s high-speed memory-efficient kernel with real-time power and performance feedback loops. The result is a system capable of dynamically optimizing kernel execution for maximum performance-per-watt and energy efficiency, without compromising model accuracy.

The idea stems from two challenges in modern AI computing:

1. The quadratic compute cost of self-attention  
2. The high and uneven power draw on GPUs during training and inference  

FlashAttention fixes the first, but not the second. Our system extends FlashAttention with adaptive GPU telemetry-driven control, creating a more sustainable and power-optimized execution environment for LLMs and Transformer workloads.

---

## Objectives

1. Implement baseline attention using scaled dot-product attention in PyTorch.  
2. Recreate FlashAttention (memory-efficient tiled attention) as baseline.  
3. Integrate NVIDIA NVML and nvidia-smi for live GPU telemetry.  
4. Design a dynamic scheduler that adjusts block size, precision, and recompute strategy based on GPU power and temperature.  
5. Measure runtime, throughput, FLOPS, and performance-per-watt.  
6. Demonstrate improved power stability and scalability across different sequence lengths and batch sizes.

---

## Motivation and Problem Statement

Transformers have become foundational in AI, yet their efficiency remains bounded by quadratic attention and GPU power constraints. FlashAttention efficiently handles memory usage, but GPU power fluctuations still waste energy and can throttle performance. With rising datacenter costs and carbon footprint concerns, energy-aware model execution has become a top research priority for NVIDIA, AMD, and cloud AI providers.

Dynamic FlashAttention++ introduces closed-loop GPU control, allowing the kernel to dynamically adjust based on power, temperature, and utilization metrics. It achieves adaptive optimization, similar to dynamic voltage and frequency scaling (DVFS) in processors but tailored to deep learning workloads.

---

## Technologies Used

### Programming
- Python  
- CUDA  
- PyTorch Lightning  
- NVML API  

### Profiling
- NVIDIA NVML  
- Nsight Systems  
- nvidia-smi  

### Visualization
- Matplotlib  
- Seaborn  
- Pandas  

### Environment
- fosscuda/2020b  
- Python/3.9.6-GCCcore-11.2.0  

---

## Our Approach

1. Develop Baseline: Implement naive attention using matrix multiplication.  
2. Recreate FlashAttention: Optimize memory and speed using tile-based recomputation.  
3. Integrate GPU Telemetry: Monitor real-time power, utilization, and thermal data.  
4. Build Dynamic Scheduler: Adjust precision, block size, and iteration strategy.  
5. Benchmark Results: Compare against both naive and static FlashAttention across various sequence lengths.

---

## System Architecture
Input → FlashAttention Kernel → Power Monitor (NVML) → Adaptive Scheduler → GPU Execution → Feedback Loop


---

## Module Description

- `baseline_attention.py` – Implements standard attention for reference.  
- `flash_attention_dynamic.py` – Enhanced FlashAttention kernel with adaptive block computation.  
- `power_monitor.py` – Tracks GPU power, utilization, and thermal metrics.  
- `controller.py` – Adjusts precision and computation strategy based on telemetry.  
- `utils.py` – Handles profiling, logging, and plotting.

---

## Implementation Plan

### Phase 1 – Setup & Baseline
- Implement and validate standard attention.  
- Record baseline performance.

### Phase 2 – FlashAttention Implementation
- Implement tile-based and recompute-friendly GPU kernel.  
- Validate performance and correctness.

### Phase 3 – Power-Aware Extension
- Integrate NVML to record live power and thermal metrics.  
- Build adaptive control logic to switch computation parameters dynamically.

### Phase 4 – Evaluation & Reporting
- Benchmark results across sequence lengths (128–2048).  
- Plot FLOPS, power draw, and efficiency comparisons.

---

## Key Features

- Adaptive GPU scheduling for attention.  
- Real-time telemetry integration.  
- Dynamic precision scaling (FP32 ↔ FP16).  
- Optimized GPU power stability.  
- High-throughput performance-per-watt benchmarking.

---

## Goals

- **Speedup:** 2–3× over baseline attention.  
- **Efficiency:** 15–30% higher performance-per-watt.  
- **Scalability:** Stable runtime for sequence lengths up to 2048.

---

## Environment Setup

- module purge
- module load fosscuda/2020b
- module load Python/3.9.6-GCCcore-11.2.0
- python3 -m venv my_flash_env
- source my_flash_env/bin/activate
- pip install torch torchvision torchaudio pytorch-lightning matplotlib pyyaml

---

## Research Context and Relevance

Energy-efficient Transformer computation is a critical topic in modern AI infrastructure. NVIDIA Hopper and AMD MI300 architectures emphasize adaptive energy control, making this project directly aligned with industry focus areas.

### Key References

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention* (NeurIPS 2022)
- NVIDIA Hopper Architecture Whitepaper (2023)
- Zeus (USENIX 2023): GPU Energy Optimization Framework

---

## Experimental Setup

### Hardware

- **GPU:** NVIDIA H100 NVL MIG 1g.12GB  
- **CPU:** AMD EPYC 7H12 @ 2.6 GHz  
- **RAM:** 32 GB  

---

### Workload

- **Sequence lengths:** 128–2048  
- **Batch sizes:** 32, 64, 128  
- **Precision:** FP32, FP16  

---

## Evaluation Metrics

- Latency (s)  
- GPU Power (W)  
- Energy (J)  
- Performance-per-Watt (GFLOPS/W)  
- GPU Utilization (%)  
- Temperature (°C)  

---

## Expected Impact

This project highlights the intersection of AI model performance and sustainable computing. It aligns closely with NVIDIA’s and AMD’s ongoing work on adaptive energy-efficient architectures. The proposed design can:

- Reduce operational power in AI clusters  
- Enable adaptive scheduling for real-world workloads  
- Serve as a foundation for future “green” deep learning frameworks  

---

## Conclusion

Dynamic FlashAttention++ redefines efficiency in Transformer computation by unifying power awareness, kernel optimization, and scalability. It presents a real-world step toward sustainable high-performance AI.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017)  
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention* (2022)  
- NVIDIA Hopper Architecture Whitepaper (2023)  
- AMD ROCm Documentation (2024)  
- USENIX Zeus: GPU Energy Optimization (2023)  
- FlashAttention YouTube Explanation  
- NVIDIA H100 GPU Overview  
- Efficient Attention Computation Overview  
- NVIDIA Developer Blog – Maximizing Energy and Power Efficiency in Applications with NVIDIA GPUs  



