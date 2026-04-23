# Comparing GPU and CPU processing speeds on a parallel compute problem
In this week's lab we compared GPU to CPU in terms of their architecture and different use cases. 
## Introduction
GPUs were originally designed for graphics rendering, which involves processing thousands of polygons independently. GPUs excel at problems that are 'embarrassingly parallel'.
<img src="https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/ArchCPUGPUcores.png"
     alt="results"
     width="400"
     style="display:block; margin-left:0;">

A GPU has thousands of small, simple cores designed to run the same instruction on many pieces of data at once. That makes it ideal for workloads where tasks can be split into lots of independent pieces. 
By contrast, a CPU has fewer, more complex cores optimized for sequential tasks, branching logic, and low-latency responses, but not massive parallel processing capacity.

## Experiment
For the experiments, we used an NVIDIA RTX Pro 4000 Blackwell GPU. This device features 70 Streaming Multiprocessors (SMs), with a total of 8,960 Streaming Processors (SPs), enabling a high degree of parallel execution. In total, the GPU can support up to 143,360 concurrent threads, allowing it to handle large-scale data-parallel workloads efficiently.

Vector addition was chosen for this experiment, because it is an embarrassingly parallel problem where each operation is independent. This allows efficient distribution across many GPU threads and provides a simple, controlled way to compare parallel performance without interference from complex algorithms.

## Results
<img src="cpu_vs_gpu_logscale.png" alt="results" width="400">

For the CPU, the computation time grows exponentially with the array size, while the GPU exhibits roughly linear growth.
CPU and GPU take the same time for adding 50000 vector elements. For 10x more elements, we already observe a 6x speedup. The difference grows larger as the array length increases: For 500000000 elements, the GPU computation is over 120x faster than the CPU. 
For smaller arrays, the CPU actually outperforms the GPU. It is over eight times faster at adding 5000 elements.

## Conclusion
The GPU times scale very well for vector addition, as it is designed for computations that are highly parallelizable. This advantage is due to the higher aggregate memory bandwidth and larger number of processing units of the GPU. The GPU, however also has some bottlenecks that lead to worse relative performance for smaller computations. The GPU kernel has to be launched from the CPU thread to prepare the computation, which may cost tens of milliseconds. Additionally, the input vectors initially live in the CPU RAM and have to be transferred to the GPU memory. For future one may further explore these bottleneck effects. How long do kernel launch and memory transfer each take exactly? 
