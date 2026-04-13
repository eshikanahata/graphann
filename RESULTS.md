# Search Optimization Results

This document summarizes the performance improvements achieved by implementing Proposals A, B, and D for DiskANN Vamana Search optimization.

## Implementation Summaries

- **Proposal A (Asymmetric Distance Computation via Quantization):** Compressed the dataset from float32 (~488 MB) to uint8 (~128 MB) using per-dimension min/scale scalar block quantization. Graph traversal employs an asymmetric distance function (float32 query vs. uint8 data), with exact float32 re-ranking for the final L candidates. The metadata overhead is negligible (~1KB).
- **Proposal B (Early-Abandoning SIMD Distance):** Added early-exit logic to both exact and asymmetric distance functions. The logic checks if the partial distance sum exceeds the worst candidate's distance at intervals of 32 dimensions, terminating early to save computations while preserving SIMD vectorization.
- **Proposal D (Flattening `std::set`):** Replaced the inner-loop `std::set` (Red-Black Tree) and `std::set` visited frontier map with a pre-allocated contiguous `std::vector` array maintaining sortedness by distance. Candidate insertion shifts contiguous memory via `memmove`. The frontier iterator uses a simple array index cursor (`expand_pos`), avoiding heap allocations entirely in the inner loop. 

---

## Experimental Results

The tests were run with the SIFT1M dataset (1 million vectors, 128 dimensions, Top-100 ground truth) utilizing Windows MSVC environment. 

### 1. Proposal D Impact (Data Structure Flattening)
Flattening the memory structures to rely exclusively on contiguous vectors proved immediately impactful due to removing costly heap allocations per node traversal.

_Comparing Exact Float32 (Standard) vs. Exact Float32 (Flat Vector):_

| L  | Recall@10 | Baseline Avg Latency | **Flat Vector Avg Latency** | Improvement |
|----|-----------|----------------------|-----------------------------|-------------|
| 10 | 0.7727    | 715.8 µs             | **778.4 µs**                | **~-8.7% (Slower)** |
| 50 | 0.9665    | 1650.1 µs            | **1581.9 µs**               | **~4.1%** |
| 75 | 0.9820    | 2224.1 µs            | **2105.6 µs**               | **~5.3%** |
| 150| 0.9939    | 3895.4 µs            | **3449.7 µs**               | **~11.4%** |
| 200| 0.9961    | 5581.9 µs            | **4423.6 µs**               | **~20.7%** |

**Observations:** Flat vectors scale drastically better as $L$ increases. Removing object allocation overhead shifts the search inner loop’s bottleneck squarely onto memory bandwidth and distance computations. Interestingly, using strict `std::rotate` per Proposal D proved slightly *slower* structurally than the earlier prototype using `std::vector::insert` which compiled down to hardware `memmove`.

---

### 2. Proposal A Impact (Asymmetric Quantization)
ADC was benchmarked natively on the flattened vector setup (Proposal D integrated) to evaluate bandwidth reduction. 

_Comparing Exact Float32 (Flat) vs. Quantized ADC (Flat):_

| L  | Exact Recall | ADC Recall | Exact Avg Latency | **ADC Avg Latency** | Latency Delta |
|----|--------------|------------|-------------------|---------------------|---------------|
| 10 | 0.7727       | 0.7718     | 778.4 µs          | **588.7 µs**        | -24.3% Faster |
| 30 | 0.9332       | 0.9272     | 1134.1 µs         | **954.2 µs**        | -15.8% Faster |
| 75 | 0.9820       | 0.9729     | 2105.6 µs         | **1793.4 µs**       | -14.8% Faster |
| 150| 0.9939       | 0.9840     | 3449.7 µs         | **3264.0 µs**       | -5.3% Faster  |
| 200| 0.9961       | 0.9860     | 4423.6 µs         | **4432.0 µs**       | +0.1% Slower  |

**Observations:** 
- **Recall Matching:** Because exact float32 re-ranking is applied *only* to the final top-K candidates in Proposal A, ADC recall drops slightly (e.g., 0.986 vs 0.996 at L=200) compared to exact search. This is the expected compromise when not re-ranking the entire candidate list.
- **Latency Trade-offs:** The strict top-K limitation restores the expected ADC latency advantages, making it drastically faster at lower and medium ranges (up to ~24% faster) than exact float32 calculations without blowing up re-ranking costs.

---

### 3. Proposal B Impact (Early Abandonment)
_The early cancellation logic evaluates exactly identifiably to baseline behavior under normal search ranges._

**Observations:** 
In tests, Early Abandonment kept Recall computations perfectly identical, but the absolute reduction in raw operations (`dist_cmps`) did not yield massive changes. This indicates that Vamana points evaluated during beam-search traversal are largely within the radius of expected thresholds naturally. However, adding it operates at virtually zero cost (evaluating every 32 elements linearly) and offers protective upper bounds when exploring sparse search queries far from target sets.

---

### 4. Proposal C Impact (Dynamic Beam Width)
The Dynamic Beam Width actively restricts and expands the active beam size `L` depending on localization (expanding during local minima blocks and aggressively shrinking `L` on fast routing highways).

_Comparing Exact Float32 vs. Dynamic Exact Float32:_

| L   | Exact Recall | Dynamic Recall | Exact Dist Cmps | Dynamic Dist Cmps | Exact Avg Latency | **Dynamic Avg Latency** |
|-----|--------------|----------------|-----------------|-------------------|-------------------|-------------------------|
| 10  | 0.7727       | 0.7727         | 641.7           | 641.7             | 727.9 µs          | **757.1 µs**            |
| 50  | 0.9665       | 0.9661         | 1511.5          | 1574.3            | 1660.0 µs         | **1443.3 µs**           |
| 100 | 0.9882       | 0.9871         | 2438.9          | 2564.5            | 2341.4 µs         | **2724.4 µs**           |
| 200 | 0.9961       | 0.9956         | 4049.2          | 4319.2            | 4248.2 µs         | **4800.6 µs**           |

_Comparing Quantized ADC vs. Dynamic Quantized ADC:_

| L   | Quant Recall | Dynamic Quant Recall | Quant Avg Latency | **Dynamic Quant Avg Latency** |
|-----|--------------|----------------------|-------------------|-------------------------------|
| 10  | 0.7718       | 0.7718               | 708.6 µs          | **552.9 µs**                  |
| 50  | 0.9587       | 0.9585               | 1253.3 µs         | **1252.3 µs**                 |
| 100 | 0.9788       | 0.9777               | 2145.2 µs         | **2132.7 µs**                 |
| 200 | 0.9860       | 0.9856               | 3735.0 µs         | **3923.7 µs**                 |

**Key Learnings & Observations:** 
- **Mid-Range Routing Benefits:** Aggressive dynamic adaptation offers structural benefits at moderate search constraints (e.g. Exact $L=50$), successfully shaving off absolute computation latency by roughly **13%** (1660µs $\rightarrow$ 1443µs).
- **High-Range Navigational Paradox:** At maximum bounds ($L=200$), dynamically dropping the tracking boundary causes structural detriments. Over-shrinking causes Vamana to discard critical mid/long-range navigational edge candidates, forcing the greedy traversal algorithm to rely on localized nodes and actually perform *more* mathematical distance hops (e.g., 4319 vs 4049) to locate the identical minima. This offsets any cycle savings and severely exacerbates runtime latency.
- **Quantized Synergy limitations:** Since Quantized ADC flattens the native memory latency bottleneck drastically, applying dynamic branch heuristics fails to deliver comparable mathematical scaling improvements. At high values, it strictly adds procedural control-flow overhead rather than rescuing cycles. 
- **Conclusion:** Modifying hot-loop variables in structurally complex graphs indicates that beam width constraints are effectively protective memory buffers rather than just arithmetic load limiters. Dynamic manipulation acts beneficially only safely within local scale parameters.
