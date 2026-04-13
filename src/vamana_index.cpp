#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#ifdef _MSC_VER
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    #define aligned_free(ptr) free(ptr)
#endif

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
    if (quantized_data_) { std::free(quantized_data_); quantized_data_ = nullptr; }
    if (quant_min_)      { std::free(quant_min_);      quant_min_ = nullptr; }
    if (quant_scale_)    { std::free(quant_scale_);     quant_scale_ = nullptr; }
}

// ============================================================================
// Greedy Search
// ============================================================================
// Beam search starting from start_node_. Maintains a candidate list of at most
// L nodes, always expanding the closest unvisited node. Returns when no
// unvisited candidates remain.
//
// Uses a sorted std::vector<Candidate> (flat list) instead of std::set to
// avoid heap allocations in the inner loop. For L<=200, contiguous memmove
// is drastically faster than Red-Black Tree pointer-chasing.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L, bool dynamic_L) const {
    // Candidate list: sorted by (distance, id), bounded at size L.
    // Pre-allocate to avoid reallocation.
    std::vector<Candidate> candidates;
    candidates.reserve(L + 1);  // +1 for temporary insertion before trim

    // Track which nodes we've already visited.
    std::vector<bool> visited(npts_, false);

    uint32_t dist_cmps = 0;

    // Seed with start node
    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidates.push_back({start_dist, start_node_});
    visited[start_node_] = true;

    // Cursor: all entries at index < expand_pos have been expanded.
    // We always expand the entry at expand_pos (the closest un-expanded).
    uint32_t expand_pos = 0;

    // Proposal C Dynamics
    uint32_t active_L = dynamic_L ? std::min((uint32_t)10, L) : L;
    float best_dist = FLT_MAX;
    uint32_t hops_without_improvement = 0;

    while (expand_pos < candidates.size()) {
        if (dynamic_L) {
            float current_best = candidates.front().first;
            if (current_best < best_dist * 0.95f) {
                best_dist = current_best;
                hops_without_improvement = 0;
                active_L = std::max((uint32_t)10, active_L > 5 ? active_L - 5 : 10);
            } else if (current_best < best_dist) {
                best_dist = current_best;
                hops_without_improvement = 0;
            } else {
                hops_without_improvement++;
                if (hops_without_improvement >= 5) {
                    active_L = std::min(L, active_L + 10);
                    hops_without_improvement = 0;
                }
            }
            if (candidates.size() > active_L) {
                candidates.resize(active_L);
                if (expand_pos > active_L) expand_pos = active_L;
            }
        }

        uint32_t best_node = candidates[expand_pos].second;
        expand_pos++;

        // Copy neighbor list under lock to avoid data race with parallel build
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }
        for (uint32_t nbr : neighbors) {
            if (visited[nbr])
                continue;
            visited[nbr] = true;

            // Early-abandon threshold: worst candidate distance (or FLT_MAX if list not full)
            float threshold = (candidates.size() >= active_L)
                ? candidates.back().first : FLT_MAX;
            float d = compute_l2sq_ea(query, get_vector(nbr), dim_, threshold);
            dist_cmps++;

            // Skip if early-abandoned
            if (d == FLT_MAX) continue;

            // Skip if list is full and this is worse than the worst
            if (candidates.size() >= active_L && d >= candidates.back().first)
                continue;

            // Binary search for insertion point to maintain sorted order
            Candidate new_cand = {d, nbr};
            auto pos = std::lower_bound(candidates.begin(), candidates.end(), new_cand);
            size_t insert_idx = pos - candidates.begin();

            if (candidates.size() < active_L) {
                // Not yet at capacity
                candidates.insert(pos, new_cand);
            } else {
                if (pos == candidates.end()) continue; // Handled by threshold check
                // Statically-sized replacement using std::rotate
                candidates.back() = new_cand;
                std::rotate(pos, candidates.end() - 1, candidates.end());
            }

            // Backtrack cursor if we inserted a closer candidate before it.
            // Already-expanded nodes will be harmlessly re-visited (visited[]
            // prevents recomputation).
            if (insert_idx < expand_pos)
                expand_pos = insert_idx;
        }
    }

    return {candidates, dist_cmps};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================
// Given a node and a set of candidates, greedily select neighbors that are
// "diverse" — a candidate c is added only if it's not too close to any
// already-selected neighbor (within a factor of alpha).
//
// Formally: add c if for ALL already-chosen neighbors n:
//     dist(node, c) <= alpha * dist(c, n)
//
// This ensures good graph navigability by keeping some long-range edges
// (alpha > 1 makes it easier for a candidate to survive pruning).

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Remove self from candidates if present
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort by distance to node (ascending)
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R)
            break;

        // Check alpha-RNG condition against all already-selected neighbors
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }

        if (keep)
            new_neighbors.push_back(cand_id);
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // --- Initialize empty graph and per-node locks ---
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // --- Pick random start node ---
    std::mt19937 rng(42);  // fixed seed for reproducibility
    start_node_ = rng() % npts_;
    std::cout << "  Start node: " << start_node_ << std::endl;

    // --- Create random insertion order ---
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // --- Build graph: parallel insertion with per-node locking ---
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R << ")..." << std::endl;

    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t idx = 0; idx < (int64_t)npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1: Search for this point in the current graph to find candidates
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L);

        // Step 2: Prune candidates to get this point's neighbors
        // We don't need to lock graph_[point] here because each point appears
        // exactly once in the permutation — only this thread writes to it now.
        robust_prune(point, candidates, alpha, R);

        // Step 3: Add backward edges from each new neighbor back to this point
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);

            // Add backward edge
            graph_[nbr].push_back(point);

            // Step 4: If neighbor's degree exceeds gamma*R, prune its neighborhood
            if (graph_[nbr].size() > gamma_R) {
                // Build candidate list from current neighbors of nbr
                std::vector<Candidate> nbr_candidates;
                nbr_candidates.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_candidates.push_back({d, nn});
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }

        // Progress reporting (from one thread only)
        if (idx % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "\r  Inserted " << idx << " / " << npts_
                          << " points" << std::flush;
            }
        }
    }

    double build_time = build_timer.elapsed_seconds();

    // Compute average degree
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds."
              << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Build Quantized Data
// ============================================================================
// Computes per-dimension min/scale and quantizes the entire dataset to uint8.
// Quantization: q[d] = round((val[d] - min[d]) / scale[d]), clamped to [0,255]
// Dequantization: val[d] ≈ q[d] * scale[d] + min[d]

void VamanaIndex::build_quantized_data() {
    if (npts_ == 0 || dim_ == 0 || data_ == nullptr)
        throw std::runtime_error("Cannot quantize: data not loaded");

    std::cout << "Building quantized data (uint8 scalar quantization)..." << std::endl;
    Timer qt;

    // Allocate per-dimension statistics
    quant_min_   = static_cast<float*>(std::malloc(dim_ * sizeof(float)));
    quant_scale_ = static_cast<float*>(std::malloc(dim_ * sizeof(float)));
    if (!quant_min_ || !quant_scale_)
        throw std::runtime_error("Failed to allocate quantization tables");

    // Compute per-dimension min and max
    for (uint32_t d = 0; d < dim_; d++) {
        float dmin = FLT_MAX, dmax = -FLT_MAX;
        for (uint32_t i = 0; i < npts_; i++) {
            float val = data_[(size_t)i * dim_ + d];
            if (val < dmin) dmin = val;
            if (val > dmax) dmax = val;
        }
        quant_min_[d] = dmin;
        float range = dmax - dmin;
        quant_scale_[d] = (range > 1e-9f) ? (range / 255.0f) : 1.0f;
    }

    // Allocate quantized data (64-byte aligned for SIMD)
    size_t qdata_size = (size_t)npts_ * dim_;
    size_t aligned_size = (qdata_size + 63) & ~(size_t)63;
    quantized_data_ = static_cast<uint8_t*>(aligned_alloc(64, aligned_size));
    if (!quantized_data_)
        throw std::runtime_error("Failed to allocate quantized data");

    // Quantize all vectors
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)npts_; i++) {
        const float* src = data_ + i * dim_;
        uint8_t* dst = quantized_data_ + i * dim_;
        for (uint32_t d = 0; d < dim_; d++) {
            float normalized = (src[d] - quant_min_[d]) / quant_scale_[d];
            int val = static_cast<int>(std::round(normalized));
            dst[d] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
        }
    }

    has_quantized_ = true;
    std::cout << "  Quantized " << npts_ << " vectors in "
              << qt.elapsed_seconds() << "s" << std::endl;
    std::cout << "  Quantized data size: "
              << (qdata_size / (1024.0 * 1024.0)) << " MB (vs "
              << ((size_t)npts_ * dim_ * sizeof(float) / (1024.0 * 1024.0))
              << " MB float32)" << std::endl;
}

// ============================================================================
// Greedy Search — Quantized (ADC)
// ============================================================================
// Same beam search as greedy_search(), but uses asymmetric distance
// (float32 query vs uint8 dataset) for graph traversal. After the beam search
// completes, all L candidates are re-ranked using exact float32 distances.
// Uses flat sorted vector (no std::set) for zero heap allocations.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search_quantized(const float* query, uint32_t L, uint32_t K, bool dynamic_L) const {
    std::vector<Candidate> candidates;
    candidates.reserve(L + 1);

    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;

    // Seed with start node (use asymmetric distance)
    float start_dist = compute_l2sq_asymmetric(
        query, get_quantized_vector(start_node_),
        quant_min_, quant_scale_, dim_);
    dist_cmps++;
    candidates.push_back({start_dist, start_node_});
    visited[start_node_] = true;

    uint32_t expand_pos = 0;

    // Proposal C Dynamics
    uint32_t active_L = dynamic_L ? std::min((uint32_t)10, L) : L;
    float best_dist = FLT_MAX;
    uint32_t hops_without_improvement = 0;

    while (expand_pos < candidates.size()) {
        if (dynamic_L) {
            float current_best = candidates.front().first;
            if (current_best < best_dist * 0.95f) {
                best_dist = current_best;
                hops_without_improvement = 0;
                active_L = std::max((uint32_t)10, active_L > 5 ? active_L - 5 : 10);
            } else if (current_best < best_dist) {
                best_dist = current_best;
                hops_without_improvement = 0;
            } else {
                hops_without_improvement++;
                if (hops_without_improvement >= 5) {
                    active_L = std::min(L, active_L + 10);
                    hops_without_improvement = 0;
                }
            }
            if (candidates.size() > active_L) {
                candidates.resize(active_L);
                if (expand_pos > active_L) expand_pos = active_L;
            }
        }

        uint32_t best_node = candidates[expand_pos].second;
        expand_pos++;

        // Copy neighbor list under lock
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        for (uint32_t nbr : neighbors) {
            if (visited[nbr])
                continue;
            visited[nbr] = true;

            // Early-abandon threshold
            float threshold = (candidates.size() >= active_L)
                ? candidates.back().first : FLT_MAX;
            float d = compute_l2sq_asymmetric_ea(
                query, get_quantized_vector(nbr),
                quant_min_, quant_scale_, dim_, threshold);
            dist_cmps++;

            // Skip if early-abandoned
            if (d == FLT_MAX) continue;

            // Skip if list is full and this is worse than the worst
            if (candidates.size() >= active_L && d >= candidates.back().first)
                continue;

            // Binary search for sorted insertion
            Candidate new_cand = {d, nbr};
            auto pos = std::lower_bound(candidates.begin(), candidates.end(), new_cand);
            size_t insert_idx = pos - candidates.begin();

            if (candidates.size() < active_L) {
                candidates.insert(pos, new_cand);
            } else {
                if (pos == candidates.end()) continue;
                candidates.back() = new_cand;
                std::rotate(pos, candidates.end() - 1, candidates.end());
            }

            // Backtrack cursor if we inserted a closer candidate before it
            if (insert_idx < expand_pos)
                expand_pos = insert_idx;
        }
    }

    // Re-rank only the top-K candidates with exact float32 distance
    uint32_t num_to_rerank = std::min((uint32_t)candidates.size(), K);
    std::vector<Candidate> reranked;
    reranked.reserve(num_to_rerank);
    
    for (uint32_t i = 0; i < num_to_rerank; i++) {
        uint32_t id = candidates[i].second;
        float exact_dist = compute_l2sq(query, get_vector(id), dim_);
        reranked.push_back({exact_dist, id});
    }
    std::sort(reranked.begin(), reranked.end());

    return {reranked, dist_cmps};
}

// ============================================================================
// Search
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L,
                                 bool use_quantized, bool dynamic_L) const {
    if (L < K) L = K;

    Timer t;
    std::pair<std::vector<Candidate>, uint32_t> search_result;

    if (use_quantized && has_quantized_) {
        search_result = greedy_search_quantized(query, L, K, dynamic_L);
    } else {
        search_result = greedy_search(query, L, dynamic_L);
    }

    auto& [candidates, dist_cmps] = search_result;
    double latency = t.elapsed_us();

    // Return top-K results
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//   [uint32] npts
//   [uint32] dim
//   [uint32] start_node
//   For each node i in [0, npts):
//     [uint32] degree
//     [uint32 * degree] neighbor IDs

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = static_cast<uint32_t>(graph_[i].size());
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) {
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}
