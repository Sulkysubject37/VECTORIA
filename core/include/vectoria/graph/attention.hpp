#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a Scaled Dot-Product Attention operation into a composed subgraph.
 * 
 * Semantics:
 * K_t = Transpose(K, {1, 0})
 * S = MatMul(Q, K_t)
 * scale = 1.0 / sqrt(d_k)
 * S_scaled = Mul(S, scale)
 * P = StableSoftmax(S_scaled)
 * O = MatMul(P, V)
 * 
 * @param graph The graph to append nodes to.
 * @param q_id The Query node ID.
 * @param k_id The Key node ID.
 * @param v_id The Value node ID.
 * @return The node ID of the final Attention output.
 */
int add_attention_composed(ir::Graph& graph, int q_id, int k_id, int v_id);

} // namespace graph
} // namespace vectoria
