#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a Transformer Encoder Block into a composed subgraph.
 * 
 * @param graph The graph to append nodes to.
 * @param x_id Input node ID [T, d_model].
 * @param w_q_id, w_k_id, w_v_id, w_o_id MHA projection weights.
 * @param num_heads Number of attention heads.
 * @param gamma1_id, beta1_id LayerNorm 1 parameters.
 * @param w1_id, b1_id, w2_id, b2_id FFN weights and biases.
 * @param gamma2_id, beta2_id LayerNorm 2 parameters.
 * @return The node ID of the final Encoder Block output.
 */
int add_transformer_encoder_composed(
    ir::Graph& graph, 
    int x_id,
    int w_q_id, int w_k_id, int w_v_id, int w_o_id, int num_heads,
    int gamma1_id, int beta1_id,
    int w1_id, int b1_id, int w2_id, int b2_id,
    int gamma2_id, int beta2_id
);

} // namespace graph
} // namespace vectoria
