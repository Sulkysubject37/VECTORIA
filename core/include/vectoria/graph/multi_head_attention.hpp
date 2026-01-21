#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a Multi-Head Attention operation into a composed subgraph.
 * 
 * @param graph The graph to append nodes to.
 * @param x_id Input node ID [T, d_model].
 * @param w_q_id Query projection weights [d_model, d_model].
 * @param w_k_id Key projection weights [d_model, d_model].
 * @param w_v_id Value projection weights [d_model, d_model].
 * @param w_o_id Output projection weights [d_model, d_model].
 * @param num_heads Number of heads.
 * @return The node ID of the final MHA output.
 */
int add_multi_head_attention_composed(
    ir::Graph& graph, 
    int x_id, 
    int w_q_id, int w_k_id, int w_v_id, int w_o_id, 
    int num_heads
);

} // namespace graph
} // namespace vectoria
