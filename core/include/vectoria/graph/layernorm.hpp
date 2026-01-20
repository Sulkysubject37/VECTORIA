#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a Layer Normalization operation into a composed subgraph.
 * 
 * Semantics:
 * mean = reduce_mean(x, axis=-1)
 * var = reduce_mean((x - mean)^2, axis=-1)
 * y = (x - mean) / sqrt(var + epsilon)
 * z = y * gamma + beta
 * 
 * @param graph The graph to append nodes to.
 * @param input_id The input node ID.
 * @param gamma_id The gamma parameter node ID.
 * @param beta_id The beta parameter node ID.
 * @return The node ID of the final LayerNorm output.
 */
int add_layernorm_composed(ir::Graph& graph, int input_id, int gamma_id, int beta_id);

} // namespace graph
} // namespace vectoria
