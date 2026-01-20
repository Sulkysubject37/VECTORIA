#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a LogSoftmax operation into a composed subgraph.
 * 
 * Semantics:
 * m = reduce_max(x, axis=-1)
 * shifted = x - m
 * exp_shifted = exp(shifted)
 * sum_exp = reduce_sum(exp_shifted, axis=-1)
 * log_sum = log(sum_exp)
 * result = shifted - log_sum
 * 
 * @param graph The graph to append nodes to.
 * @param input_id The input node ID.
 * @return The node ID of the final LogSoftmax output.
 */
int add_logsoftmax_composed(ir::Graph& graph, int input_id);

} // namespace graph
} // namespace vectoria
