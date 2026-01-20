#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a Stable Softmax operation into a composed subgraph.
 * 
 * Semantics:
 * y = LogSoftmax(x)
 * result = Exp(y)
 * 
 * This ensures numerical stability by reusing the log-sum-exp trick
 * implemented in LogSoftmax.
 * 
 * @param graph The graph to append nodes to.
 * @param input_id The input node ID.
 * @return The node ID of the final Softmax output.
 */
int add_softmax_stable_composed(ir::Graph& graph, int input_id);

} // namespace graph
} // namespace vectoria
