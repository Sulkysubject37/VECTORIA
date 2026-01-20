#pragma once

#include "vectoria/ir.hpp"
#include "vectoria/graph/layernorm.hpp"
#include "vectoria/graph/logsoftmax.hpp"
#include "vectoria/graph/stable_softmax.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a Softmax operation into a composed subgraph.
 * 
 * Semantics:
 * y = exp(x - reduce_max(x, axis=-1))
 * z = y / reduce_sum(y, axis=-1)
 * 
 * @param graph The graph to append nodes to.
 * @param input_id The input node ID.
 * @return The node ID of the final Softmax output.
 */
int add_softmax_composed(ir::Graph& graph, int input_id);

} // namespace graph
} // namespace vectoria
