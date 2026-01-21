#pragma once

#include "vectoria/ir.hpp"
#include <vector>

namespace vectoria {
namespace graph {

/**
 * Adds a Concatenation operation to the graph.
 * 
 * @param graph The graph to append nodes to.
 * @param input_ids List of input node IDs to concatenate.
 * @param axis The axis along which to concatenate.
 * @return The node ID of the Concat output.
 */
int add_concat(ir::Graph& graph, const std::vector<int>& input_ids, int64_t axis);

} // namespace graph
} // namespace vectoria
