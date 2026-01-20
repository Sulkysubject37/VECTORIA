#pragma once

#include "vectoria/ir.hpp"
#include <vector>

namespace vectoria {
namespace graph {

/**
 * Adds a Reshape operation to the graph.
 * 
 * @param graph The graph to append nodes to.
 * @param input_id The input node ID.
 * @param new_shape The target shape.
 * @return The node ID of the Reshape output.
 */
int add_reshape(ir::Graph& graph, int input_id, const std::vector<int64_t>& new_shape);

} // namespace graph
} // namespace vectoria
