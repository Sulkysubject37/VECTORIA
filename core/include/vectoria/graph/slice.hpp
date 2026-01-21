#pragma once

#include "vectoria/ir.hpp"
#include <vector>

namespace vectoria {
namespace graph {

/**
 * Adds a Slice operation to the graph.
 * 
 * @param graph The graph to append nodes to.
 * @param input_id The input node ID.
 * @param axis The axis to slice along.
 * @param start Start index.
 * @param end End index.
 * @return The node ID of the Slice output.
 */
int add_slice(ir::Graph& graph, int input_id, int64_t axis, int64_t start, int64_t end);

} // namespace graph
} // namespace vectoria
