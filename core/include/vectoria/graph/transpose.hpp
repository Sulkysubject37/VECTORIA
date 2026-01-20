#pragma once

#include "vectoria/ir.hpp"
#include <vector>

namespace vectoria {
namespace graph {

/**
 * Adds a Transpose operation to the graph.
 * 
 * @param graph The graph to append nodes to.
 * @param input_id The input node ID.
 * @param perm The permutation vector.
 * @return The node ID of the Transpose output.
 */
int add_transpose(ir::Graph& graph, int input_id, const std::vector<int64_t>& perm);

} // namespace graph
} // namespace vectoria
