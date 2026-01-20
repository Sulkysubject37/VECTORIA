#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace graph {

/**
 * Expands a CrossEntropy operation into a composed subgraph.
 * Inference-Only.
 * 
 * Semantics:
 * log_probs = LogSoftmax(logits)
 * weighted = target * log_probs
 * sum = reduce_sum(weighted, axis=-1)
 * result = -1.0 * sum
 * 
 * @param graph The graph to append nodes to.
 * @param logits_id The logits input node ID.
 * @param target_id The target input node ID.
 * @return The node ID of the final CrossEntropy output.
 */
int add_crossentropy_composed(ir::Graph& graph, int logits_id, int target_id);

} // namespace graph
} // namespace vectoria
