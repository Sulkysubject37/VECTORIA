#pragma once

#include "vectoria/ir.hpp"
#include "vectoria/graph/layernorm.hpp"
#include "vectoria/graph/logsoftmax.hpp"
#include "vectoria/graph/stable_softmax.hpp"
#include "vectoria/graph/crossentropy.hpp"
#include "vectoria/graph/attention.hpp"
#include "vectoria/graph/multi_head_attention.hpp"
#include "vectoria/graph/transformer_encoder.hpp"
#include "vectoria/graph/transpose.hpp"
#include "vectoria/graph/reshape.hpp"
#include "vectoria/graph/concatenation.hpp"
#include "vectoria/graph/slice.hpp"

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
