#pragma once

#include "vectoria/ir.hpp"

namespace vectoria {
namespace lowering {

/**
 * Validates a graph for CoreML deployment.
 * Checks for supported ops, valid shapes, and semantic constraints.
 * 
 * @param graph The graph to validate.
 * @throws std::runtime_error if validation fails.
 */
void validate_for_deployment(const ir::Graph& graph);

} // namespace lowering
} // namespace vectoria
