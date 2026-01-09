#pragma once

#include "vectoria/ir.hpp"
#include <string>

namespace vectoria {
namespace lowering {

/**
 * Exports a Graph to a CoreML Model Package.
 * 
 * @param graph The valid VECTORIA IR Graph.
 * @param output_path Path to write the .mlpackage (must end in .mlpackage).
 * @throws std::runtime_error if graph is invalid or lowering fails.
 */
void export_to_coreml(const ir::Graph& graph, const std::string& output_path);

} // namespace lowering
} // namespace vectoria
