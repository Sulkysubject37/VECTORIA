#include "vectoria/engine.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace vectoria {

Engine::Engine(const ir::Graph& graph) : graph_(graph) {}

bool Engine::validate() const {
    // Basic validation: ensure all OpNode inputs refer to existing nodes
    // and that there are no obvious cycles (though topo-sort will catch them too).
    for (const auto& node : graph_.nodes) {
        if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            for (const auto& input_id : op->inputs) {
                if (input_id.index >= graph_.nodes.size()) {
                    return false;
                }
                // In a strictly ordered DAG, inputs must have lower indices 
                // if we assume construction order.
                if (input_id.index >= node.id.index) {
                    return false;
                }
            }
        }
    }
    return true;
}

void Engine::compile() {
    if (!validate()) {
        throw std::runtime_error("Graph validation failed");
    }

    schedule_.clear();
    // For now, we assume the IR construction order is a valid topological sort.
    // In a more complex IR, we would perform a real DFS/Kahn's algorithm.
    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
        schedule_.push_back(i);
    }
    
    compiled_ = true;
}

void Engine::execute() {
    if (!compiled_) {
        throw std::runtime_error("Engine must be compiled before execution");
    }

    // Traversal only, no computation yet.
    for (size_t node_idx : schedule_) {
        const auto& node = graph_.nodes[node_idx];
        // Placeholder for kernel dispatch
    }
}

} // namespace vectoria