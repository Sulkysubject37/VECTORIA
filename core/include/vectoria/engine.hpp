#pragma once

#include "vectoria/ir.hpp"
#include "vectoria/memory.hpp"
#include "vectoria/kernel_policy.hpp"
#include "vectoria/execution_mode.hpp"
#include "vectoria/trace.hpp"
#include <vector>

namespace vectoria {

struct EngineConfig {
    KernelPolicy policy = KernelPolicy::Reference;
    ExecutionMode mode = ExecutionMode::Research;
};

/**
 * Execution engine entry point.
 * Owns scheduling, validation, and kernel dispatch.
 */
class Engine {
public:
    explicit Engine(const ir::Graph& graph, EngineConfig config = {});

    /**
     * Validates graph invariants (no cycles, valid node references).
     * @return true if valid.
     */
    bool validate() const;

    /**
     * Compiles the graph into a static execution schedule.
     * Also performs memory allocation.
     */
    void compile();

    /**
     * Executes the compiled schedule.
     */
    void execute();

    /**
     * Returns the execution order (node indices).
     */
    const std::vector<size_t>& get_schedule() const { return schedule_; }

    /**
     * Get the raw buffer pointer for a specific node.
     * Useful for setting inputs and reading outputs.
     */
    void* get_buffer(size_t node_idx) const;

    /**
     * Access the execution trace.
     */
    const trace::Tracer& get_tracer() const { return tracer_; }

private:
    const ir::Graph& graph_;
    EngineConfig config_;
    std::vector<size_t> schedule_;
    bool compiled_ = false;

    // Memory management
    memory::Arena arena_;
    std::vector<void*> node_buffers_;
    
    // Observability
    trace::Tracer tracer_;

    // Helper to calculate byte size of a node's output
    size_t calculate_size_bytes(const ir::TensorShape& shape, ir::DataType dtype) const;
};

} // namespace vectoria