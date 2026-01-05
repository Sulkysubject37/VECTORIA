#pragma once

#include "vectoria/ir.hpp"
#include <vector>

namespace vectoria {

/**
 * Execution engine entry point.
 * Owns scheduling, validation, and kernel dispatch.
 */
class Engine {
public:
    explicit Engine(const ir::Graph& graph);

    /**
     * Validates graph invariants (no cycles, valid node references).
     * @return true if valid.
     */
    bool validate() const;

    /**
     * Compiles the graph into a static execution schedule.
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

private:
    const ir::Graph& graph_;
    std::vector<size_t> schedule_;
    bool compiled_ = false;
};

} // namespace vectoria