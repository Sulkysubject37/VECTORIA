#pragma once

namespace vectoria {

/**
 * Execution engine entry point.
 * Owns scheduling, memory planning, and kernel dispatch.
 */
class Engine {
public:
    void execute();
};

} // namespace vectoria
