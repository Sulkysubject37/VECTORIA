#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <cstdint>

namespace vectoria {
namespace trace {

enum class EventType {
    GraphCompilation,
    MemoryAllocation,
    NodeExecutionStart,
    NodeExecutionEnd,
    KernelDispatch
};

struct TraceEvent {
    EventType type;
    uint64_t timestamp_ns;
    size_t node_id;         // Optional, or -1
    std::string details;    // E.g., "Reference", "SIMD", "1024 bytes"
};

class Tracer {
public:
    void log(EventType type, size_t node_id = -1, const std::string& details = "");
    
    const std::vector<TraceEvent>& get_events() const { return events_; }
    void clear() { events_.clear(); }

private:
    std::vector<TraceEvent> events_;
};

} // namespace trace
} // namespace vectoria
