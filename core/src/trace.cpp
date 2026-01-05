#include "vectoria/trace.hpp"

namespace vectoria {
namespace trace {

void Tracer::log(EventType type, size_t node_id, const std::string& details) {
    using namespace std::chrono;
    uint64_t now = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    
    events_.push_back({type, now, node_id, details});
}

} // namespace trace
} // namespace vectoria
