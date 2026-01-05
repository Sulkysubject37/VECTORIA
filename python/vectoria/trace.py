from enum import Enum
from dataclasses import dataclass

class EventType(Enum):
    GraphCompilation = 0
    MemoryAllocation = 1
    NodeExecutionStart = 2
    NodeExecutionEnd = 3
    KernelDispatch = 4

@dataclass
class TraceEvent:
    type: EventType
    timestamp_ns: int
    node_id: int
    details: str

    def __repr__(self):
        return f"[{self.timestamp_ns}] {self.type.name} (Node {self.node_id}) {self.details}"
