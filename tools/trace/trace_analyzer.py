import json
from typing import List, Dict, Any

class TraceAnalyzer:
    """
    Computes insights from VECTORIA execution traces.
    """
    def __init__(self, events: List[Dict[str, Any]]):
        self.events = events

    def analyze(self) -> Dict[str, Any]:
        order = []
        simd_count = 0
        ref_count = 0
        memory = {}
        node_types = {}
        
        # Track active nodes for timing
        starts = {}
        timings = {}

        for ev in self.events:
            etype = ev["type"]
            nid = ev["node_id"]
            details = ev["details"]
            ts = ev["timestamp_ns"]

            if etype == "NodeExecutionStart":
                order.append(nid)
                starts[nid] = ts
            elif etype == "NodeExecutionEnd":
                if nid in starts:
                    timings[nid] = ts - starts[nid]
            elif etype == "KernelDispatch":
                d_lower = details.lower()
                if "simd" in d_lower:
                    simd_count += 1
                elif "reference" in d_lower:
                    ref_count += 1
            elif etype == "MemoryAllocation":
                try:
                    size = int(details.split()[0])
                    memory[nid] = size
                except (ValueError, IndexError):
                    pass

        dispatch_total = simd_count + ref_count
        
        summary = {
            "execution_order": order,
            "kernel_dispatch": {
                "total": dispatch_total,
                "simd": simd_count,
                "reference": ref_count,
                "simd_ratio": simd_count / dispatch_total if dispatch_total > 0 else 0
            },
            "memory_footprint": {
                "per_node": memory,
                "total_bytes": sum(memory.values())
            },
            "timings_ns": timings,
            "composed_op_summary": self._summarize_composed_ops(order)
        }
        return summary

    def _summarize_composed_ops(self, order: List[int]) -> Dict[str, Any]:
        # Without graph IR, we can't be sure about expansions.
        # But we can provide a summary of the sequence.
        return {
            "total_nodes_executed": len(order),
            "unique_nodes": len(set(order))
        }

if __name__ == "__main__":
    import sys
    import os
    # Add parent dir to path to allow relative import if needed, 
    # but we want it to be standalone-ish.
    # We can just import TraceReader if we assume it's in the same dir.
    try:
        from trace_reader import TraceReader
    except ImportError:
        # Fallback for different execution contexts
        import sys
        sys.path.append(os.path.dirname(__file__))
        from trace_reader import TraceReader

    if len(sys.argv) > 1:
        try:
            events = TraceReader.load_json(sys.argv[1])
            analyzer = TraceAnalyzer(events)
            print(json.dumps(analyzer.analyze(), indent=2))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
