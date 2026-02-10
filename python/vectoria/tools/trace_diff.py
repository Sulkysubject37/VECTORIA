import sys
import os
from typing import List, Dict, Any

class TraceDiff:
    """
    Compares two VECTORIA traces for determinism verification.
    """
    @staticmethod
    def compare(trace1: List[Dict[str, Any]], trace2: List[Dict[str, Any]]) -> List[str]:
        differences = []

        if len(trace1) != len(trace2):
            differences.append(f"Trace length mismatch: {len(trace1)} vs {len(trace2)} events")
            # Continue to compare as many as possible
        
        limit = min(len(trace1), len(trace2))
        for i in range(limit):
            e1 = trace1[i]
            e2 = trace2[i]

            # Structural Checks
            if e1["type"] != e2["type"]:
                differences.append(f"Event {i}: Type mismatch ('{e1['type']}' != '{e2['type']}')")
            
            if e1["node_id"] != e2["node_id"]:
                differences.append(f"Event {i}: Node ID mismatch ({e1['node_id']} != {e2['node_id']})")

            # Semantic Checks
            if e1["type"] == "MemoryAllocation":
                if e1["details"] != e2["details"]:
                    differences.append(f"Event {i}: Allocation size mismatch ('{e1['details']}' != '{e2['details']}')")

            if e1["type"] == "KernelDispatch":
                # We check the dispatch string (e.g. "SIMD" vs "Reference")
                if e1["details"] != e2["details"]:
                    differences.append(f"Event {i}: Kernel dispatch mismatch ('{e1['details']}' != '{e2['details']}')")

        return differences

if __name__ == "__main__":
    import sys
    try:
        from .trace_reader import TraceReader
    except ImportError:
        try:
            from trace_reader import TraceReader
        except ImportError:
            import sys
            sys.path.append(os.path.dirname(__file__))
            from trace_reader import TraceReader

    if len(sys.argv) != 3:
        print("Usage: python trace_diff.py <trace1.json> <trace2.json>")
        sys.exit(1)

    try:
        t1 = TraceReader.load_json(sys.argv[1])
        t2 = TraceReader.load_json(sys.argv[2])
        
        diffs = TraceDiff.compare(t1, t2)
        if not diffs:
            print("Traces are identical (Deterministic).")
            sys.exit(0)
        else:
            print(f"Traces differ ({len(diffs)} violations):")
            for d in diffs:
                print(f"  - {d}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
