import sys
import os
from typing import List, Dict, Any

class TraceViz:
    """
    Generates visualizations from VECTORIA traces.
    """
    def __init__(self, events: List[Dict[str, Any]]):
        self.events = events

    def generate_timeline_svg(self, output_path: str):
        # Only consider execution events for timeline
        exec_events = [e for e in self.events if e["type"] in ("NodeExecutionStart", "NodeExecutionEnd")]
        if not exec_events:
            print("No execution events found to visualize.")
            return

        min_ts = exec_events[0]["timestamp_ns"]
        max_ts = exec_events[-1]["timestamp_ns"]
        duration = max_ts - min_ts
        if duration == 0: duration = 1

        width = 1000
        padding = 50
        
        # Calculate node boundaries
        nodes = {}
        for e in exec_events:
            nid = e["node_id"]
            if nid not in nodes: nodes[nid] = {"start": None, "end": None}
            if e["type"] == "NodeExecutionStart":
                nodes[nid]["start"] = e["timestamp_ns"]
            else:
                nodes[nid]["end"] = e["timestamp_ns"]

        sorted_nids = sorted(nodes.keys())
        row_height = 30
        
        svg = [
            f'<svg width="{width + 2*padding}" height="{len(sorted_nids) * row_height + 2*padding}" xmlns="http://www.w3.org/2000/svg">',
            '<rect width="100%" height="100%" fill="#f8f9fa" />',
            f'<text x="{padding}" y="30" font-family="monospace" font-size="16" fill="#333">VECTORIA Execution Timeline (Total: {duration/1e6:.2f} ms)</text>'
        ]

        for i, nid in enumerate(sorted_nids):
            n = nodes[nid]
            if n["start"] is None or n["end"] is None: continue
            
            x_start = padding + (n["start"] - min_ts) / duration * width
            x_end = padding + (n["end"] - min_ts) / duration * width
            y = padding + 40 + i * row_height
            
            w = max(x_end - x_start, 2)
            
            svg.append(f'  <rect x="{x_start}" y="{y}" width="{w}" height="20" fill="#007bff" rx="2" />')
            svg.append(f'  <text x="{padding-10}" y="{y+15}" font-family="monospace" font-size="12" text-anchor="end" fill="#666">Node {nid}</text>')

        svg.append('</svg>')
        
        with open(output_path, 'w') as f:
            f.write("\n".join(svg))
        print(f"Timeline SVG written to {output_path}")

    def generate_dot_graph(self, output_path: str):
        # Reconstruct graph from KernelDispatch details
        # e.g. "Reference | Inputs: [0, 1]"
        dot = ["digraph VectoriaExecution {", "  rankdir=LR;", '  node [shape=box, fontname="monospace"];']
        
        nodes_seen = set()
        edges = set()

        for ev in self.events:
            nid = ev["node_id"]
            if nid == -1: continue
            
            nodes_seen.add(nid)
            if ev["type"] == "KernelDispatch":
                details = ev["details"]
                if "Inputs: [" in details:
                    try:
                        inputs_str = details.split("Inputs: [")[1].split("]")[0]
                        inputs = [int(i.strip()) for i in inputs_str.split(",")]
                        for inp in inputs:
                            edges.add((inp, nid))
                            nodes_seen.add(inp)
                    except:
                        pass
                
                # Style node based on dispatch
                color = "#d1ecf1" if "simd" in details.lower() else "#fff3cd"
                label = f"Node {nid}\\n{details.split('|')[0].strip()}"
                dot.append(f'  node_{nid} [label="{label}", style=filled, fillcolor="{color}"];')

        for nid in nodes_seen:
            if not any(f'node_{nid}' in line for line in dot):
                dot.append(f'  node_{nid} [label="Node {nid}"];')

        for src, dst in edges:
            dot.append(f'  node_{src} -> node_{dst};')

        dot.append("}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(dot))
        print(f"DOT graph written to {output_path}")

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

    if len(sys.argv) < 3:
        print("Usage: python trace_viz.py <trace.json> <output_base>")
        sys.exit(1)

    try:
        events = TraceReader.load_json(sys.argv[1])
        viz = TraceViz(events)
        viz.generate_timeline_svg(sys.argv[2] + "_timeline.svg")
        viz.generate_dot_graph(sys.argv[2] + "_graph.dot")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)