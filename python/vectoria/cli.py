import sys
import argparse
import json
from vectoria.tools.trace_reader import TraceReader
from vectoria.tools.trace_analyzer import TraceAnalyzer
from vectoria.tools.trace_diff import TraceDiff
from vectoria.tools.trace_viz import TraceViz

def main():
    parser = argparse.ArgumentParser(prog="vectoria-trace", description="VECTORIA Trace Introspection Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a trace file")
    analyze_parser.add_argument("file", help="Trace JSON file")

    # Diff
    diff_parser = subparsers.add_parser("diff", help="Compare two trace files")
    diff_parser.add_argument("file1", help="First trace file")
    diff_parser.add_argument("file2", help="Second trace file")

    # Viz
    viz_parser = subparsers.add_parser("viz", help="Visualize a trace file")
    viz_parser.add_argument("file", help="Trace JSON file")
    viz_parser.add_argument("output", help="Output base name (e.g. 'my_viz')")

    args = parser.parse_args()

    if args.command == "analyze":
        try:
            events = TraceReader.load_json(args.file)
            analyzer = TraceAnalyzer(events)
            print(json.dumps(analyzer.analyze(), indent=2))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "diff":
        try:
            t1 = TraceReader.load_json(args.file1)
            t2 = TraceReader.load_json(args.file2)
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

    elif args.command == "viz":
        try:
            events = TraceReader.load_json(args.file)
            viz = TraceViz(events)
            viz.generate_timeline_svg(args.output + "_timeline.svg")
            viz.generate_dot_graph(args.output + "_graph.dot")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
