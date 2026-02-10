import json
import os
from typing import List, Dict, Any

class TraceReader:
    """
    Loads and validates VECTORIA execution traces.
    """
    VALID_EVENT_TYPES = {
        "GraphCompilation",
        "MemoryAllocation",
        "NodeExecutionStart",
        "NodeExecutionEnd",
        "KernelDispatch"
    }

    @staticmethod
    def load_json(file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trace file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in trace file: {e}")

        if not isinstance(data, list):
            raise ValueError("Trace data must be a list of events")

        validated_events = []
        for i, event in enumerate(data):
            validated_events.append(TraceReader.validate_event(event, i))
        
        return validated_events

    @staticmethod
    def validate_event(event: Dict[str, Any], index: int) -> Dict[str, Any]:
        required_fields = {"type", "timestamp_ns", "node_id", "details"}
        missing = required_fields - set(event.keys())
        if missing:
            raise ValueError(f"Event {index} missing fields: {missing}")

        if event["type"] not in TraceReader.VALID_EVENT_TYPES:
            raise ValueError(f"Event {index} has invalid type: {event['type']}")

        if not isinstance(event["timestamp_ns"], int):
            raise ValueError(f"Event {index} timestamp_ns must be int")

        if not isinstance(event["node_id"], int):
            raise ValueError(f"Event {index} node_id must be int")

        if not isinstance(event["details"], str):
            raise ValueError(f"Event {index} details must be str")

        return event

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            events = TraceReader.load_json(sys.argv[1])
            print(f"Successfully loaded {len(events)} events.")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
