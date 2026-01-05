from typing import List, Optional, Union, Tuple
from enum import Enum
import json

class DType(Enum):
    FLOAT32 = "Float32"
    FLOAT16 = "Float16"
    INT32 = "Int32"
    INT8 = "Int8"

class Node:
    def __init__(self, id: int):
        self.id = id

class Graph:
    def __init__(self):
        self.nodes: List[dict] = []
        self.inputs: List[dict] = []
        self.parameters: List[dict] = []
        self.ops: List[dict] = []
        self.outputs: List[int] = []
        self._frozen = False

    def add_input(self, name: str, shape: List[int], dtype: DType) -> Node:
        self._check_frozen()
        node_id = len(self.nodes)
        node_data = {
            "type": "Input",
            "id": node_id,
            "name": name,
            "shape": shape,
            "dtype": dtype.value
        }
        self.nodes.append(node_data)
        self.inputs.append(node_data)
        return Node(node_id)

    def add_parameter(self, name: str, shape: List[int], dtype: DType, buffer_id: int) -> Node:
        self._check_frozen()
        node_id = len(self.nodes)
        node_data = {
            "type": "Parameter",
            "id": node_id,
            "name": name,
            "shape": shape,
            "dtype": dtype.value,
            "buffer_id": buffer_id
        }
        self.nodes.append(node_data)
        self.parameters.append(node_data)
        return Node(node_id)

    def add_op(self, op_type: str, inputs: List[Node], output_shape: List[int], output_dtype: DType) -> Node:
        self._check_frozen()
        node_id = len(self.nodes)
        input_ids = [n.id for n in inputs]
        node_data = {
            "type": "Op",
            "id": node_id,
            "op": op_type,
            "inputs": input_ids,
            "output_shape": output_shape,
            "output_dtype": output_dtype.value
        }
        self.nodes.append(node_data)
        self.ops.append(node_data)
        return Node(node_id)

    def set_output(self, node: Node):
        self._check_frozen()
        self.outputs.append(node.id)

    def compile(self):
        """
        Freezes the graph and validates basic structure.
        In a real scenario, this would serialize to C++ IR.
        """
        self._frozen = True
        # Basic validation: ensure all outputs exist
        for out_id in self.outputs:
            if out_id >= len(self.nodes):
                raise ValueError(f"Output node {out_id} does not exist")

    def to_json(self) -> str:
        return json.dumps({
            "nodes": self.nodes,
            "outputs": self.outputs
        }, indent=2)

    def _check_frozen(self):
        if self._frozen:
            raise RuntimeError("Graph is frozen. Cannot modify.")