import ctypes
import os
import sys
from typing import List
from .graph import Graph, DType

# Load Library
_lib_path = os.path.abspath("libvectoria.dylib")
try:
    _lib = ctypes.CDLL(_lib_path)
except OSError:
    # Fallback for when running from a different directory or if not built yet
    _lib = None
    print("Warning: libvectoria.dylib not found. Runtime execution disabled.")

if _lib:
    # Types
    c_graph_t = ctypes.c_void_p
    c_engine_t = ctypes.c_void_p

    # Signatures
    _lib.vectoria_graph_create.restype = c_graph_t
    _lib.vectoria_graph_destroy.argtypes = [c_graph_t]

    _lib.vectoria_graph_add_input.argtypes = [c_graph_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int]
    _lib.vectoria_graph_add_input.restype = ctypes.c_int

    _lib.vectoria_graph_add_parameter.argtypes = [c_graph_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int]
    _lib.vectoria_graph_add_parameter.restype = ctypes.c_int

    _lib.vectoria_graph_add_op_matmul.argtypes = [c_graph_t, ctypes.c_int, ctypes.c_int]
    _lib.vectoria_graph_add_op_matmul.restype = ctypes.c_int

    _lib.vectoria_graph_add_op_bias_add.argtypes = [c_graph_t, ctypes.c_int, ctypes.c_int]
    _lib.vectoria_graph_add_op_bias_add.restype = ctypes.c_int

    _lib.vectoria_graph_add_op_relu.argtypes = [c_graph_t, ctypes.c_int]
    _lib.vectoria_graph_add_op_relu.restype = ctypes.c_int

    _lib.vectoria_graph_add_op_add.argtypes = [c_graph_t, ctypes.c_int, ctypes.c_int]
    _lib.vectoria_graph_add_op_add.restype = ctypes.c_int

    _lib.vectoria_graph_add_op_mul.argtypes = [c_graph_t, ctypes.c_int, ctypes.c_int]
    _lib.vectoria_graph_add_op_mul.restype = ctypes.c_int

    _lib.vectoria_graph_add_op_reduce_sum.argtypes = [c_graph_t, ctypes.c_int]
    _lib.vectoria_graph_add_op_reduce_sum.restype = ctypes.c_int

    _lib.vectoria_graph_add_softmax.argtypes = [c_graph_t, ctypes.c_int]
    _lib.vectoria_graph_add_softmax.restype = ctypes.c_int

    _lib.vectoria_graph_add_softmax_stable.argtypes = [c_graph_t, ctypes.c_int]
    _lib.vectoria_graph_add_softmax_stable.restype = ctypes.c_int

    _lib.vectoria_graph_add_logsoftmax.argtypes = [c_graph_t, ctypes.c_int]
    _lib.vectoria_graph_add_logsoftmax.restype = ctypes.c_int

    _lib.vectoria_graph_add_layernorm.argtypes = [c_graph_t, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib.vectoria_graph_add_layernorm.restype = ctypes.c_int

    _lib.vectoria_graph_set_output.argtypes = [c_graph_t, ctypes.c_int]

    _lib.vectoria_engine_create.argtypes = [c_graph_t]
    _lib.vectoria_engine_create.restype = c_engine_t
    _lib.vectoria_engine_destroy.argtypes = [c_engine_t]

    _lib.vectoria_engine_compile.argtypes = [c_engine_t]
    _lib.vectoria_engine_execute.argtypes = [c_engine_t]
    
    _lib.vectoria_engine_get_buffer.argtypes = [c_engine_t, ctypes.c_int]
    _lib.vectoria_engine_get_buffer.restype = ctypes.c_void_p

    _lib.vectoria_engine_get_trace_size.argtypes = [c_engine_t]
    _lib.vectoria_engine_get_trace_size.restype = ctypes.c_size_t

    _lib.vectoria_engine_get_trace_event.argtypes = [
        c_engine_t, ctypes.c_size_t, 
        ctypes.POINTER(ctypes.c_int), 
        ctypes.POINTER(ctypes.c_uint64), 
        ctypes.POINTER(ctypes.c_int64), 
        ctypes.c_char_p, ctypes.c_size_t
    ]

class Runtime:
    def __init__(self):
        if not _lib:
            raise RuntimeError("Vectoria native library not loaded.")
        self._graph_handle = _lib.vectoria_graph_create()
        self._engine_handle = None
        self._node_map = {} # Python Node ID -> C API ID

    def __del__(self):
        if self._engine_handle:
            _lib.vectoria_engine_destroy(self._engine_handle)
        if self._graph_handle:
            _lib.vectoria_graph_destroy(self._graph_handle)

    def load_graph(self, graph: Graph):
        """
        Reconstructs the Python graph in the C++ backend.
        """
        # Iterate over nodes in order
        for node in graph.nodes:
            nid = node['id']
            ntype = node['type']
            
            # Map DataType string to enum int
            # IR: Float32=0, Float16=1, Int32=2, Int8=3
            dtype_map = {"Float32": 0, "Float16": 1, "Int32": 2, "Int8": 3}
            
            cid = -1
            if ntype == "Input":
                shape = (ctypes.c_int64 * len(node['shape']))(*node['shape'])
                dtype = dtype_map[node['dtype']]
                name = node['name'].encode('utf-8')
                cid = _lib.vectoria_graph_add_input(self._graph_handle, name, shape, len(node['shape']), dtype)
                
            elif ntype == "Parameter":
                shape = (ctypes.c_int64 * len(node['shape']))(*node['shape'])
                dtype = dtype_map[node['dtype']]
                name = node['name'].encode('utf-8')
                cid = _lib.vectoria_graph_add_parameter(self._graph_handle, name, shape, len(node['shape']), dtype)
                
            elif ntype == "Op":
                op_type = node['op']
                if op_type == "MatMul":
                    inp0 = self._node_map[node['inputs'][0]]
                    inp1 = self._node_map[node['inputs'][1]]
                    cid = _lib.vectoria_graph_add_op_matmul(self._graph_handle, inp0, inp1)
                elif op_type == "BiasAdd":
                    inp0 = self._node_map[node['inputs'][0]]
                    inp1 = self._node_map[node['inputs'][1]]
                    cid = _lib.vectoria_graph_add_op_bias_add(self._graph_handle, inp0, inp1)
                elif op_type == "Relu":
                    inp0 = self._node_map[node['inputs'][0]]
                    cid = _lib.vectoria_graph_add_op_relu(self._graph_handle, inp0)
                elif op_type == "Add":
                    inp0 = self._node_map[node['inputs'][0]]
                    inp1 = self._node_map[node['inputs'][1]]
                    cid = _lib.vectoria_graph_add_op_add(self._graph_handle, inp0, inp1)
                elif op_type == "Mul":
                    inp0 = self._node_map[node['inputs'][0]]
                    inp1 = self._node_map[node['inputs'][1]]
                    cid = _lib.vectoria_graph_add_op_mul(self._graph_handle, inp0, inp1)
                elif op_type == "ReduceSum":
                    inp0 = self._node_map[node['inputs'][0]]
                    cid = _lib.vectoria_graph_add_op_reduce_sum(self._graph_handle, inp0)
                elif op_type == "Softmax":
                    inp0 = self._node_map[node['inputs'][0]]
                    cid = _lib.vectoria_graph_add_softmax(self._graph_handle, inp0)
                elif op_type == "SoftmaxStable":
                    inp0 = self._node_map[node['inputs'][0]]
                    cid = _lib.vectoria_graph_add_softmax_stable(self._graph_handle, inp0)
                elif op_type == "LogSoftmax":
                    inp0 = self._node_map[node['inputs'][0]]
                    cid = _lib.vectoria_graph_add_logsoftmax(self._graph_handle, inp0)
                elif op_type == "LayerNorm":
                    inp0 = self._node_map[node['inputs'][0]]
                    gamma = self._node_map[node['inputs'][1]]
                    beta = self._node_map[node['inputs'][2]]
                    cid = _lib.vectoria_graph_add_layernorm(self._graph_handle, inp0, gamma, beta)
                else:
                    raise ValueError(f"Unsupported Op: {op_type}")
            
            if cid != -1:
                self._node_map[nid] = cid

        for out_id in graph.outputs:
            _lib.vectoria_graph_set_output(self._graph_handle, self._node_map[out_id])

        self._engine_handle = _lib.vectoria_engine_create(self._graph_handle)
        _lib.vectoria_engine_compile(self._engine_handle)

    def execute(self):
        if not self._engine_handle:
            raise RuntimeError("Graph not loaded.")
        _lib.vectoria_engine_execute(self._engine_handle)

    def get_buffer(self, node_id: int):
        """
        Returns a ctypes pointer to the buffer.
        """
        if not self._engine_handle:
            raise RuntimeError("Graph not loaded.")
        cid = self._node_map[node_id]
        ptr = _lib.vectoria_engine_get_buffer(self._engine_handle, cid)
        return ptr

    def set_input(self, node_id: int, data: List[float]):
        ptr = self.get_buffer(node_id)
        if not ptr:
            raise ValueError("Invalid node ID or no buffer allocated")
        
        # Assume Float32
        c_float_p = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        for i, val in enumerate(data):
            c_float_p[i] = val

    def get_output(self, node_id: int, size: int) -> List[float]:
        ptr = self.get_buffer(node_id)
        if not ptr:
            raise ValueError("Invalid node ID or no buffer allocated")
            
        c_float_p = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        return [c_float_p[i] for i in range(size)]

    def get_trace(self) -> List['TraceEvent']:
        from .trace import TraceEvent, EventType
        if not self._engine_handle:
            return []
            
        count = _lib.vectoria_engine_get_trace_size(self._engine_handle)
        events = []
        
        c_type = ctypes.c_int()
        c_ts = ctypes.c_uint64()
        c_nid = ctypes.c_int64()
        c_buf = ctypes.create_string_buffer(256)
        
        for i in range(count):
            _lib.vectoria_engine_get_trace_event(
                self._engine_handle, i, 
                ctypes.byref(c_type), 
                ctypes.byref(c_ts), 
                ctypes.byref(c_nid), 
                c_buf, 256
            )
            
            # Map C ID back to Python ID
            # This is tricky because we only stored Python->C map.
            # We need a reverse lookup.
            c_id_val = c_nid.value
            py_id = -1
            if c_id_val != -1:
                # O(N) lookup for now
                for k, v in self._node_map.items():
                    if v == c_id_val:
                        py_id = k
                        break
            
            events.append(TraceEvent(
                EventType(c_type.value), 
                c_ts.value, 
                py_id, 
                c_buf.value.decode('utf-8')
            ))
            
        return events

