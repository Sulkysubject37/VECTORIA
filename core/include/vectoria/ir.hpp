#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <variant>

namespace vectoria {
namespace ir {

enum class DataType : uint8_t {
    Float32,
    Float16,
    Int32,
    Int8
};

struct TensorShape {
    std::vector<int64_t> dims;
};

enum class OpType : uint16_t {
    Add,
    BiasAdd,
    MatMul,
    Relu,
    Softmax,
    Mul,
    ReduceSum,
    ReduceMax,
    Exp,
    Sub,
    Div,
    Sqrt,
    Log,
    Transpose,
    Reshape
};

struct NodeId {
    size_t index;
};

struct InputNode {
    std::string name;
    TensorShape shape;
    DataType dtype;
};

struct ParameterNode {
    std::string name;
    TensorShape shape;
    DataType dtype;
    // Buffer ownership is handled by the Memory Model, 
    // but IR nodes reference the buffer identity.
    uint64_t buffer_id; 
};

struct ConstantNode {
    TensorShape shape;
    DataType dtype;
    std::vector<float> data_f32;
};

struct OpNode {
    OpType op;
    std::vector<NodeId> inputs;
    TensorShape output_shape;
    DataType output_dtype;
    std::vector<int64_t> int_params;
};

using NodeData = std::variant<InputNode, ParameterNode, ConstantNode, OpNode>;

struct Node {
    NodeId id;
    NodeData data;
};

/**
 * Immutable Graph Representation.
 * Once constructed, the topology and types are frozen.
 * Execution engine uses this to build the static schedule.
 */
struct Graph {
    std::vector<Node> nodes;
    std::vector<NodeId> outputs;

    // Disallow mutation after creation by providing a builder or 
    // simply relying on the engine to treat this as a read-only spec.
};

} // namespace ir
} // namespace vectoria
