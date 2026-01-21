#include "vectoria/c_api.h"
#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/capabilities.hpp"
#include "vectoria/graph_ops.hpp"
#include "vectoria/graph/layernorm.hpp"
#include "vectoria/graph/logsoftmax.hpp"
#include "vectoria/graph/stable_softmax.hpp"
#include "vectoria/graph/crossentropy.hpp"
#include "vectoria/graph/attention.hpp"
#include "vectoria/graph/transpose.hpp"
#include "vectoria/graph/reshape.hpp"
#include "vectoria/lowering/coreml.hpp"
#include <vector>
#include <cstring>
#include <iostream>

using namespace vectoria;

extern "C" {

int vectoria_export_coreml(vectoria_graph_t g, const char* output_path) {
    try {
        auto* graph = static_cast<ir::Graph*>(g);
        lowering::export_to_coreml(*graph, std::string(output_path));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CoreML Export Error: " << e.what() << std::endl;
        return -1;
    }
}

void vectoria_get_capabilities(
    int* arch, 
    int* simd_compiled, 
    int* simd_supported,
    char* arch_name_buffer,
    size_t arch_name_len
) {
    auto caps = capabilities::get_system_capabilities();
    if (arch) *arch = static_cast<int>(caps.arch);
    if (simd_compiled) *simd_compiled = caps.simd_compiled ? 1 : 0;
    if (simd_supported) *simd_supported = caps.simd_supported_on_host ? 1 : 0;
    
    if (arch_name_buffer && arch_name_len > 0) {
        strncpy(arch_name_buffer, caps.arch_name.c_str(), arch_name_len - 1);
        arch_name_buffer[arch_name_len - 1] = '\0';
    }
}

vectoria_graph_t vectoria_graph_create() {
    return new ir::Graph();
}

void vectoria_graph_destroy(vectoria_graph_t g) {
    delete static_cast<ir::Graph*>(g);
}

int vectoria_graph_add_input(vectoria_graph_t g, const char* name, const int64_t* shape, int rank, int dtype) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::InputNode node;
    node.name = name;
    node.shape.dims.assign(shape, shape + rank);
    node.dtype = static_cast<ir::DataType>(dtype); // Careful with mapping!

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_parameter(vectoria_graph_t g, const char* name, const int64_t* shape, int rank, int dtype) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::ParameterNode node;
    node.name = name;
    node.shape.dims.assign(shape, shape + rank);
    node.dtype = static_cast<ir::DataType>(dtype);
    node.buffer_id = 0; // Placeholder

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_op_matmul(vectoria_graph_t g, int input_a, int input_b) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::OpNode node;
    node.op = ir::OpType::MatMul;
    node.inputs = { {static_cast<size_t>(input_a)}, {static_cast<size_t>(input_b)} };
    
    // Auto-infer output shape for convenience in C API
    // This duplicates logic in Python/Engine, but C API needs to construct valid IR.
    // Ideally IR construction is robust.
    // For now, minimal inference: [M, K] x [K, N] -> [M, N]
    // We need to look up inputs.
    // Using unchecked access for speed in minimal impl.
    // Real implementation would be safer.
    
    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph->nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    auto shape_a = get_shape(input_a);
    auto shape_b = get_shape(input_b);
    
    // Basic shape inference
    if (shape_a.dims.size() >= 2 && shape_b.dims.size() >= 2) {
        node.output_shape.dims = {shape_a.dims[0], shape_b.dims[1]};
    }
    
    node.output_dtype = ir::DataType::Float32; // Hardcoded for this phase

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_op_bias_add(vectoria_graph_t g, int input, int bias) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::OpNode node;
    node.op = ir::OpType::BiasAdd;
    node.inputs = { {static_cast<size_t>(input)}, {static_cast<size_t>(bias)} };
    
    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph->nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    node.output_shape = get_shape(input);
    node.output_dtype = ir::DataType::Float32;

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_op_relu(vectoria_graph_t g, int input) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::OpNode node;
    node.op = ir::OpType::Relu;
    node.inputs = { {static_cast<size_t>(input)} };
    
    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph->nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    node.output_shape = get_shape(input);
    node.output_dtype = ir::DataType::Float32;

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_op_add(vectoria_graph_t g, int input_a, int input_b) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::OpNode node;
    node.op = ir::OpType::Add;
    node.inputs = { {static_cast<size_t>(input_a)}, {static_cast<size_t>(input_b)} };
    
    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph->nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    // Auto-infer shape: assume inputs are same shape
    node.output_shape = get_shape(input_a);
    node.output_dtype = ir::DataType::Float32;

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_op_mul(vectoria_graph_t g, int input_a, int input_b) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::OpNode node;
    node.op = ir::OpType::Mul;
    node.inputs = { {static_cast<size_t>(input_a)}, {static_cast<size_t>(input_b)} };
    
    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph->nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    node.output_shape = get_shape(input_a);
    node.output_dtype = ir::DataType::Float32;

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_op_reduce_sum(vectoria_graph_t g, int input) {
    auto* graph = static_cast<ir::Graph*>(g);
    ir::OpNode node;
    node.op = ir::OpType::ReduceSum;
    node.inputs = { {static_cast<size_t>(input)} };
    
    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph->nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* c = std::get_if<ir::ConstantNode>(&n.data)) return c->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    ir::TensorShape s = get_shape(input);
    if (!s.dims.empty()) {
        s.dims.pop_back(); // Reduce last dim
        node.output_shape = s;
    }
    node.output_dtype = ir::DataType::Float32;

    size_t id = graph->nodes.size();
    graph->nodes.push_back({ {id}, node });
    return static_cast<int>(id);
}

int vectoria_graph_add_op_transpose(vectoria_graph_t g, int input, const int64_t* perm, int rank) {
    auto* graph = static_cast<ir::Graph*>(g);
    std::vector<int64_t> p(perm, perm + rank);
    return vectoria::graph::add_transpose(*graph, input, p);
}

int vectoria_graph_add_op_reshape(vectoria_graph_t g, int input, const int64_t* new_shape, int rank) {
    auto* graph = static_cast<ir::Graph*>(g);
    std::vector<int64_t> s(new_shape, new_shape + rank);
    return vectoria::graph::add_reshape(*graph, input, s);
}

int vectoria_graph_add_softmax(vectoria_graph_t g, int input) {
    auto* graph = static_cast<ir::Graph*>(g);
    // Use the composed graph op
    return vectoria::graph::add_softmax_composed(*graph, input);
}

int vectoria_graph_add_softmax_stable(vectoria_graph_t g, int input) {
    auto* graph = static_cast<ir::Graph*>(g);
    return vectoria::graph::add_softmax_stable_composed(*graph, input);
}

int vectoria_graph_add_logsoftmax(vectoria_graph_t g, int input) {
    auto* graph = static_cast<ir::Graph*>(g);
    return vectoria::graph::add_logsoftmax_composed(*graph, input);
}

int vectoria_graph_add_crossentropy(vectoria_graph_t g, int logits, int target) {
    auto* graph = static_cast<ir::Graph*>(g);
    return vectoria::graph::add_crossentropy_composed(*graph, logits, target);
}

int vectoria_graph_add_attention(vectoria_graph_t g, int q, int k, int v) {
    auto* graph = static_cast<ir::Graph*>(g);
    return vectoria::graph::add_attention_composed(*graph, q, k, v);
}

int vectoria_graph_add_layernorm(vectoria_graph_t g, int input, int gamma, int beta) {
    auto* graph = static_cast<ir::Graph*>(g);
    return vectoria::graph::add_layernorm_composed(*graph, input, gamma, beta);
}

void vectoria_graph_set_output(vectoria_graph_t g, int node_id) {
    auto* graph = static_cast<ir::Graph*>(g);
    graph->outputs.push_back({static_cast<size_t>(node_id)});
}

vectoria_engine_t vectoria_engine_create(vectoria_graph_t g) {
    auto* graph = static_cast<ir::Graph*>(g);
    return new Engine(*graph);
}

vectoria_engine_t vectoria_engine_create_with_policy(vectoria_graph_t g, int policy) {
    auto* graph = static_cast<ir::Graph*>(g);
    EngineConfig cfg;
    cfg.policy = static_cast<KernelPolicy>(policy);
    return new Engine(*graph, cfg);
}

void vectoria_engine_destroy(vectoria_engine_t e) {
    delete static_cast<Engine*>(e);
}

void vectoria_engine_compile(vectoria_engine_t e) {
    static_cast<Engine*>(e)->compile();
}

void vectoria_engine_execute(vectoria_engine_t e) {
    static_cast<Engine*>(e)->execute();
}

void* vectoria_engine_get_buffer(vectoria_engine_t e, int node_id) {
    return static_cast<Engine*>(e)->get_buffer(static_cast<size_t>(node_id));
}

size_t vectoria_engine_get_trace_size(vectoria_engine_t e) {
    return static_cast<Engine*>(e)->get_tracer().get_events().size();
}

void vectoria_engine_get_trace_event(
    vectoria_engine_t e, 
    size_t index, 
    int* type, 
    uint64_t* timestamp_ns, 
    int64_t* node_id, 
    char* details_buffer, 
    size_t buffer_len
) {
    const auto& events = static_cast<Engine*>(e)->get_tracer().get_events();
    if (index >= events.size()) return;

    const auto& evt = events[index];
    if (type) *type = static_cast<int>(evt.type);
    if (timestamp_ns) *timestamp_ns = evt.timestamp_ns;
    if (node_id) *node_id = (evt.node_id == static_cast<size_t>(-1)) ? -1 : static_cast<int64_t>(evt.node_id);
    
    if (details_buffer && buffer_len > 0) {
        strncpy(details_buffer, evt.details.c_str(), buffer_len - 1);
        details_buffer[buffer_len - 1] = '\0';
    }
}

} // extern "C"
