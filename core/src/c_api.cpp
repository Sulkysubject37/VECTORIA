#include "vectoria/c_api.h"
#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include <vector>
#include <cstring>

using namespace vectoria;

extern "C" {

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

void vectoria_graph_set_output(vectoria_graph_t g, int node_id) {
    auto* graph = static_cast<ir::Graph*>(g);
    graph->outputs.push_back({static_cast<size_t>(node_id)});
}

vectoria_engine_t vectoria_engine_create(vectoria_graph_t g) {
    auto* graph = static_cast<ir::Graph*>(g);
    return new Engine(*graph);
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

} // extern "C"
