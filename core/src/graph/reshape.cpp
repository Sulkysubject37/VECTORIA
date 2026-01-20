#include "vectoria/graph/reshape.hpp"
#include <stdexcept>
#include <variant>
#include <numeric>

namespace vectoria {
namespace graph {

int add_reshape(ir::Graph& graph, int input_id, const std::vector<int64_t>& new_shape) {
    const auto& n = graph.nodes[input_id];
    ir::TensorShape in_shape;
    ir::DataType dtype;

    if (auto* i = std::get_if<ir::InputNode>(&n.data)) {
        in_shape = i->shape;
        dtype = i->dtype;
    } else if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) {
        in_shape = p->shape;
        dtype = p->dtype;
    } else if (auto* c = std::get_if<ir::ConstantNode>(&n.data)) {
        in_shape = c->shape;
        dtype = c->dtype;
    } else if (auto* o = std::get_if<ir::OpNode>(&n.data)) {
        in_shape = o->output_shape;
        dtype = o->output_dtype;
    } else {
        throw std::runtime_error("Invalid input node for Reshape");
    }

    size_t in_elements = 1;
    for (auto d : in_shape.dims) in_elements *= d;

    size_t out_elements = 1;
    for (auto d : new_shape) out_elements *= d;

    if (in_elements != out_elements) {
        throw std::runtime_error("Reshape element count mismatch");
    }

    size_t id = graph.nodes.size();
    ir::OpNode op;
    op.op = ir::OpType::Reshape;
    op.inputs = {{static_cast<size_t>(input_id)}};
    op.output_shape.dims = new_shape;
    op.output_dtype = dtype;
    // Reshape doesn't need int_params if output_shape is sufficient, 
    // but storing it explicitly doesn't hurt.
    // However, OpNode already has output_shape.
    
    graph.nodes.push_back({ {id}, op });
    return static_cast<int>(id);
}

} // namespace graph
} // namespace vectoria
