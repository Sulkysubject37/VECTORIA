#include "vectoria/graph/slice.hpp"
#include <stdexcept>
#include <variant>

namespace vectoria {
namespace graph {

int add_slice(ir::Graph& graph, int input_id, int64_t axis, int64_t start, int64_t end) {
    const auto& n = graph.nodes[input_id];
    ir::TensorShape in_shape;
    ir::DataType dtype;

    if (auto* i = std::get_if<ir::InputNode>(&n.data)) { in_shape = i->shape; dtype = i->dtype; }
    else if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) { in_shape = p->shape; dtype = p->dtype; }
    else if (auto* c = std::get_if<ir::ConstantNode>(&n.data)) { in_shape = c->shape; dtype = c->dtype; }
    else if (auto* o = std::get_if<ir::OpNode>(&n.data)) { in_shape = o->output_shape; dtype = o->output_dtype; }
    else throw std::runtime_error("Invalid input node for Slice");

    if (axis < 0) axis += in_shape.dims.size();
    if (axis < 0 || axis >= static_cast<int64_t>(in_shape.dims.size())) throw std::runtime_error("Slice axis out of range");

    if (start < 0) start += in_shape.dims[axis];
    if (end < 0) end += in_shape.dims[axis];
    
    if (start < 0 || end > in_shape.dims[axis] || start >= end) throw std::runtime_error("Invalid slice range");

    ir::TensorShape out_shape = in_shape;
    out_shape.dims[axis] = end - start;

    size_t id = graph.nodes.size();
    ir::OpNode op;
    op.op = ir::OpType::Slice;
    op.inputs = {{static_cast<size_t>(input_id)}};
    op.output_shape = out_shape;
    op.output_dtype = dtype;
    op.int_params = {axis, start, end};

    graph.nodes.push_back({ {id}, op });
    return static_cast<int>(id);
}

} // namespace graph
} // namespace vectoria
