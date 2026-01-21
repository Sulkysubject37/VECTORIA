#include "vectoria/graph/concatenation.hpp"
#include <stdexcept>
#include <variant>

namespace vectoria {
namespace graph {

int add_concat(ir::Graph& graph, const std::vector<int>& input_ids, int64_t axis) {
    if (input_ids.empty()) {
        throw std::runtime_error("Concat requires at least one input");
    }

    auto get_node_info = [&](int idx) {
        const auto& n = graph.nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return std::make_pair(i->shape, i->dtype);
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return std::make_pair(p->shape, p->dtype);
        if (auto* c = std::get_if<ir::ConstantNode>(&n.data)) return std::make_pair(c->shape, c->dtype);
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return std::make_pair(o->output_shape, o->output_dtype);
        throw std::runtime_error("Invalid node index for Concat");
    };

    auto [first_shape, first_dtype] = get_node_info(input_ids[0]);
    size_t rank = first_shape.dims.size();

    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }

    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
        throw std::runtime_error("Concat axis out of range");
    }

    ir::TensorShape out_shape = first_shape;
    int64_t total_concat_dim = first_shape.dims[axis];

    std::vector<ir::NodeId> inputs;
    inputs.push_back({static_cast<size_t>(input_ids[0])});

    for (size_t i = 1; i < input_ids.size(); ++i) {
        auto [shape, dtype] = get_node_info(input_ids[i]);
        
        if (shape.dims.size() != rank) {
            throw std::runtime_error("Concat input rank mismatch");
        }
        if (dtype != first_dtype) {
            throw std::runtime_error("Concat input dtype mismatch");
        }

        for (size_t j = 0; j < rank; ++j) {
            if (j == static_cast<size_t>(axis)) {
                total_concat_dim += shape.dims[j];
            } else if (shape.dims[j] != first_shape.dims[j]) {
                throw std::runtime_error("Concat input dimension mismatch on non-concat axis");
            }
        }
        inputs.push_back({static_cast<size_t>(input_ids[i])});
    }

    out_shape.dims[axis] = total_concat_dim;

    size_t id = graph.nodes.size();
    ir::OpNode op;
    op.op = ir::OpType::Concat;
    op.inputs = inputs;
    op.output_shape = out_shape;
    op.output_dtype = first_dtype;
    op.int_params = {axis};

    graph.nodes.push_back({ {id}, op });
    return static_cast<int>(id);
}

} // namespace graph
} // namespace vectoria
