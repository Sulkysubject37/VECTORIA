#include "vectoria/graph/transpose.hpp"
#include <stdexcept>
#include <variant>
#include <numeric>
#include <algorithm>
#include <set>

namespace vectoria {
namespace graph {

int add_transpose(ir::Graph& graph, int input_id, const std::vector<int64_t>& perm) {
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
        throw std::runtime_error("Invalid input node for Transpose");
    }

    if (perm.size() != in_shape.dims.size()) {
        throw std::runtime_error("Transpose permutation rank mismatch");
    }

    // Validate permutation
    std::vector<int64_t> sorted_perm = perm;
    std::sort(sorted_perm.begin(), sorted_perm.end());
    for (size_t i = 0; i < sorted_perm.size(); ++i) {
        if (sorted_perm[i] != static_cast<int64_t>(i)) {
            throw std::runtime_error("Invalid permutation vector");
        }
    }

    ir::TensorShape out_shape;
    out_shape.dims.resize(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        out_shape.dims[i] = in_shape.dims[perm[i]];
    }

    size_t id = graph.nodes.size();
    ir::OpNode op;
    op.op = ir::OpType::Transpose;
    op.inputs = {{static_cast<size_t>(input_id)}};
    op.output_shape = out_shape;
    op.output_dtype = dtype;
    op.int_params = perm;

    graph.nodes.push_back({ {id}, op });
    return static_cast<int>(id);
}

} // namespace graph
} // namespace vectoria
