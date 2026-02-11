#include "vectoria/graph/layernorm.hpp"
#include <stdexcept>
#include <variant>
#include <vector>

namespace vectoria {
namespace graph {

int add_layernorm_composed(ir::Graph& graph, int input_id, int gamma_id, int beta_id) {
    auto mk_op = [&](ir::OpType type, std::vector<size_t> inputs, const ir::TensorShape& out_shape) {
        size_t id = graph.nodes.size();
        std::vector<ir::NodeId> ins;
        for(auto i : inputs) ins.push_back({i});
        graph.nodes.push_back({ {id}, ir::OpNode{type, ins, out_shape, ir::DataType::Float32} });
        return static_cast<int>(id);
    };

    auto mk_const = [&](float val, const ir::TensorShape& shape) {
        size_t id = graph.nodes.size();
        ir::ConstantNode c;
        c.shape = shape;
        c.dtype = ir::DataType::Float32;
        c.data_f32 = {val};
        c.shape.dims = {}; 
        
        graph.nodes.push_back({ {id}, c });
        return static_cast<int>(id);
    };

    auto get_shape = [&](int idx) -> ir::TensorShape {
        const auto& n = graph.nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* c = std::get_if<ir::ConstantNode>(&n.data)) return c->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    ir::TensorShape in_shape = get_shape(input_id);
    if (in_shape.dims.empty()) throw std::runtime_error("LayerNorm input cannot be empty");

    int64_t last_dim_size = in_shape.dims.back();
    if (last_dim_size <= 0) throw std::runtime_error("LayerNorm last dimension invalid");

    ir::TensorShape reduced_shape = in_shape;
    reduced_shape.dims.pop_back();

    int n_const = mk_const(static_cast<float>(last_dim_size), {});
    int eps_const = mk_const(1e-5f, {});

    // 1. Mean calculation
    int sum_node = mk_op(ir::OpType::ReduceSum, {static_cast<size_t>(input_id)}, reduced_shape);
    int mean_node = mk_op(ir::OpType::Div, {static_cast<size_t>(sum_node), static_cast<size_t>(n_const)}, reduced_shape);

    // 2. Variance calculation
    int diff_node = mk_op(ir::OpType::Sub, {static_cast<size_t>(input_id), static_cast<size_t>(mean_node)}, in_shape);
    int sq_diff_node = mk_op(ir::OpType::Mul, {static_cast<size_t>(diff_node), static_cast<size_t>(diff_node)}, in_shape);
    int var_sum_node = mk_op(ir::OpType::ReduceSum, {static_cast<size_t>(sq_diff_node)}, reduced_shape);
    int var_node = mk_op(ir::OpType::Div, {static_cast<size_t>(var_sum_node), static_cast<size_t>(n_const)}, reduced_shape);

    // 3. Normalization
    int var_eps_node = mk_op(ir::OpType::Add, {static_cast<size_t>(var_node), static_cast<size_t>(eps_const)}, reduced_shape);
    int std_node = mk_op(ir::OpType::Sqrt, {static_cast<size_t>(var_eps_node)}, reduced_shape);
    int norm_node = mk_op(ir::OpType::Div, {static_cast<size_t>(diff_node), static_cast<size_t>(std_node)}, in_shape);

    // 4. Scale and shift (Affine Transform)
    int mul_node = mk_op(ir::OpType::Mul, {static_cast<size_t>(norm_node), static_cast<size_t>(gamma_id)}, in_shape);
    int res_node = mk_op(ir::OpType::BiasAdd, {static_cast<size_t>(mul_node), static_cast<size_t>(beta_id)}, in_shape);

    return res_node;
}

} // namespace graph
} // namespace vectoria