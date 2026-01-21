#include "vectoria/graph/multi_head_attention.hpp"
#include "vectoria/graph/attention.hpp"
#include "vectoria/graph/transpose.hpp"
#include "vectoria/graph/reshape.hpp"
#include "vectoria/graph/slice.hpp"
#include "vectoria/graph/concatenation.hpp"
#include <stdexcept>
#include <variant>

namespace vectoria {
namespace graph {

int add_multi_head_attention_composed(
    ir::Graph& graph, 
    int x_id, 
    int w_q_id, int w_k_id, int w_v_id, int w_o_id, 
    int num_heads
) {
    auto mk_op = [&](ir::OpType type, std::vector<size_t> inputs, const ir::TensorShape& out_shape) {
        size_t id = graph.nodes.size();
        std::vector<ir::NodeId> ins;
        for(auto i : inputs) ins.push_back({i});
        graph.nodes.push_back({ {id}, ir::OpNode{type, ins, out_shape, ir::DataType::Float32} });
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

    ir::TensorShape x_shape = get_shape(x_id);
    int64_t seq_len = x_shape.dims[0];
    int64_t d_model = x_shape.dims[1];
    int64_t d_k = d_model / num_heads;

    // 1. Projections
    ir::TensorShape proj_shape; proj_shape.dims = {seq_len, d_model};
    int q_all = mk_op(ir::OpType::MatMul, {static_cast<size_t>(x_id), static_cast<size_t>(w_q_id)}, proj_shape);
    int k_all = mk_op(ir::OpType::MatMul, {static_cast<size_t>(x_id), static_cast<size_t>(w_k_id)}, proj_shape);
    int v_all = mk_op(ir::OpType::MatMul, {static_cast<size_t>(x_id), static_cast<size_t>(w_v_id)}, proj_shape);

    // 2. Split Heads [T, d_model] -> [T, h, d_k] -> [h, T, d_k]
    ir::TensorShape split_view; split_view.dims = {seq_len, static_cast<int64_t>(num_heads), d_k};
    int q_split = add_reshape(graph, q_all, split_view.dims);
    int k_split = add_reshape(graph, k_all, split_view.dims);
    int v_split = add_reshape(graph, v_all, split_view.dims);

    int q_trans = add_transpose(graph, q_split, {1, 0, 2});
    int k_trans = add_transpose(graph, k_split, {1, 0, 2});
    int v_trans = add_transpose(graph, v_split, {1, 0, 2});

    // 3. Per-head Attention
    std::vector<int> head_outputs;
    for (int h = 0; h < num_heads; ++h) {
        // Slice along axis 0 (heads dimension)
        int q_h = add_slice(graph, q_trans, 0, h, h + 1);
        int k_h = add_slice(graph, k_trans, 0, h, h + 1);
        int v_h = add_slice(graph, v_trans, 0, h, h + 1);

        // After slicing [1, T, d_k], we must reshape to [T, d_k] for the Attention helper
        ir::TensorShape head_shape; head_shape.dims = {seq_len, d_k};
        int q_h_2d = add_reshape(graph, q_h, head_shape.dims);
        int k_h_2d = add_reshape(graph, k_h, head_shape.dims);
        int v_h_2d = add_reshape(graph, v_h, head_shape.dims);

        int head_out = add_attention_composed(graph, q_h_2d, k_h_2d, v_h_2d);
        head_outputs.push_back(head_out);
    }

    // 4. Concatenation
    int concat_heads = add_concat(graph, head_outputs, 1); // [T, d_model]

    // 5. Output Projection
    ir::TensorShape final_shape; final_shape.dims = {seq_len, d_model};
    int result = mk_op(ir::OpType::MatMul, {static_cast<size_t>(concat_heads), static_cast<size_t>(w_o_id)}, final_shape);

    return result;
}

} // namespace graph
} // namespace vectoria