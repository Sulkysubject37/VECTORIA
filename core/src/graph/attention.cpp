#include "vectoria/graph/attention.hpp"
#include "vectoria/graph/transpose.hpp"
#include "vectoria/graph/stable_softmax.hpp"
#include <stdexcept>
#include <cmath>
#include <variant>

namespace vectoria {
namespace graph {

int add_attention_composed(ir::Graph& graph, int q_id, int k_id, int v_id) {
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
        c.shape.dims = {}; // Scalar
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

    ir::TensorShape q_shape = get_shape(q_id);
    ir::TensorShape k_shape = get_shape(k_id);
    ir::TensorShape v_shape = get_shape(v_id);

    // Validate ranks (assuming 2D [T, d] for now per scope "Single-head attention")
    if (q_shape.dims.size() != 2 || k_shape.dims.size() != 2 || v_shape.dims.size() != 2) {
        throw std::runtime_error("Attention currently supports only 2D tensors [T, d]");
    }

    int64_t d_k = q_shape.dims[1];
    if (k_shape.dims[1] != d_k) {
        throw std::runtime_error("Q and K dimension mismatch");
    }
    if (k_shape.dims[0] != v_shape.dims[0]) {
        throw std::runtime_error("K and V sequence length mismatch"); // In self-attention they match
    }

    // 1. K_t = Transpose(K) -> [d_k, T]
    int k_t_node = add_transpose(graph, k_id, {1, 0});
    
    // 2. Scores = MatMul(Q, K_t) -> [T, T]
    int64_t seq_len = q_shape.dims[0];
    ir::TensorShape scores_shape;
    scores_shape.dims = {seq_len, seq_len}; // Q[T, d] * K_t[d, T] -> [T, T]
    int scores_node = mk_op(ir::OpType::MatMul, {static_cast<size_t>(q_id), static_cast<size_t>(k_t_node)}, scores_shape);

    // 3. Scale = 1 / sqrt(d_k)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    int scale_const = mk_const(scale_factor, {});
    
    // 4. ScaledScores = Mul(Scores, Scale) -> [T, T]
    int scaled_scores_node = mk_op(ir::OpType::Mul, {static_cast<size_t>(scores_node), static_cast<size_t>(scale_const)}, scores_shape);

    // 5. Probs = StableSoftmax(ScaledScores) -> [T, T]
    int probs_node = add_softmax_stable_composed(graph, scaled_scores_node);

    // 6. Output = MatMul(Probs, V) -> [T, d_v]
    // Probs[T, T] * V[T, d_v] -> [T, d_v]
    ir::TensorShape out_shape;
    out_shape.dims = {seq_len, v_shape.dims[1]};
    int output_node = mk_op(ir::OpType::MatMul, {static_cast<size_t>(probs_node), static_cast<size_t>(v_id)}, out_shape);

    return output_node;
}

} // namespace graph
} // namespace vectoria
