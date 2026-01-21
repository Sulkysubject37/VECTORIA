#include "vectoria/graph/transformer_encoder.hpp"
#include "vectoria/graph/multi_head_attention.hpp"
#include "vectoria/graph/layernorm.hpp"
#include <stdexcept>
#include <variant>

namespace vectoria {
namespace graph {

int add_transformer_encoder_composed(
    ir::Graph& graph, 
    int x_id,
    int w_q_id, int w_k_id, int w_v_id, int w_o_id, int num_heads,
    int gamma1_id, int beta1_id,
    int w1_id, int b1_id, int w2_id, int b2_id,
    int gamma2_id, int beta2_id
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
    if (x_shape.dims.size() != 2) throw std::runtime_error("Encoder Block input must be 2D [T, d_model]");
    int64_t seq_len = x_shape.dims[0];
    int64_t d_model = x_shape.dims[1];

    // 1. Multi-Head Attention
    int mha_out = add_multi_head_attention_composed(graph, x_id, w_q_id, w_k_id, w_v_id, w_o_id, num_heads);

    // 2. Residual + LayerNorm 1
    int add1 = mk_op(ir::OpType::Add, {static_cast<size_t>(x_id), static_cast<size_t>(mha_out)}, x_shape);
    int ln1 = add_layernorm_composed(graph, add1, gamma1_id, beta1_id);

    // 3. FFN: Linear1 -> ReLU -> Linear2
    ir::TensorShape w1_shape = get_shape(w1_id);
    if (w1_shape.dims.size() != 2) throw std::runtime_error("FFN W1 must be 2D");
    int64_t d_ff = w1_shape.dims[1];

    ir::TensorShape ffn1_shape; ffn1_shape.dims = {seq_len, d_ff};
    int ffn1_mm = mk_op(ir::OpType::MatMul, {static_cast<size_t>(ln1), static_cast<size_t>(w1_id)}, ffn1_shape);
    int ffn1_bias = mk_op(ir::OpType::BiasAdd, {static_cast<size_t>(ffn1_mm), static_cast<size_t>(b1_id)}, ffn1_shape);
    int ffn1_relu = mk_op(ir::OpType::Relu, {static_cast<size_t>(ffn1_bias)}, ffn1_shape);

    ir::TensorShape ffn2_shape; ffn2_shape.dims = {seq_len, d_model};
    int ffn2_mm = mk_op(ir::OpType::MatMul, {static_cast<size_t>(ffn1_relu), static_cast<size_t>(w2_id)}, ffn2_shape);
    int ffn2_bias = mk_op(ir::OpType::BiasAdd, {static_cast<size_t>(ffn2_mm), static_cast<size_t>(b2_id)}, ffn2_shape);

    // 4. Residual + LayerNorm 2
    int add2 = mk_op(ir::OpType::Add, {static_cast<size_t>(ln1), static_cast<size_t>(ffn2_bias)}, x_shape);
    int ln2 = add_layernorm_composed(graph, add2, gamma2_id, beta2_id);

    return ln2;
}

} // namespace graph
} // namespace vectoria
