#include "vectoria/graph/crossentropy.hpp"
#include "vectoria/graph/logsoftmax.hpp"
#include <stdexcept>
#include <variant>
#include <vector>

namespace vectoria {
namespace graph {

int add_crossentropy_composed(ir::Graph& graph, int logits_id, int target_id) {
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

    ir::TensorShape logits_shape = get_shape(logits_id);
    ir::TensorShape target_shape = get_shape(target_id);

    // Basic shape validation
    // Logits and Target must match in rank. 
    // Exact dimension check is done at execution time usually, 
    // but strict composition prefers checking here if possible.
    // For now we rely on the engine's runtime checks for Mul.

    // 1. LogSoftmax(logits)
    int log_probs_node = add_logsoftmax_composed(graph, logits_id);

    // 2. Weighted = Mul(target, log_probs)
    // Supports broadcasting if target matches log_probs shape.
    int weighted_node = mk_op(ir::OpType::Mul, {static_cast<size_t>(target_id), static_cast<size_t>(log_probs_node)}, logits_shape);

    // 3. Sum = ReduceSum(weighted) -> [Outer]
    ir::TensorShape reduced_shape = logits_shape;
    if (!reduced_shape.dims.empty()) reduced_shape.dims.pop_back();
    
    int sum_node = mk_op(ir::OpType::ReduceSum, {static_cast<size_t>(weighted_node)}, reduced_shape);

    // 4. Negate = Mul(Sum, -1.0) -> [Outer]
    // Scalar broadcast Mul
    int neg_const = mk_const(-1.0f, {});
    int result_node = mk_op(ir::OpType::Mul, {static_cast<size_t>(sum_node), static_cast<size_t>(neg_const)}, reduced_shape);

    return result_node;
}

} // namespace graph
} // namespace vectoria
