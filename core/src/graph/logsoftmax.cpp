#include "vectoria/graph/logsoftmax.hpp"
#include <stdexcept>
#include <variant>
#include <vector>

namespace vectoria {
namespace graph {

int add_logsoftmax_composed(ir::Graph& graph, int input_id) {
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

    ir::TensorShape in_shape = get_shape(input_id);
    if (in_shape.dims.empty()) throw std::runtime_error("LogSoftmax input cannot be empty");

    // Shapes
    ir::TensorShape reduced_shape = in_shape;
    reduced_shape.dims.pop_back(); // Rank N-1

    // 1. Max = ReduceMax(Input) -> [Outer]
    int max_node = mk_op(ir::OpType::ReduceMax, {static_cast<size_t>(input_id)}, reduced_shape);

    // 2. Shifted = Sub(Input, Max) -> [Outer, Inner] (Broadcast Max)
    // Supports broadcasting [Outer] from Sub broadcast logic.
    int shifted_node = mk_op(ir::OpType::Sub, {static_cast<size_t>(input_id), static_cast<size_t>(max_node)}, in_shape);

    // 3. ExpShifted = Exp(Shifted) -> [Outer, Inner]
    int exp_shifted_node = mk_op(ir::OpType::Exp, {static_cast<size_t>(shifted_node)}, in_shape);

    // 4. SumExp = ReduceSum(ExpShifted) -> [Outer]
    int sum_exp_node = mk_op(ir::OpType::ReduceSum, {static_cast<size_t>(exp_shifted_node)}, reduced_shape);

    // 5. LogSum = Log(SumExp) -> [Outer]
    int log_sum_node = mk_op(ir::OpType::Log, {static_cast<size_t>(sum_exp_node)}, reduced_shape);

    // 6. Result = Sub(Shifted, LogSum) -> [Outer, Inner] (Broadcast LogSum)
    int result_node = mk_op(ir::OpType::Sub, {static_cast<size_t>(shifted_node), static_cast<size_t>(log_sum_node)}, in_shape);

    return result_node;
}

} // namespace graph
} // namespace vectoria
