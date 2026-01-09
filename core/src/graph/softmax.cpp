#include "vectoria/graph_ops.hpp"
#include <stdexcept>
#include <variant>

namespace vectoria {
namespace graph {

int add_softmax_composed(ir::Graph& graph, int input_id) {
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
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    ir::TensorShape in_shape = get_shape(input_id);
    if (in_shape.dims.empty()) throw std::runtime_error("Softmax input cannot be empty");

    // reduce_max(x, axis=-1) -> [Outer] (assuming last dim reduced)
    ir::TensorShape reduced_shape = in_shape;
    reduced_shape.dims.pop_back(); 
    
    // 1. Max = ReduceMax(Input)
    int max_node = mk_op(ir::OpType::ReduceMax, {static_cast<size_t>(input_id)}, reduced_shape);

    // 2. Sub = Sub(Input, Max) -> [Outer, Inner]
    // The engine handles the broadcast if second input has 1 less dim or count=1
    int sub_node = mk_op(ir::OpType::Sub, {static_cast<size_t>(input_id), static_cast<size_t>(max_node)}, in_shape);

    // 3. Exp = Exp(Sub) -> [Outer, Inner]
    int exp_node = mk_op(ir::OpType::Exp, {static_cast<size_t>(sub_node)}, in_shape);

    // 4. Sum = ReduceSum(Exp) -> [Outer]
    int sum_node = mk_op(ir::OpType::ReduceSum, {static_cast<size_t>(exp_node)}, reduced_shape);

    // 5. Div = Div(Exp, Sum) -> [Outer, Inner]
    int div_node = mk_op(ir::OpType::Div, {static_cast<size_t>(exp_node), static_cast<size_t>(sum_node)}, in_shape);

    return div_node;
}

} // namespace graph
} // namespace vectoria
