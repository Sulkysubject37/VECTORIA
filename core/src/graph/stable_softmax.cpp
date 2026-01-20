#include "vectoria/graph/stable_softmax.hpp"
#include "vectoria/graph/logsoftmax.hpp"
#include <stdexcept>
#include <variant>
#include <vector>

namespace vectoria {
namespace graph {

int add_softmax_stable_composed(ir::Graph& graph, int input_id) {
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

    // 1. LogSoftmax
    int log_softmax_node = add_logsoftmax_composed(graph, input_id);

    // 2. Exp(LogSoftmax)
    ir::TensorShape shape = get_shape(log_softmax_node);
    int exp_node = mk_op(ir::OpType::Exp, {static_cast<size_t>(log_softmax_node)}, shape);

    return exp_node;
}

} // namespace graph
} // namespace vectoria
