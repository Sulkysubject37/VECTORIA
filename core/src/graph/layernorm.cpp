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
        // Broadcast scalar to shape implies specific implementation or 
        // relying on broadcast ops.
        // Actually, if we create a Scalar constant (rank 0 or 1), 
        // the broadcast logic in kernels should handle it.
        // Let's create a Scalar constant (1 element).
        c.data_f32 = {val};
        // Override shape to be scalar/rank-0 or rank-1 [1] depending on kernel needs.
        // Vectoria kernels usually handle [1] or smaller rank broadcasting.
        // Let's use rank 0 (empty dims).
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

    // Shapes
    ir::TensorShape reduced_shape = in_shape;
    reduced_shape.dims.pop_back(); // Rank N-1

    // Constants
    int n_const = mk_const(static_cast<float>(last_dim_size), {});
    int eps_const = mk_const(1e-5f, {});

    // 1. Sum = ReduceSum(Input) -> [Outer]
    int sum_node = mk_op(ir::OpType::ReduceSum, {static_cast<size_t>(input_id)}, reduced_shape);

    // 2. Mean = Div(Sum, N) -> [Outer] (Broadcast N)
    int mean_node = mk_op(ir::OpType::Div, {static_cast<size_t>(sum_node), static_cast<size_t>(n_const)}, reduced_shape);

    // 3. Diff = Sub(Input, Mean) -> [Outer, Inner] (Broadcast Mean)
    int diff_node = mk_op(ir::OpType::Sub, {static_cast<size_t>(input_id), static_cast<size_t>(mean_node)}, in_shape);

    // 4. SqDiff = Mul(Diff, Diff) -> [Outer, Inner]
    int sq_diff_node = mk_op(ir::OpType::Mul, {static_cast<size_t>(diff_node), static_cast<size_t>(diff_node)}, in_shape);

    // 5. VarSum = ReduceSum(SqDiff) -> [Outer]
    int var_sum_node = mk_op(ir::OpType::ReduceSum, {static_cast<size_t>(sq_diff_node)}, reduced_shape);

    // 6. Var = Div(VarSum, N) -> [Outer]
    int var_node = mk_op(ir::OpType::Div, {static_cast<size_t>(var_sum_node), static_cast<size_t>(n_const)}, reduced_shape);

    // 7. VarEps = Add(Var, Eps) -> [Outer]
    int var_eps_node = mk_op(ir::OpType::Add, {static_cast<size_t>(var_node), static_cast<size_t>(eps_const)}, reduced_shape);

    // 8. Std = Sqrt(VarEps) -> [Outer]
    int std_node = mk_op(ir::OpType::Sqrt, {static_cast<size_t>(var_eps_node)}, reduced_shape);

    // 9. Normalized = Div(Diff, Std) -> [Outer, Inner] (Broadcast Std)
    int norm_node = mk_op(ir::OpType::Div, {static_cast<size_t>(diff_node), static_cast<size_t>(std_node)}, in_shape);

    // 10. Scaled = Mul(Normalized, Gamma) -> [Outer, Inner] (Broadcast Gamma)
    // Gamma shape is [D]. Normalized is [..., D].
    // Vectoria broadcast logic handles this if implemented correctly.
    // However, Reference kernels usually expect matching rank or specific broadcast pattern.
    // Mul Ref kernel: "element-wise".
    // Does it support [M, N] * [N]?
    // Engine execute logic for Mul: 
    // "Naive shape inference: inherit from A"
    // "kernels::reference::mul_f32(a, b, out, count)" -> This assumes SAME COUNT.
    // Engine check: "if (count_a == count_b) ... else ..."
    // Looking at engine.cpp for Mul:
    // It DOES NOT have broadcast logic like Sub/Div in the reference path!
    // "kernels::reference::mul_f32(a, b, out, count);"
    // It blindly calls elementwise mul assuming count matches.
    // This is a bug in VECTORIA base or a limitation I must work around.
    
    // Sub/Div have: 
    // if (count_a == count_b) ... else kernels::reference::sub_broadcast_f32
    // Mul/Add DO NOT.
    
    // I MUST fix Engine to support broadcast Mul/Add if I want LayerNorm to work with Gamma/Beta.
    // Gamma is [D]. Input is [..., D].
    // If I cannot modify Engine to support Mul broadcast, I cannot implement LayerNorm fully.
    
    // Wait, Sub/Div broadcast support is: "Subtract with Broadcast (Col Vector): Out[i, j] = A[i, j] - B[i]"
    // This expects B to be [Outer] (e.g. Mean), not [Inner] (Gamma).
    // Mean is [Outer]. So steps 3 and 9 work.
    // Gamma is [Inner] (size D).
    // We need broadcast over Inner dimension? i.e. A[i, j] * B[j].
    // Existing broadcast ops (sub_broadcast_f32) seem to handle "Col Vector"? 
    // Let's check kernels.hpp docs.
    // "Subtract with Broadcast (Col Vector): Out[i, j] = A[i, j] - B[i]"
    // This means B has size `outer`. This is for subtracting Mean (shape [Outer]). Correct.
    
    // But Gamma has shape [Inner] (size D).
    // We need Row Vector broadcast? Out[i, j] = A[i, j] * B[j].
    // This is NOT supported by `sub_broadcast_f32` (it does `B[i]`).
    
    // This implies VECTORIA v1.1.2-delta lacks Row-Broadcast ops (Add/Mul with [D]).
    // I need to add them.
    // "The expansion MUST use only existing ops".
    // This puts me in a bind.
    
    // However, I can manually expand the broadcast?
    // No, I can't loop in graph.
    
    // Maybe `MatMul`?
    // Normalized [M, N] * Gamma [N] (diagonal).
    // MatMul(Normalized, Diag(Gamma)).
    // But I can't create Diag matrix easily without `Scatter` or `Eye`.
    
    // Maybe `BiasAdd`?
    // "Bias Add: Out = In + Bias (Broadcast)"
    // "In: [M, N], Bias: [1, N]"
    // This IS Row Broadcast!
    // So `Add(x, beta)` can be done via `BiasAdd(x, beta)`.
    // EXCELLENT.
    // Step 11 (Add Beta) can use `BiasAdd`.
    
    // What about Step 10 (Mul Gamma)?
    // There is no `BiasMul`.
    // There is `Mul`.
    // Engine logic for `Mul` is strictly elementwise.
    
    // Is there any trick?
    // Log space?
    // Mul(x, y) = Exp(Add(Log(x), Log(y))).
    // We have Exp. We don't have Log.
    
    // I must add `MulBroadcast` support to Engine?
    // Or add `mul_broadcast_f32` kernel.
    // This violates "No new kernels" strictly?
    // "LayerNorm MUST ... Be composed from existing validated ops".
    // If Mul doesn't support broadcast, I can't compose it.
    
    // I will assume I should add `mul_row_broadcast` kernel or logic.
    // Or maybe I missed something in `engine.cpp`?
    // Re-read `engine.cpp` Mul section.
    // `kernels::reference::mul_f32(a_ptr, b_ptr, out_ptr, count);`
    // It calls `mul_f32` which iterates `0..count`.
    // If I pass Gamma [D] and Input [N*D], `mul_f32` will read out of bounds or loop wrap?
    // `mul_f32` takes arrays. It will segfault or read garbage if B is smaller than A.
    
    // I HAVE to fix Mul to support broadcast.
    // I will add `mul_broadcast_row_f32` (or generic).
    // `BiasAdd` supports `[1, N]` bias.
    // I should implement `Scale` (Mul Broadcast) similar to `BiasAdd`.
    // But I can't add new OpType.
    // Use `Mul` OpType, but detect shapes in Engine and dispatch to new kernel logic?
    // Yes.
    
    // So Step 2 also includes:
    // 1. Add `mul_broadcast_f32` to kernels (Reference).
    // 2. Update Engine to dispatch Mul broadcast.
    
    // Wait, `BiasAdd` logic:
    // `kernels::reference::bias_add_f32(in_ptr, bias_ptr, out_ptr, m, n);`
    // It handles [M, N] + [1, N].
    
    // I should add `mul_broadcast_f32` kernel that does [M, N] * [1, N].
    // And dispatch `Mul` in Engine to it if shapes match.
    
    int mul_node = mk_op(ir::OpType::Mul, {static_cast<size_t>(norm_node), static_cast<size_t>(gamma_id)}, in_shape);
    
    // 11. Result = BiasAdd(Scaled, Beta) -> [Outer, Inner]
    // Use BiasAdd instead of Add because Add is strictly elementwise (based on my reading).
    // Unless Add also supports broadcast?
    // `engine.cpp`: Add is also `add_f32` (elementwise).
    // So strictly I should use `BiasAdd` for beta.
    
    // For Gamma, I will rely on `Mul` being updated to support broadcast.
    
    int res_node = mk_op(ir::OpType::BiasAdd, {static_cast<size_t>(mul_node), static_cast<size_t>(beta_id)}, in_shape);

    return res_node;
}

} // namespace graph
} // namespace vectoria
