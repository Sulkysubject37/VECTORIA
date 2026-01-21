#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/graph_ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace vectoria;

void test_encoder_shape() {
    std::cout << "Testing Transformer Encoder Shape..." << std::endl;
    ir::Graph g;
    
    int64_t T = 2;
    int64_t d_model = 4;
    int heads = 2;
    int64_t d_ff = 8;

    ir::InputNode x_node;
    x_node.name = "X";
    x_node.shape.dims = {T, d_model};
    x_node.dtype = ir::DataType::Float32;
    int x_id = static_cast<int>(g.nodes.size());
    g.nodes.push_back({ {static_cast<size_t>(x_id)}, x_node });

    auto add_weight = [&](std::string name, std::vector<int64_t> shape) {
        ir::ParameterNode p;
        p.name = name;
        p.shape.dims = shape;
        p.dtype = ir::DataType::Float32;
        int id = static_cast<int>(g.nodes.size());
        g.nodes.push_back({ {static_cast<size_t>(id)}, p });
        return id;
    };

    int wq = add_weight("WQ", {d_model, d_model});
    int wk = add_weight("WK", {d_model, d_model});
    int wv = add_weight("WV", {d_model, d_model});
    int wo = add_weight("WO", {d_model, d_model});

    int g1 = add_weight("G1", {d_model});
    int b1 = add_weight("B1", {d_model});

    int wf1 = add_weight("WF1", {d_model, d_ff});
    int bf1 = add_weight("BF1", {d_ff});
    int wf2 = add_weight("WF2", {d_ff, d_model});
    int bf2 = add_weight("BF2", {d_model});

    int g2 = add_weight("G2", {d_model});
    int b2 = add_weight("B2", {d_model});

    int out_id = graph::add_transformer_encoder_composed(
        g, x_id, wq, wk, wv, wo, heads, g1, b1, wf1, bf1, wf2, bf2, g2, b2
    );
    g.outputs.push_back({static_cast<size_t>(out_id)});

    Engine e(g);
    e.compile();
    e.execute();

    std::cout << "Encoder Shape Passed." << std::endl;
}

int main() {
    test_encoder_shape();
    return 0;
}
