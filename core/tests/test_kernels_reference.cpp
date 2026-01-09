#include "vectoria/kernels.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace vectoria::kernels::reference;

void test_add() {
    std::cout << "Testing Reference Add..." << std::endl;
    std::vector<float> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> b = {10.0, 20.0, 30.0, 40.0};
    std::vector<float> out(4);
    
    add_f32(a.data(), b.data(), out.data(), 4);
    
    assert(out[0] == 11.0);
    assert(out[1] == 22.0);
    assert(out[2] == 33.0);
    assert(out[3] == 44.0);
    std::cout << "PASSED" << std::endl;
}

void test_mul() {
    std::cout << "Testing Reference Mul..." << std::endl;
    std::vector<float> a = {2.0, 3.0, 4.0, 5.0};
    std::vector<float> b = {0.5, 2.0, -1.0, 0.0};
    std::vector<float> out(4);
    
    mul_f32(a.data(), b.data(), out.data(), 4);
    
    assert(out[0] == 1.0);
    assert(out[1] == 6.0);
    assert(out[2] == -4.0);
    assert(out[3] == 0.0);
    std::cout << "PASSED" << std::endl;
}

void test_reduce_sum() {
    std::cout << "Testing Reference ReduceSum..." << std::endl;
    // 2x3 Matrix
    // [1, 2, 3] -> sum = 6
    // [4, 5, 6] -> sum = 15
    std::vector<float> input = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<float> out(2);
    
    reduce_sum_f32(input.data(), out.data(), 2, 3);
    
    assert(out[0] == 6.0);
    assert(out[1] == 15.0);
    std::cout << "PASSED" << std::endl;
}

void test_reduce_max() {
    std::cout << "Testing Reference ReduceMax..." << std::endl;
    std::vector<float> input = {1.0, 5.0, 3.0, 4.0, 2.0, 6.0}; // [1,5,3], [4,2,6]
    std::vector<float> out(2);
    
    reduce_max_f32(input.data(), out.data(), 2, 3);
    
    assert(out[0] == 5.0);
    assert(out[1] == 6.0);
    std::cout << "PASSED" << std::endl;
}

void test_exp() {
    std::cout << "Testing Reference Exp..." << std::endl;
    std::vector<float> input = {0.0, 1.0, -1.0};
    std::vector<float> out(3);
    
    exp_f32(input.data(), out.data(), 3);
    
    assert(std::abs(out[0] - 1.0f) < 1e-5f);
    assert(std::abs(out[1] - 2.71828f) < 1e-5f);
    assert(std::abs(out[2] - 0.367879f) < 1e-5f);
    std::cout << "PASSED" << std::endl;
}

void test_sub_broadcast() {
    std::cout << "Testing Reference Sub (Broadcast)..." << std::endl;
    // [2, 3] - [2]
    // [[1, 2, 3], [10, 20, 30]] - [1, 10]
    // -> [[0, 1, 2], [0, 10, 20]]
    std::vector<float> a = {1.0, 2.0, 3.0, 10.0, 20.0, 30.0};
    std::vector<float> b = {1.0, 10.0};
    std::vector<float> out(6);
    
    sub_broadcast_f32(a.data(), b.data(), out.data(), 2, 3);
    
    assert(out[0] == 0.0); assert(out[1] == 1.0); assert(out[2] == 2.0);
    assert(out[3] == 0.0); assert(out[4] == 10.0); assert(out[5] == 20.0);
    std::cout << "PASSED" << std::endl;
}

void test_div_broadcast() {
    std::cout << "Testing Reference Div (Broadcast)..." << std::endl;
    // [2, 3] / [2]
    // [[2, 4, 6], [20, 40, 60]] / [2, 10]
    // -> [[1, 2, 3], [2, 4, 6]]
    std::vector<float> a = {2.0, 4.0, 6.0, 20.0, 40.0, 60.0};
    std::vector<float> b = {2.0, 10.0};
    std::vector<float> out(6);
    
    div_broadcast_f32(a.data(), b.data(), out.data(), 2, 3);
    
    assert(out[0] == 1.0); assert(out[1] == 2.0); assert(out[2] == 3.0);
    assert(out[3] == 2.0); assert(out[4] == 4.0); assert(out[5] == 6.0);
    std::cout << "PASSED" << std::endl;
}

int main() {
    test_add();
    test_mul();
    test_reduce_sum();
    test_reduce_max();
    test_exp();
    test_sub_broadcast();
    test_div_broadcast();
    return 0;
}
