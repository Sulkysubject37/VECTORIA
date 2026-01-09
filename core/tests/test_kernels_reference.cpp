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

int main() {
    test_add();
    test_mul();
    test_reduce_sum();
    return 0;
}
