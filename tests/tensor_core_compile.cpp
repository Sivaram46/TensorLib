#include <vector>
#include <iostream>
#include <cassert>

#include "TensorLib/tensor_core/tensor_compile.hpp"

using namespace std;

void test_compile_tensor()
{
    TL::Tensor<int, 2, 3, 2> A;
    TL::Tensor<double, 2, 3> B ({2, 3, 4, 1, 4, 4});
    cout << B(0, 1) << "\n";
}

int main()
{
    test_compile_tensor();
}