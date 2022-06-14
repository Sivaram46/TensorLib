#include <vector>
#include <iostream>

#include "tensor.cpp"
#include "tensor_descriptor.cpp"

using namespace std;

int main()
{
    vector<double> vec = {0, 1, 2, 3, 4, 5};

    LA::Tensor<double, 2> ten1 (vec, {2, 3});
    LA::Tensor<double, 2> ten2 (2, 3);
    ten2 = 4.0;

    LA::Tensor<double, 2> ten3 = ten1 / 2.0;
    ten1(0, 0) = 10;

    cout << ten1 << "\n\n";
    cout << ten2 << "\n\n";
    cout << ten3 << "\n";
}