#include <vector>
#include <iostream>

#include "tensor.cpp"
#include "tensor_slice.cpp"

using namespace std;

int main()
{
    vector<double> vec = {0, 1, 2, 3, 4, 5};

    LA::Tensor<double, 3> ten1 (vec, {2, 3, 1});
    LA::Tensor<double, 3> ten2 (2, 3, 1);
    ten2 = 4.0;

    LA::Tensor<double, 3> ten3 = ten1 + ten2; // + ten1;

    cout << ten1 << "\n";
    cout << ten2 << "\n";
    // cout << ten3 << "\n";
}