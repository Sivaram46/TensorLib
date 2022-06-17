#include <vector>
#include <iostream>

#include "tensor.cpp"
#include "tensor_descriptor.cpp"
#include "utils.cpp"

using namespace std;

int main()
{
    vector<double> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    LA::Tensor<double, 2> ten1 (vec, {3, 4});
    LA::Tensor<double, 2> ten2 (2, 3);
    ten2 = 4.0;

    LA::Tensor<double, 2> ten3 = ten1 / 2.0;
    ten1(0, 0) = 10;

    cout << ten1 << "\n\n";
    cout << "Shape: " << ten1.shape(0) << " " << ten1.shape(1) << "\n\n";
    // cout << ten2 << "\n\n";
    // cout << ten3 << "\n\n";

    using R = LA::Range;
    using LA::Slice;
    auto roi = ten1(Slice(R(3), R(1, 3)));
    cout << roi << "\n\n";
    cout << "Shape: " << roi.shape(0) << " " << roi.shape(1) << "\n\n";
    
    auto roi2 = roi(Slice(R(1), R(2)));
    cout << roi2 << "\n\n";
    cout << "Shape: " << roi2.shape(0) << " " << roi2.shape(1) << "\n\n";

    auto roi3 = roi2(Slice(0, R(0, 1)));
    cout << roi3 << "\n\n";
    cout << "Shape: " << roi3.shape(0) << " " << roi3.shape(1) << "\n\n";

    roi3(0,0) = 100;
}