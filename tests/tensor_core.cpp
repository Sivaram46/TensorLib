#include <vector>
#include <iostream>

#include "../TensorLib/tensor_core"

using namespace std;

int main()
{
    vector<double> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    TL::Tensor<double, 2> ten1 (vec, {3, 4});
    TL::Tensor<double, 2> ten2 (2, 3);
    ten2 = 4.0;

    TL::Tensor<double, 2> ten3 = ten1 / 2.0;
    ten1(0, 0) = 10;

    cout << ten1;
    cout << "Shape: " << ten1.shape(0) << " " << ten1.shape(1) << "\n\n";
    // cout << ten2 << "\n\n";
    // cout << ten3 << "\n\n";

    using R = TL::Range;
    using TL::Slice;
    // auto roi = ten1(Slice(R(3), R(1, 3)));
    // cout << roi << "\n\n";
    // cout << "Shape: " << roi.shape(0) << " " << roi.shape(1) << "\n\n";
    
    // auto roi2 = roi(Slice(R(1), R(2)));
    // cout << roi2 << "\n\n";
    // cout << "Shape: " << roi2.shape(0) << " " << roi2.shape(1) << "\n\n";

    // auto roi3 = roi2(Slice(0, R(0, 1)));
    // cout << roi3 << "\n\n";
    // cout << "Shape: " << roi3.shape(0) << " " << roi3.shape(1) << "\n\n";

    // roi3(0,0) = 100;

    cout << string(50, '-') << "\n";
    
    TL::Tensor<int, 3> range_ten(R(60), {3, 4, 5});
    auto sliced = range_ten(Slice(R(1, 3), R(3), R(2, 5))); // range_ten[1:3, :3, 2:]

    auto it = sliced.begin() + sliced.size();
    auto it2 = sliced.end();
    cout << (it == it2) << "\n";

    for (auto it = sliced.begin(); it != sliced.end(); ++it) {
        cout << *it << " ";
    }
    cout << "\n";

    range_ten(0, 0, 0) = 100;
    cout << range_ten << "\n";

    TL::Tensor<double, 3> one_element(1, 3, 1);
    one_element = 3.1415926;
    one_element.format.precision = 5;
    // one_element.format.float_mode = TL::TensorFormatter::FloatMode::Fixed;
    one_element(0, 0, 0) = 0; 
    cout << one_element << "\n";

    cout << TL::Tensor<int, 4>() << "\n";
}