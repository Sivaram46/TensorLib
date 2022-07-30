#include <vector>
#include <iostream>
#include <cassert>

#include "../TensorLib/tensor_core"

using namespace std;

using R = TL::Range;
using Slice = TL::Slice;
using TL::operator<<;

void test_constructs()
{
    // Constructs by std::vector
    vector<int> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    TL::Tensor<int> A (vec, {3, 4});  // A of shape (3, 4)
    assert(A(0, 0) == vec[0]);

    // Constructs empty tensor of given shape
    TL::Tensor<double> B (2, 3);     // B of shape (2, 3)
    B = 4.0;
    assert(B(1, 2) == 4.0);

    // Constructs by TL::Range
    TL::Tensor<int> C (R(24), {2, 3, 4});
    assert(C(0, 1, 2) == 6);

    // Copy constructor - makes a reference
    TL::Tensor<int> A_ref (A);
    A_ref(0, 1) = 100;
    assert(A(0, 1) == A_ref(0, 1));
    
    // Tensor::copy() makes a copy to the tensor. Changing an element doesn't 
    // reflect in its copy.
    auto A_copy = A.copy();
    A_copy(0, 1) = 50;
    assert(A(0, 1) != A_copy(0, 1));
}

void test_arithmetic_op()
{
    TL::Tensor<int> A(R(6), {2, 3}), B(2, 3);
    B = 3;
    auto C = (A / B) + (A * 2);
    C -= 4;
}

void test_slice()
{
    TL::Tensor<int> A(R(60), {3, 4, 5});
    auto A1 = A(Slice(R(1, 3), R(3), 2)); // [1:3, :3, 2]
    auto A2 = A1(Slice(R(2), 0, 0));

    // Using operator[]
    auto B1 = A[1][0];
}

void test_iterator()
{
    // Arithmetic and deref operations of iterator
    TL::Tensor<int> A(R(60), {3, 4, 5});
    auto it = A.begin();
    assert(*it == 0);

    it++;
    assert(*it == 1);

    it += 10;
    assert(*it == 11);

    it = it - 5;
    assert(*it == 6);

    // it will point to A.end() and it shouldn't be deferenced.
    it = A.begin();
    assert(it + A.size() == A.end());

    try {
        it = A.begin() - 1;
    } catch(std::out_of_range& e) {
        cout << e.what() << "\n";
    }

    // Equality operators
    it = A.begin();
    ++it;
    auto it2 = A.begin() + 1;
    assert(it == it2);

    it2--;
    assert(it != it2);

    // Range for loop
    for(auto& elem : A) {}
}

void test_const_iterator()
{
    TL::Tensor<int> A (R(8), {2, 4});
    const TL::Tensor<int> B (R(8), {2, 4});
    auto cit = A.cbegin();
    cit++;
    assert(*cit == 1);

    auto cit2 = B.begin();
    cit2 += 2;
    assert(*cit2 == 2);
}

void test_print()
{
    TL::Tensor<int> A(R(60), {3, 4, 5});
    A(0, 0, 0) = 100;
    cout    << "A = \n"
            << A;
    cout << "Shape: " << A.shape() << "\n\n";
    
    TL::Tensor<double> B(1, 3, 1);
    B = 3.1415926;
    auto default_format = B.format;
    B.format.precision = 4;
    cout    << "B = \n"
            << B;
    cout << "Shape: " << B.shape() << "\n\n";
        
    // Printing empty tensor
    cout    << "Empty Tensor: "
            << TL::Tensor<int>()
            << "\n";

    B.format = default_format;
    B.format.float_mode = TL::TensorFormatter::FloatMode::Scientific;
    cout    << "Scientific float format\n"
            << B
            << "\n";

    // Printing matrix
    TL::Tensor<int> M (2, 3);
    M = 5;
    cout    << "M = \n"
            << M;
    cout << "Shape: " << M.shape() << "\n\n";

    // Printing vector
    TL::Tensor<double> V (5);
    V = 2.8182;
    cout    << "V = \n"
            << V;
    cout << "Shape: " << V.shape() << "\n\n";
}

void test_reshape_squeeze()
{
    // reshape
    TL::Tensor<int> A (R(24), {2, 3, 4});
    auto B = A.reshape(4, 6);
    assert(B.shape() == vector<size_t>({4, 6}));
    assert(B.ndim() == 2);

    // squeeze
    B = A.reshape(1, 1, 24);
    auto C = B.squeeze();
    assert(C.shape() == vector<size_t>({24}));
    assert(C.ndim() == 1);

    C = B.squeeze(0);
    assert(C.shape() == vector<size_t>({1, 24}));
    assert(C.ndim() == 2);

    // expand_dim
    auto D = A.expand_dims(1);
    assert(D.shape() == vector<size_t>({2, 1, 3, 4}));
    assert(D.ndim() == A.ndim() + 1);

    // ravel
    auto E = A.ravel();
    assert(E.shape() == std::vector<size_t>({A.size()}));
    assert(E.ndim() == 1);
}

int main()
{   
    test_constructs();
    test_arithmetic_op();
    test_slice();
    test_iterator();
    test_const_iterator();
    test_print();
    test_reshape_squeeze();   
}