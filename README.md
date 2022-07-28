# TensorLib
A multi-dimensional C++ tensor library for arbitrary data types.

## Usage
```cpp
using TL::Tensor;

/* Construct by elements and shape. */
Tensor<int> A({0, 1, 2, 3, 4, 5}, {2, 3});
/*      A = [[0, 1, 2]
             [3, 4, 5]] */

/* Construct by shape only. A 3-dimensional tensor. */
Tensor<double> B(1, 4, 3);

/* Construct by TL::Range and shape. */
Tensor<int> C(TL::Range(24), {2, 3, 4});
```
Refer `examples/` for other ways to constructing a tensor. 

Almost all arithmetic operators are overloaded for a tensor to enable them using in arithmetic expressions. The arithmetic operations are supported between tensors of same type and dimension as of now.

```cpp
/* Assign a tensor to a scalar. Tensor of all 4s */
B = 4.0;

/* Can be used in an arithmetic expressions. */
auto D = A + (A * 2);
```
### Accessing Elements
Tensors are accessed to get individual elements / subtensors by the overloaded `operator()`. Tensors can also be accessed by `operator[]`. Accessing always returns reference to tensors, so changing value in subtensors will reflect in the original tensor. To avoid references use `Tensor::copy()`.

```cpp
/* Access elements by just calling the tensor. */
auto& elem = A(0, 1);     // elem = 1 

/* Accessing returns a lvalue reference. */
A(0, 1) = 10;     // elem = 10

/* Can be sliced usign TL::Slice and TL::Range. */
using TL::Slice;
using R = TL::Range;
auto sliced = A(Slice(1, R(2)));    // A[1, :2] = [[3, 4]]

/* Slicing returns a reference to the original tensor. */
sliced(0, 0) = 100;   // A(1, 0) = 100

/* Acessing by subscript operator */
auto A1 = A[1];
// A1 of shape (3)
```

Range for-loops traverses a tensor in flattened version. For example.
```cpp
for (auto& x : A) {
    cout << x << ", ";
    // 0, 10, 2, 100, 4, 5
}
```

### Printing Tensors
Tensors can be printed to different output streams by `Tensor::print()`. Can be printed to ostreams by the overloaded `operator<<`.
```cpp
cout << "C = \n" << C; 
/* 
C = 
[[[ 0,  1,  2,  3]
  [ 4,  5,  6,  7]
  [ 8,  9, 10, 11]]

 [[12, 13, 14, 15]
  [16, 17, 18, 19]
  [20, 21, 22, 23]]]
*/
```
`Tensor.format` provides different format options for printing a tensor like precision, linewidths, float modes etc.
```cpp
Tensor<double> P(1, 1, 3);
P = 3.1415926;
P.format.precision = 3; /* set to 3 precision */
cout << "P = \n" << P; 
/* 
P =
[[[3.14, 3.14, 3.14]]]
*/
```

## Setting up and Compilation
The library can be compiled and ran by adding the `TensorLib`'s path to the include path of the compiler (gcc here).

```cpp
/* main.cpp */
...
#include <TensorLib/tensor_core>

int main() {
    TL::Tensor<double> A;
    ...
}
```
```bash
$ g++ -I /path/to/TensorLib/ main.cpp -o main
$ ./main
```
