# TensorLib
A multi-dimensional C++ tensor library which provides compile time tensor for arbitrary built-in data types.

## Usage
```cpp
using TL::Tensor;

/* Construct by elements and shape. */
Tensor<int, 2> ten({0, 1, 2, 3, 4, 5}, {2, 3});
/* ten = [[0, 1, 2]
         [3, 4, 5]] */

/* Construct by shape only. An empty 3D tensor. */
Tensor<double, 3> ten3d(1, 4, 3);
```
Refer `examples/` for other ways to constructing a tensor. 

Almost all arithmetic operators are overloaded for a tensor to enable them using in arithmetic expressions. The arithmetic operations are supported between tensors of same type and dimension as of now.

```cpp
/* Assign a tensor to a scalar. Tensor of all 4s */
ten3d = 4.0;

/* Can be used in an arithmetic expressions. */
auto tensor_op = ten + (ten * 2);
```

There are two ways for accessing the elements in tensor. Both of them return references, so changing the result might reflect to the base tensor. To avoid references use `Tensor::copy`.

```cpp
/* Access elements by just calling the tensor. */
auto& elem = ten(0, 1);     // elem = 1 

/* Accessing returns a lvalue reference. */
ten(0, 1) = 10;     // elem = 10

/* Can be sliced usign TL::Slice and TL::Range. */
using TL::Slice;
using R = TL::Range;
auto sliced = ten(Slice(1, R(2)));    // ten[1, :2] = [[3, 4]]

/* Slicing returns a reference to the original tensor. */
sliced(0, 0) = 100;   // ten(1, 0) = 100
```

Range for-loops traverses a tensor in flattened version. For example.
```cpp
for (auto& x : ten) {
    cout << x << ", ";
    // 0, 10, 2, 100, 4, 5
}
```

The library can be compiled and ran by,

```cpp
/* main.cpp */
...
#include <TensorLib/tensor_core>

int main() {
    TL::Tensor<double, 2> A;
    ...
}
```
```bash
$ g++ -I /path/to/TensorLib/ main.cpp -o main
$ ./main
```
