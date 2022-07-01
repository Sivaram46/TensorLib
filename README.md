# TensorLib
A multi-dimensional C++ tensor library which provides compile time tensor for arbitrary built-in data types.

## Usage
```cpp
using LA::Tensor;

/* Construct by elements and shape. */
Tensor<double, 2> ten({0, 1, 2, 4, 5, 6}, {2, 3});

/* Construct by shape only. An empty 3D tensor. */
Tensor<int, 3> ten3d(1, 4, 3);

/* Assign a tensor to a scalar. Tensor of all 4s */
ten3d = 4.0;

/* Access elements by just calling the tensor. */
auto& elem = ten(0, 1);     // elem = 1 

/* Accessing returns a lvalue reference. */
ten(0, 1) = 10;     // elem = 10

/* Can be used in an arithmetic expressions. */
auto ten2 = ten + (ten * 2);

/* Can be sliced usign LA::Slice and LA::Range. */
using LA::Slice;
using R = LA::Range;
auto rect = ten(Slice(1, R(2)));    // == ten[1, :2]

/* Slicing returns a reference to the original tensor. */
rect(0, 0) = 100;   // ten(1, 0) = 100
```

The library can be compiled and ran by,

```cpp
/* main.cpp */
...
#include "tensor_include.hpp"

int main() {
    LA::Tensor<double, 2> ten;
    ...
}
```
```bash
$ g++ main.cpp -o main
$ ./main
```

TODOs
- [x] Tensor slice
- [ ] Iterator over the tensor
- [ ] Printing tensors
- [ ] Tensor.reverse() 
- [ ] Tensor specialization for Matrix
- [ ] matmul(), transpose()
- [ ] squeeze(), expand_dims(), reshape()
- [ ] Binary operations on type different tensors
- [ ] CMake file for setting up the library in some machine
- [ ] Running time tensors
