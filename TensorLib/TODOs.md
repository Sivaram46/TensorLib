TODOs
- [x] Tensor slice
- [x] Iterator over the tensor
- [ ] Printing tensors
    - [x] Integers
    - [x] Fixed, scientific and precision for double
    - [ ] Linewidths
    - [ ] Print only summary
- [x] operator[]
- [x] Runtime tensors
- [x] Tensor.reshape(), ravel() 
- [x] squeeze(), expand_dims()
- [ ] Compile-time tensors

    - Use variadic templates `size_t...` in tensor to mention compile time dimensions. Specialize the tensor class for runtime tensors. 

    - **Restrictions:** Had to take copy whenever slicing the tensor. Provide a runtime() funtion that will convert the compile-time tensor to the run-time one.

    - While reshaping, when the number of arguments not matching the dimension, return a runtime tensor.

    - Return compile/runtime tensors for ravel(), expand_dims() based on the tensor we are operating.

    - Return run-time tensor for squeeze()

- [ ] Braced initialization list
- [x] 0 dimensional tensor
- [ ] Tensor::reverse() 
- [ ] Tensor::transpose() or permute()
- [ ] Tensor broadcasting
- [ ] CMake file for setting up the library
- [ ] Tests
- [ ] Tensor operations to avoid making copies in binary operations
- [ ] Python bindings
- [ ] Tensor specialization for Matrix
- [ ] Binary operations on type different tensors

Bugs:
- [x] operator[] for N <= 1

Improvement:
- [ ] Clearup unused memory when copying a tensor and the parent tensor went out of scope
- [ ] Vector processing