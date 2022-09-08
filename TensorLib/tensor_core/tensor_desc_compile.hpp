#ifndef TESNORLIB_TENSOR_DESCRIPTOR_COMPILE_H_
#define TESNORLIB_TENSOR_DESCRIPTOR_COMPILE_H_

#include <array>
#include <exception>
#include <cstddef>

namespace TL {

namespace internal {

template <typename T, size_t N>
class TensorDescriptor
{
public:
    TensorDescriptor();

    template <typename... Dims>
    TensorDescriptor(size_t, Dims...);

    template <typename... Dims>
    size_t operator()(Dims...) const;

private:
    size_t start = 0;
    std::array<size_t, N> shape;
    std::array<size_t, N> stride;

    void update_stride();

    template <typename... Dims>
    bool check_bound(Dims...) const;
};

template <typename T, size_t N>
void TensorDescriptor<T, N>::update_stride() {
    /* Logic:
        The strides of a N-dimensional tensor of shape (s_0, s_1, ..., s_n-1) and
        strides (t_0, t_1, ..., t_n-2, t_n-1) have the value of 
        t_n-1 = 1
        t_n-2 = s_n-1 = t_n-1 * s_n-1
        t_n-3 = t_n-2 * s_n-2
        .
        .
        .
        t_n-i = t_n-i-1 * s_n-i-1
    */
    stride[N - 1] = 1;
    for (size_t i = N - 2; i--;) {
        stride[i] = stride[i+1] * shape[i+1]; 
    }
}

template <typename T, size_t N>
template <typename... Dims>
bool TensorDescriptor<T, N>::check_bound(Dims... dims) const {
    std::array<size_t, N> idx { size_t(dims)... };
    for (size_t i = 0; i < N; ++i) {
        if (idx[i] >= shape[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, size_t N>
template <typename... Dims>
TensorDescriptor<T, N>::TensorDescriptor(size_t _start, Dims... dims) 
: start(_start), shape{ dims... } {
    update_stride();
}

template <typename T, size_t N>
template <typename... Dims>
size_t TensorDescriptor<T, N>::operator()(Dims... dims) const {
    // Check whether there are N dimensions in dims...
    static_assert(sizeof...(dims) == N);

    // Check whether all the dimensions are within range of the tensor.
    if (!check_bound(dims...)) {
        throw std::out_of_range("Index out of range");
    }

    std::vector<size_t> indices { size_t(dims)... };
    /* Index in the flat vector is just inner product of strides and given indices
    plus the start. */
    size_t res = start;
    for (size_t i = 0; i < indices.size(); ++i) {
        res += (indices[i] * stride[i]);
    }
    
    return res;
}

}   // namespace internal

}   // namespace TL


#endif  // TESNORLIB_TENSOR_DESCRIPTOR_COMPILE_H_