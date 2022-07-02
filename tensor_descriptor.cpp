#ifndef LA_TENSOR_DESCRIPTOR_CPP
#define LA_TENSOR_DESCRIPTOR_CPP

#include <numeric>
#include <algorithm>
#include <array>
#include <exception>

#include "tensor_descriptor.hpp"

namespace TL {

template <size_t N>
void TensorDescriptor<N>::_calculate_stride() {
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
    for (int i = N-2; i >= 0; --i) {
        stride[i] = stride[i+1] * shape[i+1]; 
    }
}

template <size_t N>
TensorDescriptor<N>::TensorDescriptor(size_t _sz, const std::array<size_t, N>& _shape) 
: sz(_sz), shape(_shape) {
    _calculate_stride();
}

template <size_t N>
template <typename... Dims>
TensorDescriptor<N>::TensorDescriptor(Dims... dims) {
    static_assert(sizeof...(dims) == N, "Dimensions Mismatch");

    std::array<size_t, N> _shape { size_t(dims)... };

    std::copy(_shape.begin(), _shape.end(), shape.begin());
    _calculate_stride();

    sz = std::accumulate(
        shape.begin(), shape.end(), (size_t) 1, 
        [] (size_t a, size_t b) {return a * b;}
    );   
}

template <size_t N>
template <typename... Dims>
bool TensorDescriptor<N>::_check_bound(Dims... dims) const {
    std::array<size_t, N> idx { size_t(dims)... };
    for (size_t i = 0; i < N; ++i) {
        if (idx[i] >= shape[i]) {
            return false;
        }
    }
    return true;
}

template <size_t N>
template <typename... Dims>
std::enable_if_t<
    TL::Element_valid<Dims...>(),
size_t> TensorDescriptor<N>::operator()(Dims... dims) const {
    static_assert(sizeof...(dims) == N, "");
    if (!_check_bound(dims...)) {
        throw std::out_of_range("Index out of range");
    }

    std::array<size_t, N> indices { size_t(dims)... };
    return start + std::inner_product(
        indices.begin(), indices.end(), stride.begin(), size_t(0)
    );
}

}

#endif