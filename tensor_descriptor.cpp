#ifndef LA_TENSOR_DESCRIPTOR_CPP
#define LA_TENSOR_DESCRIPTOR_CPP

#include <numeric>
#include <algorithm>
#include <array>
#include <exception>

#include "tensor_descriptor.hpp"

namespace LA {

template <size_t N>
void TensorDescriptor<N>::_calculate_stride() {
    stride[0] = shape[1];
    stride[1] = 1;
    stride[2] = shape[0] * shape[1];

    for (size_t i = 3; i < N; ++i) {
        stride[i] = stride[i-1] * shape[i-1];
    }
}

template <size_t N>
TensorDescriptor<N>::TensorDescriptor(size_t _sz, const std::array<size_t, N>& _shape) 
: sz(_sz) {
    // Copy the shape values
    std::copy(_shape.begin(), _shape.end(), shape.begin());
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
template <typename... Indices>
bool TensorDescriptor<N>::_check_bound(Indices... indices) const {
    std::array<size_t, N> idx { size_t(indices)... };
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
    All(Is_convertible<Dims, size_t>()...),
size_t> TensorDescriptor<N>::operator()(Dims... dims) const {    /* Member signature */
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