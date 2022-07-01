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
    /* Logic:
        The strides of a N-dimensional tensor of shape (s0, s1, ..., sn-1) and
        strides (t0, t1, ...., tn-1) has the value of 
        t0 = s1, since for moving from one row to next row have traverse #col
            elements. (In row major indexing)
        t1 = 1, next col elements would noramlly the next element the the flat 
            vector.
        t2 = s0 * s1, for moving to next axis have to traverse s0*s1 elements.
        and so on.
    */
    stride[0] = shape[1];
    stride[1] = 1;

    if (N > 2) {
        stride[2] = shape[0] * shape[1];
    }

    for (size_t i = 3; i < N; ++i) {
        stride[i] = stride[i-1] * shape[i-1];
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
    LA::Element_valid<Dims...>(),
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