#ifndef TENSORLIB_TENSOR_DESCRIPTOR_CPP
#define TENSORLIB_TENSOR_DESCRIPTOR_CPP

#include <numeric>
#include <algorithm>
#include <vector>
#include <exception>

#include "tensor_descriptor.hpp"

namespace TL {

namespace internal {

void TensorDescriptor::_calculate_stride() {
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
    stride[ndim() - 1] = 1;
    for (long i = ndim()-2; i >= 0; --i) {
        stride[i] = stride[i+1] * shape[i+1]; 
    }
}

TensorDescriptor::TensorDescriptor(
    const std::vector<size_t>& _shape, size_t _st
) 
: shape(_shape), stride(_shape.size(), 1), start(_st), n_dim(_shape.size()) {
    sz = std::accumulate(
        shape.begin(), shape.end(), static_cast<size_t> (1), 
        [] (size_t a, size_t b) {return a * b;}
    );
    stride.reserve(ndim());
    _calculate_stride();
}

template <typename... Dims>
TensorDescriptor::TensorDescriptor(Dims... dims)
: shape({ size_t(dims)... }), stride(sizeof...(dims)), n_dim(sizeof...(dims)) {
    _calculate_stride();

    sz = std::accumulate(
        shape.begin(), shape.end(), static_cast<size_t> (1), 
        [] (size_t a, size_t b) {return a * b;}
    );   
}

template <typename... Dims>
bool TensorDescriptor::_check_bound(Dims... dims) const {
    std::vector<size_t> idx { size_t(dims)... };
    for (size_t i = 0; i < ndim(); ++i) {
        if (idx[i] >= shape[i]) {
            return false;
        }
    }
    return true;
}

template <typename... Dims>
std::enable_if_t<
    TL::Element_valid<Dims...>(),
size_t> TensorDescriptor::operator()(Dims... dims) const {
    if (sizeof...(dims) != ndim()) {
        throw std::runtime_error("Dimensions Mismatch");
    }

    if (!_check_bound(dims...)) {
        throw std::out_of_range("Index out of range");
    }

    std::vector<size_t> indices { size_t(dims)... };
    return start + std::inner_product(
        indices.begin(), indices.end(), stride.begin(), size_t(0)
    );
}

}   // namespace internal

}   // namespace TL

#endif