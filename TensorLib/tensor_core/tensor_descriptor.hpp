#ifndef TENSORLIB_TENSOR_DESCRIPTOR_H_
#define TENSORLIB_TENSOR_DESCRIPTOR_H_

#include "utils.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <exception>

namespace TL {

template <typename T>
class TensorIterator;

template <typename T>
class Tensor;

namespace internal {

/**************************************************
            TensorDescriptor declaration 
 **************************************************/

/**
 * @brief Holds information about the tensor like shape, strides.
 */
class TensorDescriptor
{
public:
    template <typename T> friend class TL::Tensor;
    template <typename T> friend class TL::TensorIterator;

    /** 
     * @brief The default constructor. Sets the size and dimension of the Tensor to 0.
     */
    TensorDescriptor() : sz(1), n_dim(0) {}

    /** 
     * @brief Constructs descriptor from shape along each dim and optional start offset.
     * @param _shape The shape of the tensor along each dimension.
     * @param _st (Optional) The offset for subtensors.
     */
    TensorDescriptor(const std::vector<size_t>&, size_t = 0);

    /** 
     * @brief Constructs from shapes only.
     * @param dims... The shape of the Tensor along each dimension.
     */
    template<typename... Dims>
    TensorDescriptor(Dims...);

    TensorDescriptor(const TensorDescriptor&) = default;

    /**
     * @brief Returns the corresponding index in flat vector given the
     * indices in Tensor.
     * @param dims... Indices along each dimension. Should all be convertible to 
     * size_t
     * @return The index of element in flat vector.
     * @throw std::out_of_range if any of the given indices goes out of range. 
     * @throw std::runtime_error when @a dims... and @a n_dim mismatch.
     */
    template <typename... Dims>
    std::enable_if_t<
        TL::Element_valid<Dims...>(),
    size_t> operator()(Dims...) const;

    /**
     * @brief Returs the number of elements in the tensor.
     */
    size_t size() const {
        return sz;
    }

    /**
     * @brief Returns the dimension of the tensor.
     */
    size_t ndim() const {
        return n_dim;
    }

private:
    size_t sz;
    size_t n_dim;
    size_t start = 0;   /* offset for subtensors */
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    
    /**
     * @brief A utility function that calculates and updates the stride information
     * given the shape information.
     */
    void _calculate_stride();

    /** @brief A utility function that checks whether the given indices are within the
    * shape bounds. 
    * @param dims... Indices along each dimension.
    * @return @a true if indices are valid, else @a false.
    */
    template <typename... Dims>
    bool _check_bound(Dims...) const;
};

/**************************************************
            TensorDescriptor definition 
 **************************************************/

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
    /* size can be calculated from multiplying all the shapes. */
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
    /* Index in the flat vector is just inner product of strides and given indices
    plus the start. */
    return start + std::inner_product(
        indices.begin(), indices.end(), stride.begin(), size_t(0)
    );
}

}   // namespace interanl

}   // namespace TL

#endif  // TENSORLIB_TENSOR_DESCRIPTOR_H_