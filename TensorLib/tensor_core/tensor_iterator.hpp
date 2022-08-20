#ifndef TENSORLIB_TENSOR_ITERATOR_H_
#define TENSORLIB_TENSOR_ITERATOR_H_

#include "tensor.hpp"
#include "tensor_descriptor.hpp"

#include <memory>
#include <vector>
#include <exception>

namespace TL {

/**************************************************
            TensorIterator declaration 
 **************************************************/

/**
 * @b TensorIterator which iterates over the tensor as the tensor is a flattened 
 * one. Provides all necessary operators to work this as a iterator and throws
 * appropriate exceptions whenever the iterator goes out of bound.
 */
template <typename T>
template <bool Const>
class Tensor<T>::TensorIterator 
{
public:
    /**
     * Default constructor. The iterator would result in an unbounded state.
     */
    TensorIterator() = default;

    /**
     * @brief Bound the iterator to a given tensor.
     * @param tensor The tensor to which the iterator would be bounded.
     * @param _off The offset that would be used to set for initial offset.
     */
    TensorIterator(const Tensor<T>& tensor, size_t _off = 0) 
    : data(tensor.data), desc(tensor.desc), offset(_off) {}

    /**
     * Operators for the iterator. Each of these operations will throw 
     * std::runtime_error if the iterator is not bounded to any tensor.
     * std::out_of_range when the iterator goes out of range of the tensor. 
     */

    TensorIterator& operator++();
    TensorIterator operator++(int);

    TensorIterator& operator--();
    TensorIterator operator--(int);

    TensorIterator& operator+=(size_t);
    TensorIterator& operator-=(size_t);

    TensorIterator operator+(size_t);
    TensorIterator operator-(size_t);

    template <bool Q = Const>
    std::enable_if_t<!Q, T&> 
    operator*();

    template <bool Q = Const>
    std::enable_if_t<Q, const T&>
    operator*() const;

    template <bool Q = Const>
    std::enable_if_t<!Q, T*>
    operator->();

    template <bool Q = Const>
    std::enable_if_t<Q, const T*>
    operator->() const;

    bool operator==(const TensorIterator&) const;
    bool operator!=(const TensorIterator&) const;

private:
    std::weak_ptr<std::vector<T>> data;
    TL::internal::TensorDescriptor desc;
    size_t offset;

    /**
     * @brief Check whether the iterator is bounded to a tensor. If its bounded
     * return a shared pointer to the location.
     * @throw std::runtime_error When the iterator is not bounded to a tensor.
     */
    std::shared_ptr<std::vector<T>> _check() const;
};

/**************************************************
            TensorIterator definition 
 **************************************************/

template <typename T>
template <bool Const>
std::shared_ptr<std::vector<T>> Tensor<T>::TensorIterator<Const>::_check() const {
    auto ptr = data.lock();
    if (!ptr) {
        throw std::runtime_error("Unbounded Iterator");
    }

    if (!(offset >= 0 && offset <= desc.size())) {
        throw std::out_of_range("Iterator out of range");
    }

    return ptr;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const>& Tensor<T>::TensorIterator<Const>::operator++() {
    _check();
    if (offset >= desc.size()) {
        throw std::out_of_range("Increment past end");
    }
    ++offset;
    return *this;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const> Tensor<T>::TensorIterator<Const>::operator++(int) {
    auto temp = *this;
    ++*this;
    return temp;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const>& Tensor<T>::TensorIterator<Const>::operator--() {
    _check();
    if (offset == 0) {
        throw std::out_of_range("Decrement past begin");
    }
    --offset;
    return *this;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const> Tensor<T>::TensorIterator<Const>::operator--(int) {
    auto temp = *this;
    --*this;
    return temp;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const>& Tensor<T>::TensorIterator<Const>::operator+=(size_t _off) {
    _check();
    if (!(offset + _off <= desc.size())) {
        throw std::out_of_range("Increment past end");
    }
    offset += _off;
    return *this;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const>& Tensor<T>::TensorIterator<Const>::operator-=(size_t _off) {
    _check();
    // Check for underflow
    if (_off > offset) {
        throw std::out_of_range("Decrement past begin");
    }
    offset -= _off;
    return *this;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const> Tensor<T>::TensorIterator<Const>::operator+(size_t _off) {
    auto temp = *this;
    temp += _off;
    return temp;
}

template <typename T>
template <bool Const>
Tensor<T>::TensorIterator<Const> Tensor<T>::TensorIterator<Const>::operator-(size_t _off) {
    auto temp = *this;
    temp -= _off;
    return temp;
}

template <typename T>
template <bool Const>
template <bool Q>
std::enable_if_t<!Q, T&>
Tensor<T>::TensorIterator<Const>::operator*() {
    auto ret = _check();
    
    size_t x = 1, idx = 0;
    for (int i = desc.ndim()-1; i >= 0; --i) {
        /* Logic:
            There are total of x elements in the sliced tensor to move from one 
            element to the next along axis i.

            To select the correct axis, taking modulus with the corresponding axis
            shape.

            Had to move desc.stride[i] elements once the correct position in the
            sliced tensor is known.
         */
        idx += ((offset / x) % desc.shape[i]) * desc.stride[i];
        x *= desc.shape[i]; 
    }

    return (*ret)[desc.start + idx];
}

template <typename T>
template <bool Const>
template <bool Q>
std::enable_if_t<Q, const T&>
Tensor<T>::TensorIterator<Const>::operator*() const {
    auto ret = _check();
    
    size_t x = 1, idx = 0;
    for (int i = desc.ndim()-1; i >= 0; --i) {
        idx += ((offset / x) % desc.shape[i]) * desc.stride[i];
        x *= desc.shape[i]; 
    }

    return (*ret)[desc.start + idx];
}

template <typename T>
template <bool Const>
template <bool Q>
std::enable_if_t<!Q, T*>
Tensor<T>::TensorIterator<Const>::operator->() {
    _check();
    return &this->operator*();
}

template <typename T>
template <bool Const>
template <bool Q>
std::enable_if_t<Q, const T*>
Tensor<T>::TensorIterator<Const>::operator->() const {
    _check();
    return &this->operator*();
}

template <typename T>
template <bool Const>
bool Tensor<T>::TensorIterator<Const>::operator==(const TensorIterator& rhs) const {
    auto ret = _check();
    auto ret_rhs = rhs._check();
    return (offset == rhs.offset) && (ret == ret_rhs);
}

template <typename T>
template <bool Const>
bool Tensor<T>::TensorIterator<Const>::operator!=(const TensorIterator& rhs) const {
    return !(*this == rhs);
}

}   // namespace TL

#endif  // TENSORLIB_TENSOR_ITERATOR_H_