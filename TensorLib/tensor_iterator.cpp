#ifndef TENSORLIB_TENSOR_ITERATOR_CPP
#define TENSORLIB_TENSOR_ITERATOR_CPP

#include <exception>
#include "tensor_iterator.hpp"

namespace TL {

template <typename T, size_t N>
std::shared_ptr<std::vector<T>> TensorIterator<T, N>::_check() const {
    auto ptr = data.lock();
    if (!ptr) {
        throw std::runtime_error("Unbounded Iterator");
    }

    if (!(offset >= 0 && offset <= desc.size())) {
        throw std::out_of_range("Iterator out of range");
    }

    return ptr;
}

template <typename T, size_t N>
TensorIterator<T, N>& TensorIterator<T, N>::operator++() {
    _check();
    if (offset >= desc.size()) {
        throw std::out_of_range("Increment past end");
    }
    ++offset;
    return *this;
}

template <typename T, size_t N>
TensorIterator<T, N> TensorIterator<T, N>::operator++(int) {
    auto temp = *this;
    ++*this;
    return temp;
}

template <typename T, size_t N>
TensorIterator<T, N>& TensorIterator<T, N>::operator--() {
    _check();
    if (offset == 0) {
        throw std::out_of_range("Decrement past begin");
    }
    --offset;
    return *this;
}

template <typename T, size_t N>
TensorIterator<T, N> TensorIterator<T, N>::operator--(int) {
    auto temp = *this;
    --*this;
    return temp;
}

template <typename T, size_t N>
TensorIterator<T, N>& TensorIterator<T, N>::operator+=(size_t _off) {
    _check();
    if (!(offset + _off <= desc.size())) {
        throw std::out_of_range("Increment past end");
    }
    offset += _off;
    return *this;
}

template <typename T, size_t N>
TensorIterator<T, N>& TensorIterator<T, N>::operator-=(size_t _off) {
    _check();
    // Check for underflow
    if (_off > offset) {
        throw std::out_of_range("Decrement past begin");
    }
    offset -= _off;
    return *this;
}

template <typename T, size_t N>
TensorIterator<T, N> TensorIterator<T, N>::operator+(size_t _off) {
    auto temp = *this;
    temp += _off;
    return temp;
}

template <typename T, size_t N>
TensorIterator<T, N> TensorIterator<T, N>::operator-(size_t _off) {
    auto temp = *this;
    temp -= _off;
    return temp;
}

template <typename T, size_t N>
T& TensorIterator<T, N>::operator*() {
    auto ret = _check();
    
    size_t x = 1, idx = 0;
    for (int i = N-1; i >= 0; --i) {
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

template <typename T, size_t N>
const T& TensorIterator<T, N>::operator*() const {
    auto ret = _check();
    
    size_t x = 1, idx = 0;
    for (int i = N-1; i >= 0; --i) {
        idx += ((offset / x) % desc.shape[i]) * desc.stride[i];
        x *= desc.shape[i]; 
    }

    return (*ret)[desc.start + idx];
}

template <typename T, size_t N>
T* TensorIterator<T, N>::operator->() {
    _check();
    return &this->operator*();
}

template <typename T, size_t N>
const T* TensorIterator<T, N>::operator->() const {
    _check();
    return &this->operator*();
}

template <typename T, size_t N>
bool TensorIterator<T, N>::operator==(const TensorIterator& rhs) const {
    auto ret = _check();
    auto ret_rhs = rhs._check();
    return (offset == rhs.offset) && (ret == ret_rhs);
}

template <typename T, size_t N>
bool TensorIterator<T, N>::operator!=(const TensorIterator& rhs) const {
    return !(*this == rhs);
}

}

#endif