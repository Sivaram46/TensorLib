#ifndef TENSORLIB_TENSOR_ITERATOR_HPP
#define TENSORLIB_TENSOR_ITERATOR_HPP

#include <memory>
#include <vector>

namespace TL {

template <typename T, size_t N>
class Tensor;

template <size_t N>
class TensorDescriptor;

/**
 * @b TensorIterator which iterates over the tensor as the tensor is a flattened 
 * one. Provides all necessary operators to work this as a iterator and throws
 * appropriate exceptions whenever the iterator goes out of bound.
 */
template <typename T, size_t N>
class TensorIterator 
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
    TensorIterator(const Tensor<T, N>& tensor, size_t _off = 0) 
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

    T& operator*();
    const T& operator*() const;
    T* operator->();
    const T* operator->() const;

    bool operator==(const TensorIterator&) const;
    bool operator!=(const TensorIterator&) const;

private:
    std::weak_ptr<std::vector<T>> data;
    TensorDescriptor<N> desc;
    size_t offset;

    /**
     * @brief Check whether the iterator is bounded to a tensor. If its bounded
     * return a shared pointer to the location.
     * @throw std::runtime_error When the iterator is not bounded to a tensor.
     */
    std::shared_ptr<std::vector<T>> _check() const;
};

}

#endif