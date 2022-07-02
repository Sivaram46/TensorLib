#ifndef LA_TENSOR_ITERATOR_HPP
#define LA_TENSOR_ITERATOR_HPP

#include <memory>
#include <vector>

namespace TL {

template <typename T, size_t N>
class Tensor;

template <size_t N>
class TensorDescriptor;

template <typename T, size_t N>
class TensorIterator 
{
public:
    TensorIterator() = default;

    TensorIterator(const Tensor<T, N>& tensor, size_t _off = 0) 
    : data(tensor.data), desc(tensor.desc), offset(_off) {}

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

    std::shared_ptr<std::vector<T>> _check() const;
};

}

#endif