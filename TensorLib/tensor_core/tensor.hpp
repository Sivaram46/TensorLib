#ifndef TENSORLIB_TENSOR_H_
#define TENSORLIB_TENSOR_H_

#include "tensor_descriptor.hpp"
#include "tensor_formatter.hpp"
#include "slice.hpp"
#include "range.hpp"
#include "utils.hpp"

#include <memory>
#include <type_traits>
#include <vector>
#include <initializer_list>
#include <iostream>

namespace TL {

/**************************************************
                Tensor declaration 
 **************************************************/

/**
 * @brief A tensor of arbitrary type.
 * @tparam T Type of the elements in the tensor.
 */
template <typename T>
class Tensor 
{
public:
    /**
     * @brief Forward iterator over the tensor.
     */
    template <bool Const> 
    class TensorIterator;

    using iterator = TensorIterator<false>;
    using const_iterator = TensorIterator<true>;
    using value_type = T;

    /**
     * @brief Returns the dimension of the Tensor.
     */
    size_t ndim() const {
        return desc.ndim();
    }

    /**
     * @brief Returns the number of elements in the Tensor.
     */
    size_t size() const {
        return desc.size();
    }

    /**
     * @brief Returns the shape of the Tensor.
     * @return std::vector<size_t> of size Tensor::ndim()
     */
    std::vector<size_t> shape() const {
        return desc.shape;
    }

    /** 
     * @brief Returns whether the Tensor is empty or not.
     * @return @a True when the tensor is empty else @a False
     */
    bool empty() const {
        return !size();
    }

    /**
     * @brief Returns the strides of the Tensor. 
     * 
     * Stride of an axis is the number of elements to be moved in the tensor
     * to move from one axis to other.
     * 
     * @return std::vector<size_t> of size Tensor::ndim()
     */
    std::vector<size_t> strides() const {
        return desc.stride;
    }

    /* ---------- Iterators over the Tensor ---------- */

    /**
     * @brief Returns a read/write iterator that points to the first element in 
     * the tensor. 
     */
    iterator begin();

    /**
     * @brief Returns a read/write iterator that points one past the last 
     * element in the tensor.
     */
    iterator end();

    /**
     * @brief Returns a read only iterator that points to the first element in 
     * the tensor. 
     */
    const_iterator begin() const;

    /**
     * @brief Returns a read only iterator that points one past the last 
     * element in the tensor.
     */
    const_iterator end() const;
    
    /**
     * @brief Returns a read only iterator that points to the first element in 
     * the tensor. 
     */
    const_iterator cbegin() const;

    /**
     * @brief Returns a read only iterator that points one past the last 
     * element in the tensor.
     */
    const_iterator cend() const;

    /* ---------- Constructors ---------- */

    /**
     * @brief The vector shouldn't be empty.
     */
    Tensor() = delete;

    /**
     * Constructs a 0 dimensional tensor from an element of tensor type.
     * @param _val The value which will be put in a 0D tensor.
     */
    Tensor(const T& _val) : data(new std::vector<T>(1, _val)) {}

    /**
     * @brief Constructs tensor from shared_ptr<vector<T>> and TensorDescriptor
     * @param _data The shared pointer of vector<T> that will be initialized to 
     * the tensor.
     * @param _desc @a TL::interal::TensorDescriptor that holds the information 
     * about the tensor like shape and strides.
     */
    Tensor(
        std::shared_ptr<std::vector<T>> _data,
        const TL::internal::TensorDescriptor& _desc,
        const TL::TensorFormatter& _format
    )
    : data(_data), desc(_desc), format(_format) {}

    /** 
     * @brief Constructs tensor from vector<T> and shapes. 
     * Since vector and array has constructors that takes an initializer list, 
     * this constructor will be invoked upon calling with initializer lists. 
     * @param _vec The vector from which the data will be copied.
     * @param _shape Shape of the tensor to build.  
     */
    Tensor(const std::vector<T>& _vec, const std::vector<size_t>& _shape, size_t _st = 0)
    : data(new std::vector<T>(_vec)), desc(_shape, _st) {
        if (size() != _vec.size()) {
            throw std::runtime_error("Number of elements and shapes mismatch");
        }
    }

    /**
     * @brief Move version of the consTensortructor that takes vector and shape. 
     */
    Tensor(std::vector<T>&& _vec, const std::vector<size_t>& _shape, size_t _st = 0)
    : data(new std::vector<T>(_vec)), desc(_shape, _st) {
        if (size() != _vec.size()) {
            throw std::runtime_error("Number of elements and shapes mismatch");
        }
    }

    /**
     * @brief Constructs a tensor with evenly spaced elements withing the given
     * interval and shape of the tensor.
     * @param _range TL::Range object that takes @a low and @a high. Tensor will 
     * have elements in range [low, high).
     * @param _shape Shape of the tensor to build.
     */
    Tensor(TL::Range, const std::vector<size_t>&);

    /**
     * @brief Conctructs a new tensor that just refers to the given tensor. No
     * copy is actually made.
     * @param _tensor Tensor the tensor to copy.
     */
    Tensor(const Tensor&) = default;

    /**
     * @brief Assignment operator that increments the reference counter for rhs's and 
     * decrement for this's (Handled by shared_ptr). 
     */
    Tensor& operator=(const Tensor&) = default;

    /**
     * @brief Returns a copy of the tensor.
     */
    Tensor copy() const;

    ~Tensor() = default;

    /* ---------- Access Operators ---------- */

    /**
     * @brief Access tensor elements from indices. 
     * @param dims... should all be convertible to size_t.
     * @throws std::out_of_range when the given indices are out of range of tensor shape.
     * @return T& Returns a lvalue reference of the element type. 
     */
    template <typename... Dims>
    T& operator()(Dims...);

    /**
     * @brief Constant version of accessing using indices.
     */
    template <typename... Dims>
    const T& operator()(Dims...) const;

    /**
     * @brief Access tensor from slices.
     * @param sl TL::Slice object that takes two params, 1) any type convertible 
     * to size_t 2) @a TL::Range. @a Range(low,high) access elements as [low, high).
     * Any one of the param should be @a Range.
     * @throws @a std::runtime_error when Slice and tensor dimensions mismatch.
     * @throws @a std::out_of_range when Indices passed to slices goes out of range.
     */
    Tensor operator()(const Slice&);

    /**
     * @brief Constant version of accessing from slices.
     */
    const Tensor operator()(const Slice&) const;

    /** 
     * @brief Access the tensor in 0th dimension.
     * @param idx The index to be accessed.
     * @throw std::out_of_range when index goes out of range.
     * @return Tensor of ndim() - 1 dimension.
     */
    Tensor operator[](size_t);

    /** 
     * @brief Constant version of subscript indexing. 
     */
    const Tensor operator[](size_t) const;

    /**
     * @brief An utility function that applies @a F to every element in the tensor.
     * @param F A functor type.
     */
    template <typename F>
    Tensor& _apply(F);

    /**
     * @brief An utility function that applies a binary function @a F to every 
     * element in @a this and other tensor.
     * @param tensor The other tensor.
     * @param F A functor type.
     * @throw std::runtime_error when this and other tenor dimensions mismatch.
     */
    template <typename F>
    Tensor& _apply(const Tensor&, F);

    /* ---------- Arithmetic operators ---------- */

    /**
     * @brief Assign to a scalar 
     */
    Tensor& operator=(const T&);

    /* ---------- Operate with scalars --------- */

    Tensor& operator+=(const T&);
    Tensor& operator-=(const T&);
    Tensor& operator*=(const T&);
    Tensor& operator/=(const T&);
    Tensor& operator%=(const T&);

    /* ---------- Operate with Tensors ---------- */

    Tensor& operator+=(const Tensor&);
    Tensor& operator-=(const Tensor&);
    Tensor& operator*=(const Tensor&);
    Tensor& operator/=(const Tensor&);
    Tensor& operator%=(const Tensor&);

    /* ------- Binary operations with a scalar ---------- */

    Tensor operator+(const T&);
    Tensor operator-(const T&);
    Tensor operator*(const T&);
    Tensor operator/(const T&);
    Tensor operator%(const T&);

    /* ------- Binary operations with a tensor ---------- */

    Tensor operator+(const Tensor&);
    Tensor operator-(const Tensor&);
    Tensor operator*(const Tensor&);
    Tensor operator/(const Tensor&);
    Tensor operator%(const Tensor&);

    /* ------- Manipulating dimensions ------- */

    template <typename... Dims>
    std::enable_if_t<
        All(Is_convertible<Dims, size_t>()...),
    Tensor> reshape(Dims... dims) const;

    Tensor squeeze(long = -1) const;

    Tensor expand_dims(size_t) const;

    Tensor ravel() const;

    /* --------- Printing / Formatting tensor ------------ */

    /**
     * @brief Prints the tensor in the given output stream.
     * @param out std::ostream& object to which the tensor to be printed.
     */
    void print(std::ostream&) const;
    
    /** 
     * @brief Overloading of putting to stream operator for the tensor.
     */
    template <typename U>
    friend std::ostream& operator<<(std::ostream&, const Tensor<U>&);
    
    /* Holds the formatting options for a tensor. */
    TensorFormatter format;
    
private:
    /* Holds the information about the tensor like shape, strides. */
    TL::internal::TensorDescriptor desc;
    /* Shared pointer to the actual data */
    std::shared_ptr<std::vector<T>> data;
};

/**************************************************
                Tensor definition 
 **************************************************/

template <typename T>
Tensor<T>::Tensor(Range _range, const std::vector<size_t>& _shape) 
: desc(_shape) {
    std::vector<T> tmp(desc.size());
    for (size_t i = _range.low; i < _range.high; ++i) {
        tmp[i] = i;
    }

    data = std::make_shared<std::vector<T>>(tmp);
}

template <typename T>
Tensor<T> Tensor<T>::copy() const {
    Tensor temp(*data, desc.shape, desc.start);
    temp.format = format;
    return temp;
}

/* -------- Access operators ----------- */

template <typename T>
template <typename... Dims>
T& Tensor<T>::operator()(Dims... dims) {
    return (*data)[desc(dims...)];
}

template <typename T>
template <typename... Dims>
const T& Tensor<T>::operator()(Dims... dims) const {
    return (*data)[desc(dims...)];
}

template <typename T>
Tensor<T> Tensor<T>::operator()(const Slice& sl) {
    if (ndim() != sl.ranges.size()) {
        throw std::runtime_error("Dimensions Mismatch");
    }

    size_t st = 0, _sz = 1;

    TL::internal::TensorDescriptor des(desc);

    /* Logic:
        If a single size_t n is given for a Slice param, then it would be converted
        to Range(n, n+1)

        Start or offset for the new tensor slice would be the product of range's
        low and its corresponding stride. 

        Shape would be range's high minus low.
    */
    for (size_t i = 0; i < ndim(); ++i) {
        size_t low = sl.ranges[i].low, high = sl.ranges[i].high;
        if (low >= high) {
            throw std::runtime_error("`low` range should be lesser than the `high` range");
        }
        if (high > desc.shape[i]) {
            throw std::out_of_range("Index out of range");
        }

        st += low * desc.stride[i];
        des.shape[i] = high - low;
        _sz *= (high - low);

        if (sl.ranges[i].single()) {
            des.stride[i] = 0;
        }
    }

    des.start += st;
    des.sz = _sz;
    return Tensor(data, des, format);
}

template <typename T>
const Tensor<T> Tensor<T>::operator()(const Slice& sl) const {
    if (ndim() != sl.ranges.size()) {
        throw std::runtime_error("Dimensions Mismatch");
    }

    size_t st = 0, _sz = 1;

    TL::internal::TensorDescriptor des(desc);
    for (size_t i = 0; i < ndim(); ++i) {
        size_t low = sl.ranges[i].low, high = sl.ranges[i].high;
        if (low >= high) {
            throw std::runtime_error("`low` range should be lesser than the `high` range");
        }
        if (high > desc.shape[i]) {
            throw std::out_of_range("Index out of range");
        }

        st += low * desc.stride[i];
        des.shape[i] = high - low;
        _sz *= (high - low);

        if (sl.ranges[i].single()) {
            des.stride[i] = 0;
        }
    }

    des.start += st;
    des.sz = _sz;
    return Tensor(data, des, format);
}

template <typename T>
Tensor<T> Tensor<T>::operator[](size_t idx) {
    if (idx >= desc.shape[0]) {
        throw std::out_of_range("Index out of range");
    }

    if (ndim() == 1) {
        return Tensor((*data)[0]);
    }

    std::vector<size_t> sh (ndim() - 1);
    std::copy(desc.shape.begin() + 1, desc.shape.end(), sh.begin());
    TL::internal::TensorDescriptor tdesc(sh, desc.stride[0] * idx);
    return Tensor(data, tdesc, format);
}

template <typename T>
const Tensor<T> Tensor<T>::operator[](size_t idx) const {
    if (idx >= desc.shape[0]) {
        throw std::out_of_range("Index out of range");
    }

    if (ndim() == 1) {
        return Tensor((*data)[0]);
    }

    std::vector<size_t> sh (ndim() - 1);
    std::copy(desc.shape.begin() + 1, desc.shape.end(), sh.begin());
    TL::internal::TensorDescriptor tdesc(sh, desc.stride[0] * idx);
    return Tensor(data, tdesc, format);
}

} // namespace TL

#include "tensor_iterator.hpp"

namespace TL {

template <typename T>
auto Tensor<T>::begin() -> iterator {
    return iterator(*this);
}

template <typename T>
auto Tensor<T>::end() -> iterator {
    return iterator(*this, size());
}

template <typename T>
auto Tensor<T>::begin() const -> const_iterator {
    return const_iterator(*this);
}

template <typename T>
auto Tensor<T>::end() const -> const_iterator {
    return const_iterator(*this, size());
}

template <typename T>
auto Tensor<T>::cbegin() const -> const_iterator {
    return const_iterator(*this);
}

template <typename T>
auto Tensor<T>::cend() const -> const_iterator {
    return const_iterator(*this, size());
}

template <typename T>
template <typename F>
Tensor<T>& Tensor<T>::_apply(F func) {
    /* Room for optimization */
    for (auto& x : *data) {
        func(x);
    }
    return *this;
}

template <typename T>
template <typename F>
Tensor<T>& Tensor<T>::_apply(const Tensor<T>& tensor, F func) {
    if (shape() != tensor.shape()) {
        throw std::runtime_error("Dimensions Mismatch");
    }

    auto i = begin();
    auto j = tensor.begin();
    for (; i != end(); ++i, ++j) {
        func(*i, *j);
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const T& val) {
    return _apply([&] (T& elem) {elem = val;});
}

/* ------------ Scalar Operations ----------- */

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const T& val) {
    return _apply([&] (T& elem) {elem += val;});
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const T& val) {
    return _apply([&] (T& elem) {elem -= val;});
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const T& val) {
    return _apply([&] (T& elem) {elem *= val;});
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const T& val) {
    return _apply([&] (T& elem) {elem /= val;});
}

template <typename T>
Tensor<T>& Tensor<T>::operator%=(const T& val) {
    return _apply([&] (T& elem) {elem %= val;});
}

/* ----------- Tensor Operations -------------- */

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 += t2; });
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 -= t2; });
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 *= t2; });
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 /= t2; });
}

template <typename T>
Tensor<T>& Tensor<T>::operator%=(const Tensor<T>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 %= t2; });
}

/* -------- Binary Operations with a scalar ------------- */

template <typename T>
Tensor<T> Tensor<T>::operator+(const T& val) {
    auto lhs = this->copy();
    lhs += val;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const T& val) {
    auto lhs = this->copy();
    lhs -= val;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const T& val) {
    auto lhs = this->copy();
    lhs *= val;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const T& val) {
    auto lhs = this->copy();
    lhs /= val;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator%(const T& val) {
    auto lhs = this->copy();
    lhs %= val;
    return lhs;
}

/* ------- Binary Operations with a tensor --------------- */

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& tensor) {
    auto lhs = this->copy();
    lhs += tensor;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& tensor) {
    auto lhs = this->copy();
    lhs -= tensor;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& tensor) {
    auto lhs = this->copy();
    lhs *= tensor;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& tensor) {
    auto lhs = this->copy();
    lhs /= tensor;
    return lhs;
}

template <typename T>
Tensor<T> Tensor<T>::operator%(const Tensor<T>& tensor) {
    auto lhs = this->copy();
    lhs %= tensor;
    return lhs;
}

/* ------- Manipulating dimensions ------- */
template <typename T>
template <typename... Dims>
std::enable_if_t<
    All(Is_convertible<Dims, size_t>()...),
Tensor<T>> Tensor<T>::reshape(Dims... dims) const {
    std::vector<size_t> _shape { size_t(dims)... };
    return Tensor(*data, _shape, desc.start);
}

template <typename T>
Tensor<T> Tensor<T>::squeeze(long axis) const {
    Tensor temp = copy();
    // Squeeze out all possible dimensions
    if (axis == -1) {
        std::vector<size_t> _shape;
        for (auto& elem : desc.shape) {
            if (elem != 1) {
                _shape.push_back(elem);
            }
        }

        return Tensor(*data, _shape, desc.start);
    }

    else {
        if (desc.shape[axis] != 1) {
            throw std::runtime_error("Cannot squeeze out non zero shaped axis");
        }

        if (axis >= ndim() || axis < -1) {
            throw std::out_of_range("Axis out of bound for squeeze");
        }
        
        auto _shape = shape();
        _shape.erase(_shape.begin() + axis);
        return Tensor(*data, _shape, desc.start);
    }
}

template <typename T>
Tensor<T> Tensor<T>::expand_dims(size_t axis) const {
    if (axis > ndim()) {
        throw std::out_of_range("Axis out of bound for expand_dims");
    }

    auto _shape = shape();
    _shape.insert(_shape.begin() + axis, 1);
    return Tensor(*data, _shape, desc.start);
}

template <typename T>
Tensor<T> Tensor<T>::ravel() const {
    std::vector<size_t> _shape = {size()};
    return Tensor(*data, _shape, desc.start);
}

}   // namespace TL

/* --------- Printing / Formatting tensor ------------ */

#include "tensor_print.hpp"

namespace TL {

template <typename T>
void Tensor<T>::print(std::ostream& out) const {
    TL::internal::TensorPrint<T> pr (out, *this);
    pr.print();
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& tensor) {
    if (!tensor.ndim()) {
        return out << (*tensor.data)[0] << "\n";
    }

    tensor.print(out);
    return out;
}

}   // namespace TL

#endif  // TENSORLIB_TENSOR_H_