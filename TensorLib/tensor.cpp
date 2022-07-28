#ifndef TENSORLIB_TENSOR_CPP
#define TENSORLIB_TENSOR_CPP

#include "tensor.hpp"

namespace TL {

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
Tensor<T> Tensor<T>::copy() {
    return Tensor (*data, desc.shape);
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
    return Tensor(data, des);
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
    return Tensor(data, des);
}

template <typename T>
Tensor<T> Tensor<T>::operator[](size_t idx) {
    if (idx >= desc.shape[0]) {
        throw std::out_of_range("Index out of range");
    }

    std::vector<size_t> sh (ndim() - 1);
    std::copy(desc.shape.begin() + 1, desc.shape.end(), sh.begin());
    TL::internal::TensorDescriptor tdesc(sh, desc.stride[0] * idx);
    return Tensor(data, tdesc);
}

template <typename T>
const Tensor<T> Tensor<T>::operator[](size_t idx) const {
    if (idx >= desc.shape[0]) {
        throw std::out_of_range("Index out of range");
    }

    std::vector<size_t> sh (ndim() - 1);
    std::copy(desc.shape.begin() + 1, desc.shape.end(), sh.begin());
    TL::internal::TensorDescriptor tdesc(sh, desc.stride[0] * idx);
    return Tensor(data, tdesc);
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

/* --------- Printing / Formatting tensor ------------ */
template <typename T>
void Tensor<T>::print(std::ostream& out) const {
    TL::internal::TensorPrint<T> pr (out, *this);
    pr.print();
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& tensor) {
    if (tensor.empty()) {
        out << std::string(tensor.ndim(), '[')
            << std::string(tensor.ndim(), ']')
            << '\n';
        return out;
    }

    tensor.print(out);
    return out;
}

}   // namespace TL

#endif