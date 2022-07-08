#ifndef TENSORLIB_TENSOR_CPP
#define TENSORLIB_TENSOR_CPP

#include "tensor.hpp"

namespace TL {

template <typename T, size_t N>
Tensor<T, N>::Tensor(Range r, const std::array<size_t, N>& _shape) 
: desc(r.high - r.low, _shape) {
    std::vector<T> tmp(desc.size());
    for (int i = r.low; i < r.high; ++i) {
        tmp[i] = i;
    }

    data = std::make_shared<std::vector<T>>(tmp);
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::copy() {
    return Tensor (*data, desc.shape);
}

/* -------- Access operators ----------- */

template <typename T, size_t N>
template <typename... Dims>
T& Tensor<T, N>::operator()(Dims... dims) {
    return (*data)[desc(dims...)];
}

template <typename T, size_t N>
template <typename... Dims>
const T& Tensor<T, N>::operator()(Dims... dims) const {
    return (*data)[desc(dims...)];
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator()(const Slice& sl) {
    if (N != sl.ranges.size()) {
        throw std::runtime_error("Dimensions Mismatch");
    }

    size_t st = 0, _sz = 1;

    TL::internal::TensorDescriptor<N> des(desc);

    /* Logic:
        If a single size_t n is given for a Slice param, then it would be converted
        to Range(n, n+1)

        Start or offset for the new tensor slice would be the product of range's
        low and its corresponding stride. 

        Shape would be range's high minus low.
    */
    for (size_t i = 0; i < N; ++i) {
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

template <typename T, size_t N>
const Tensor<T, N> Tensor<T, N>::operator()(const Slice& sl) const {
    if (N != sl.ranges.size()) {
        throw std::runtime_error("Dimensions Mismatch");
    }

    size_t st = 0, _sz = 1;

    TL::internal::TensorDescriptor<N> des(desc);
    for (size_t i = 0; i < N; ++i) {
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

template <typename T, size_t N>
template <typename F>
Tensor<T, N>& Tensor<T, N>::_apply(F func) {
    /* Room for optimization */
    for (auto& x : *data) {
        func(x);
    }
    return *this;
}

template <typename T, size_t N>
template <typename F>
Tensor<T, N>& Tensor<T, N>::_apply(const Tensor<T, N>& tensor, F func) {
    auto i = begin();
    auto j = tensor.begin();
    for (; i != end(); ++i, ++j) {
        func(*i, *j);
    }
    return *this;
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator=(const T& val) {
    return _apply([&] (T& elem) {elem = val;});
}

/* ------------ Scalar Operations ----------- */

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator+=(const T& val) {
    return _apply([&] (T& elem) {elem += val;});
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator-=(const T& val) {
    return _apply([&] (T& elem) {elem -= val;});
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator*=(const T& val) {
    return _apply([&] (T& elem) {elem *= val;});
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator/=(const T& val) {
    return _apply([&] (T& elem) {elem /= val;});
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator%=(const T& val) {
    return _apply([&] (T& elem) {elem %= val;});
}

/* ----------- Tensor Operations -------------- */

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator+=(const Tensor<T, N>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 += t2; });
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator-=(const Tensor<T, N>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 -= t2; });
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator*=(const Tensor<T, N>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 *= t2; });
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator/=(const Tensor<T, N>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 /= t2; });
}

template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator%=(const Tensor<T, N>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 %= t2; });
}

/* -------- Binary Operations with a scalar ------------- */

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator+(const T& val) {
    auto lhs = this->copy();
    lhs += val;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator-(const T& val) {
    auto lhs = this->copy();
    lhs -= val;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator*(const T& val) {
    auto lhs = this->copy();
    lhs *= val;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator/(const T& val) {
    auto lhs = this->copy();
    lhs /= val;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator%(const T& val) {
    auto lhs = this->copy();
    lhs %= val;
    return lhs;
}

/* ------- Binary Operations with a tensor --------------- */

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator+(const Tensor<T, N>& tensor) {
    auto lhs = this->copy();
    lhs += tensor;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator-(const Tensor<T, N>& tensor) {
    auto lhs = this->copy();
    lhs -= tensor;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator*(const Tensor<T, N>& tensor) {
    auto lhs = this->copy();
    lhs *= tensor;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator/(const Tensor<T, N>& tensor) {
    auto lhs = this->copy();
    lhs /= tensor;
    return lhs;
}

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator%(const Tensor<T, N>& tensor) {
    auto lhs = this->copy();
    lhs %= tensor;
    return lhs;
}

/* --------- Debug ------------ */

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& out, const Tensor<T, N>& x) {
    if (x.empty()) {
        return out << "[]";
    }

    switch (N) {
        case 0: {
            out << x.data;
            break;
        }

        case 1: {
            out << "[";
            for (auto& elem : x) {
                out << elem << ", ";
            }
            out << "\b\b]";
            break;
        }

        case 2: {
            for (size_t i = 0; i < x.shape(0); ++i) {
                for (size_t j = 0; j < x.shape(1); ++j) {
                    auto stride = x.strides();
                    size_t idx = i * stride[0] + j * stride[1] + x.desc.get_offset();
                    out << (*x.data)[idx] << " ";
                }
                if (i < x.shape(0)-1) {
                    out << "\n";
                }
            }
            break;
        }

        default: {
            out << "[";
            for (auto& elem : x) {
                out << elem << ", ";
            }
            out << "\b\b]";
        }

    }

    return out;
}

}   // namespace TL

#endif