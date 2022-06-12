#ifndef LA_TENSOR_CPP
#define LA_TENSOR_CPP

#include "tensor.hpp"

namespace LA {

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::copy() {
    return Tensor (*data, desc.shape);
}

/* -------- Access operators ----------- */
template <typename T, size_t N>
template <typename... Indices> 
T& Tensor<T, N>::operator()(Indices... indices) {
    return (*data)[desc(indices...)];
}

template <typename T, size_t N>
template <typename... Indices> 
const T& Tensor<T, N>::operator()(Indices... indices) const {
    return (*data)[desc(indices...)];
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

// Assign to a scalar
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
    switch (N) {
        case 0: {
            out << x.data;
            break;
        }

        case 1: {
            if (!x.data) {
                out << "[]";
            }
            else {
                out << "[";
                for (auto& elem : *x.data) {
                    out << elem << ", ";
                }
                out << "\b\b]";
            }

            break;
        }

        case 2: {
            if (!x.data) {
                out << "[]";
                break;
            }
            else {
                for (size_t i = 0; i < x.shape(0); ++i) {
                    for (size_t j = 0; j < x.shape(1); ++j) {
                        auto stride = x.get_stride();
                        size_t idx = i * stride[0] + j * stride[1] + x.desc.get_offset();
                        out << (*x.data)[idx] << " ";
                    }
                    if (i < x.shape(0)-1) {
                        out << "\n";
                    }
                }
            }

            break;
        }

        default: {
            if (!x.data) {
                out << "[]";
            }
            else {
                out << "[";
                for (auto& elem : *x.data) {
                    out << elem << ", ";
                }
                out << "\b\b]";
            }
        }

    }

    return out;
}

}

#endif