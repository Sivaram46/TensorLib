#ifndef LA_TENSOR_CPP
#define LA_TENSOR_CPP

#include "tensor.hpp"

namespace LA {

template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::copy() {
    if (data) {
        // Tensor tensor(data, desc);
        Tensor tensor(*data, desc.shape);
        return tensor;
    }
    else {
        Tensor<T, N> tensor;
        return tensor;
    }
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
    auto j = tensor.cbegin();
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

/* ----------- Tensor Operations -------------- */
template <typename T, size_t N>
Tensor<T, N>& Tensor<T, N>::operator+=(const Tensor<T, N>& tensor) {
    return _apply(tensor, [] (T& t1, const T& t2) { t1 += t2; });
}

/* -------- Binary Operators ------------- */
template <typename T, size_t N>
Tensor<T, N> Tensor<T, N>::operator+(const Tensor<T, N>& tensor) {
    auto lhs = this->copy();
    lhs += tensor;
    return lhs;
}

/* -------- Access operators ----------- */
template <typename T, size_t N>
template <typename... Indices> 
T& Tensor<T, N>::operator()(Indices... indices) {
    return (*data)[desc(indices...)];
}

/* --------- Debug ------------ */
template <typename T, size_t N>
std::ostream& operator<<(std::ostream& out, const Tensor<T, N>& tensor) {
    switch (N) {
        case 0: {
            out << tensor.data;
            break;
        }

        case 1: {
            out << "[";
            for (auto& elem : *tensor.data) {
                out << elem << ", ";
            }
            out << "\b\b]";
            break;
        }

        case 2: {
            // for (size_t i = 0; i < tensor.shape(0); ++i) {
            //     for (size_t j = 0; j < tensor.shape(1); ++j) {
            //         out << tensor(i, j) << " ";
            //     }
            //     out << "\n";
            // }
            break;
        }

        default: {
            out << "[";
            for (auto& elem : *tensor.data) {
                out << elem << ", ";
            }
            out << "\b\b]";
        }

    }

    return out;
}

// template <typename T>
// std::ostream& operator<< <2> (std::ostream& out, const Tensor<T, 2>& tensor) {
//     return out;
// }

}

#endif