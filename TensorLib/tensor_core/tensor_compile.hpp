#ifndef TENSORLIB_TENSOR_COMPILE_H_
#define TENSORLIB_TENSOR_COMPILE_H_

#include "tensor_desc_compile.hpp"
#include "utils.hpp"

#include <array>
#include <initializer_list>
#include <cstddef>

namespace TL {

template <typename T, size_t... dims>
class Tensor
{
public:
    constexpr static size_t sz = TL::product(dims...);
    constexpr static size_t n_dim = sizeof...(dims);

    constexpr size_t ndim() const {
        return n_dim;
    }

    constexpr size_t size() const {
        return sz;
    }

    /* Constrcutors */
    Tensor() : desc(0, dims...) {}

    Tensor(const std::array<T, sz>& _data) : data(_data), desc(0, dims...) {}

    Tensor(const Tensor&) = default;

    Tensor& operator=(const T&);

    /* Accessing elements */ 
    template <typename... Dims>
    T& operator()(Dims... _dims);

    template <typename... Dims>
    const T& operator()(Dims... _dims) const;

private:
    std::array<T, sz> data;
    internal::TensorDescriptor<T, n_dim> desc;
};

template <typename T, size_t... dims>
Tensor<T, dims...>& Tensor<T, dims...>::operator=(const T& elem) {
    data.fill(elem);
}

template <typename T, size_t... dims>
template <typename... Dims>
T& Tensor<T, dims...>::operator()(Dims... _dims) {
    return data[desc(_dims...)];
}

template <typename T, size_t... dims>
template <typename... Dims>
const T& Tensor<T, dims...>::operator()(Dims... _dims) const {
    return data[desc(_dims...)];
}

}   // namespace TL

#endif  // TENSORLIB_TENSOR_COMPILE_H_