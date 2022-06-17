#ifndef LA_TENSOR_DESCRIPTOR_HEADER
#define LA_TENSOR_DESCRIPTOR_HEADER

#include <array>

#include "utils.cpp"

namespace LA {

template <typename T, size_t N> class Tensor;

template <size_t N>
class TensorDescriptor
{
public:
    template <typename T, size_t M> friend class Tensor;

    TensorDescriptor() : sz(0) {}

    TensorDescriptor(size_t, const std::array<size_t, N>&);

    // Constructor that takes all the shape
    template<typename... Dims>
    TensorDescriptor(Dims...);

    // Function that takes the individual indices along each dimension and 
    // returns the corresponding index in the flat vector.
    template <typename... Dims>
    std::enable_if_t<
        LA::Element_valid<Dims...>(),
    size_t> operator()(Dims...) const;

    void set_offset(size_t off) { start = off; }
    const size_t get_offset() const { return start; }

    constexpr size_t size() const { return sz; }

private:
    size_t sz;
    size_t start = 0;
    std::array<size_t, N> shape;
    std::array<size_t, N> stride;
    
    void _calculate_stride();
    // A utility function that checks whether the given indices are within the
    // shape bounds.
    template <typename... Dims>
    bool _check_bound(Dims...) const;
};

// A TensorDescriptor specialization for 1D tensor. Here the elements are just
// represented as flat vectors.
template<>
class TensorDescriptor<1>
{
public:
    template <typename T, size_t M> friend class Tensor;

    TensorDescriptor() : shape(0) {}
    TensorDescriptor(size_t _shape) : shape(_shape) {}
    size_t operator()(size_t idx) const {
        if (idx >= shape) {
            throw std::out_of_range("Index out of range");
        }
        return start + idx; 
    }
    void set_offset(size_t off) { start = off; }
    const size_t get_offset() const { return start; }

    constexpr size_t size() const { return shape; }
    
private:
    size_t shape;
    size_t start = 0;
};

// A TensorDescriptor specialization for 2D tensor that is for a Matrix.
template <>
class TensorDescriptor<2>
{
public:
    template <typename T, size_t M> friend class Tensor;

    TensorDescriptor() : sz(0) {}

    TensorDescriptor(size_t _sz, const std::array<size_t, 2>& _shape)
    : sz(_sz), shape(_shape), stride({_shape[1], 1}) {}

    TensorDescriptor(size_t row, size_t col)
    : sz(row * col), shape({row, col}), stride({col, 1}) {}

    size_t operator()(size_t _row, size_t _col) const {
        if (_row >= shape[0] || _col >= shape[1]) {
            throw std::out_of_range("Index out of range");
        }
        return start + (_row * stride[0] + _col * stride[1]);
    }

    void set_offset(size_t off) { start = off; }
    const size_t get_offset() const { return start; }

    constexpr size_t size() const { return sz; }

private:
    size_t sz;
    std::array<size_t, 2> stride;
    std::array<size_t, 2> shape;
    size_t start = 0;
};

}

#endif