#ifndef LA_TENSOR_DESCRIPTOR_HEADER
#define LA_TENSOR_DESCRIPTOR_HEADER

#include <array>

#include "utils.cpp"

namespace LA {

template <typename T, size_t N> class Tensor;

/**
 * A TensorDescriptor class that holds the stride and shape information for a 
 * tensor of dimension @a N.
 */
template <size_t N>
class TensorDescriptor
{
public:
    template <typename T, size_t M> friend class Tensor;

    /** 
     * The default constructor. Sets the size of the Tensor to 0.
     */
    TensorDescriptor() : sz(0) {}

    /** 
     * @brief Constructor that takes the total size and shape along each dimension.
     * @param sz The total size of the tensor.
     * @param shape An array of size N. The shape of the tensor along each dimension.
     */
    TensorDescriptor(size_t, const std::array<size_t, N>&);

    /** 
     * @brief Constructor that takes only shape information.
     * @param Dims... The shape of the Tensor along each dimension.
     */
    template<typename... Dims>
    TensorDescriptor(Dims...);

    /**
     * @brief Function to return the corresponding index in flat vector given the
     * indices in Tensor.
     * @param Dims... Indices along each dimension. Should all be convertible to 
     * size_t
     * @return The index of element in flat vector.
     * @throw std::out_of_range if any of the given indices out of range. 
     */
    template <typename... Dims>
    std::enable_if_t<
        LA::Element_valid<Dims...>(),
    size_t> operator()(Dims...) const;

    void set_offset(size_t off) { start = off; }
    const size_t get_offset() const { return start; }

    /**
     * Total number of elements in the tensor.
     */
    constexpr size_t size() const { return sz; }

private:
    size_t sz;
    size_t start = 0;
    std::array<size_t, N> shape;
    std::array<size_t, N> stride;
    
    /**
     * @brief A utility function that calculates and updates the stride info
     * given the shape info.
     */
    void _calculate_stride();

    /** @brief A utility function that checks whether the given indices are within the
    * shape bounds. 
    * @param Dims... Indices along each dimension.
    * @return A predicate depending on validity of given indices.
    */
    template <typename... Dims>
    bool _check_bound(Dims...) const;
};

/* A TensorDescriptor specialization for 1D tensor. Here the elements are just
represented as flat vectors. */
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

}

#endif