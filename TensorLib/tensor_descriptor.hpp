#ifndef TENSORLIB_TENSOR_DESCRIPTOR_HEADER
#define TENSORLIB_TENSOR_DESCRIPTOR_HEADER

#include <array>

#include "utils.cpp"

namespace TL {

template <typename T, size_t N> class Tensor;
template <typename T, size_t N> class TensorIterator;

namespace internal {

/**
 * A TensorDescriptor class that holds the stride and shape information for a 
 * tensor of dimension @a N.
 */
template <size_t N>
class TensorDescriptor
{
public:
    template <typename T, size_t M> friend class TL::Tensor;
    template <typename T, size_t M> friend class TL::TensorIterator;

    /** 
     * The default constructor. Sets the size of the Tensor to 0.
     */
    TensorDescriptor() : sz(0) {}

    /** 
     * @brief Constructor that takes the total size and shape along each dimension.
     * @param sz The total size of the tensor.
     * @param shape An array of size N. The shape of the tensor along each dimension.
     */
    TensorDescriptor(const std::array<size_t, N>&, size_t = 0);

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
        TL::Element_valid<Dims...>(),
    size_t> operator()(Dims...) const;

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

}   // namespace interanl

}   // namespace TL

#endif