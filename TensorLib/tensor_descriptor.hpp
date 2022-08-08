#ifndef TENSORLIB_TENSOR_DESCRIPTOR_HEADER
#define TENSORLIB_TENSOR_DESCRIPTOR_HEADER

#include <vector>

#include "utils.cpp"

namespace TL {

template <typename T> class TensorIterator;
template <typename T> class Tensor;

namespace internal {

/**
 * @brief Holds information about the tensor like shape, strides.
 */
class TensorDescriptor
{
public:
    template <typename T> friend class TL::Tensor;
    template <typename T> friend class TL::TensorIterator;

    /** 
     * @brief The default constructor. Sets the size and dimension of the Tensor to 0.
     */
    TensorDescriptor() : sz(1), n_dim(0) {}

    /** 
     * @brief Constructs descriptor from shape along each dim and optional start offset.
     * @param _shape The shape of the tensor along each dimension.
     * @param _st (Optional) The offset for subtensors.
     */
    TensorDescriptor(const std::vector<size_t>&, size_t = 0);

    /** 
     * @brief Constructs from shapes only.
     * @param dims... The shape of the Tensor along each dimension.
     */
    template<typename... Dims>
    TensorDescriptor(Dims...);

    TensorDescriptor(const TensorDescriptor&) = default;

    /**
     * @brief Returns the corresponding index in flat vector given the
     * indices in Tensor.
     * @param dims... Indices along each dimension. Should all be convertible to 
     * size_t
     * @return The index of element in flat vector.
     * @throw std::out_of_range if any of the given indices goes out of range. 
     * @throw std::runtime_error when @a dims... and @a n_dim mismatch.
     */
    template <typename... Dims>
    std::enable_if_t<
        TL::Element_valid<Dims...>(),
    size_t> operator()(Dims...) const;

    /**
     * @brief Returs the number of elements in the tensor.
     */
    size_t size() const { return sz; }

    /**
     * @brief Returns the dimension of the tensor.
     */
    size_t ndim() const { return n_dim; }

private:
    size_t sz;
    size_t n_dim;
    size_t start = 0;   /* offset for subtensors */
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    
    /**
     * @brief A utility function that calculates and updates the stride information
     * given the shape information.
     */
    void _calculate_stride();

    /** @brief A utility function that checks whether the given indices are within the
    * shape bounds. 
    * @param dims... Indices along each dimension.
    * @return @a true if indices are valid, else @a false.
    */
    template <typename... Dims>
    bool _check_bound(Dims...) const;
};

}   // namespace interanl

}   // namespace TL

#endif