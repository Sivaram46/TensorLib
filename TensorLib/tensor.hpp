#ifndef TENSORLIB_TENSOR_HEADER
#define TENSORLIB_TENSOR_HEADER

#include <memory>
#include <type_traits>
#include <vector>
#include <initializer_list>
#include <iostream>

#include "tensor_descriptor.hpp"
#include "tensor_print.cpp"
#include "tensor_formatter.cpp"
#include "utils.cpp"
#include "slice.cpp"

namespace TL {

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
    size_t ndim() const { return desc.ndim(); }

    /**
     * @brief Returns the number of elements in the Tensor.
     */
    size_t size() const { return desc.size(); }

    /**
     * @brief Returns the shape of the Tensor.
     * @return std::vector<size_t> of size Tensor::ndim()
     */
    const std::vector<size_t>& shape() const { return desc.shape; }

    /** 
     * @brief Returns whether the Tensor is empty or not.
     * @return @a True when the tensor is empty else @a False
     */
    bool empty() const { return !size(); }

    /**
     * @brief Returns the strides of the Tensor. 
     * 
     * Stride of an axis is the number of elements to be moved in the tensor
     * to move from one axis to other.
     * 
     * @return std::vector<size_t> of size Tensor::ndim()
     */
    const std::vector<size_t>& strides() const { return desc.stride; }

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
     * @brief The default constructor that initializes an empty tensor.
     */
    Tensor() : data(new std::vector<T>()) {}

    /**
     * @brief Constructs tensor from shared_ptr<vector<T>> and TensorDescriptor
     * @param _data The shared pointer of vector<T> that will be initialized to 
     * the tensor.
     * @param _desc @a TL::interal::TensorDescriptor that holds the information 
     * about the tensor like shape and strides.
     */
    Tensor(std::shared_ptr<std::vector<T>> _data, const TL::internal::TensorDescriptor& _desc)
    : data(_data), desc(_desc) {}

    /**
     * @brief Constructs tensor from shapes. The elements are default initialized.
     * @param dims... should all be convertible to size_t.
     */
    template <typename... Dims,
        typename = std::enable_if_t<All(Is_convertible<Dims, size_t>()...)>
    >
    Tensor(Dims... dims)
    : desc(dims...) {
        data = std::make_shared<std::vector<T>>(desc.size());
    }

    /** 
     * @brief Constructs tensor from vector<T> and shapes. 
     * Since vector and array has constructors that takes an initializer list, 
     * this constructor will be invoked upon calling with initializer lists. 
     * @param _vec The vector from which the data will be copied.
     * @param _shape Shape of the tensor to build.  
     */
    Tensor(const std::vector<T>& _vec, const std::vector<size_t>& _shape)
    : data(new std::vector<T>(_vec)), desc(_shape) {}

    /**
     * @brief Move version of the constructor that takes vector and shape. 
     */
    Tensor(std::vector<T>&& vec, const std::vector<size_t>& _shape)
    : data(new std::vector<T>(vec)), desc(_shape) {}

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
    Tensor copy();

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

}   // namespace TL

#endif