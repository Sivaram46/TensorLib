#ifndef TENSORLIB_TENSOR_HEADER
#define TENSORLIB_TENSOR_HEADER

#include <memory>
#include <type_traits>
#include <vector>
#include <initializer_list>
#include <iostream>

#include "tensor_descriptor.hpp"
#include "tensor_iterator.hpp"
#include "tensor_print.cpp"
#include "tensor_formatter.cpp"
#include "utils.cpp"
#include "slice.cpp"

namespace TL {

/**
 * A compile time tensor of type @a T and dimension @a N
 */
template <typename T, size_t N>
class Tensor 
{
public:
    using iterator = TensorIterator<T, N>;
    using const_iterator = const TensorIterator<T, N>;
    using value_type = T;

    static const size_t n_dim = N;

    constexpr size_t ndim() const { return n_dim; }
    constexpr size_t size() const { return desc.size(); }

    size_t shape(size_t dim) const { return desc.shape[dim]; }
    const std::array<size_t, N>& shape() const { return desc.shape; }
    constexpr bool empty() const { return !size(); }

    const std::array<size_t, N>& strides() const { return desc.stride; }

    /* ---------- Iterators over the Tensor ---------- */

    /**
     * Returns a forward iterator over the data. Iterator of the underlying std::vector.
     * It is not advisable to call this on sub-matrices.
     */
    iterator begin() { 
        return TensorIterator<T, N>(*this); 
    }
    iterator end() { 
        return TensorIterator<T, N>(*this, desc.size()); 
    }

    const_iterator begin() const { 
        return TensorIterator<T, N>(*this);
    }
    const_iterator end() const { 
        return TensorIterator<T, N>(*this, desc.size()); 
    }
    
    const_iterator cbegin() const { 
        return TensorIterator<T, N>(*this); 
    }
    const_iterator cend() const {
        return TensorIterator<T, N>(*this, desc.size()); 
    }

    /* ---------- Constructors ---------- */

    /* The default constructor */
    Tensor() : data(new std::vector<T>()) {}

    Tensor(std::shared_ptr<std::vector<T>> _data, const TL::internal::TensorDescriptor<N>& _desc)
    : data(_data), desc(_desc) {}

    /**
     * @brief Constructor that build Tensor from shapes. This constructs an empty tensor.
     * @param Dims... should all be convertible to size_t, otherwise the constructor
     * won't be enabled at compile time.
     */
    template <typename... Dims,
        typename = std::enable_if_t<All(Is_convertible<Dims, size_t>()...)>
    >
    Tensor(Dims... dims)
    : desc(dims...) {
        data = std::make_shared<std::vector<T>>(desc.size());
    }

    /** 
     * @brief Constructor that take a vector of data and array of shapes. 
     * Since vector and array has constructors that takes an initializer list, 
     * this constructor will be invoked upon calling with initializer lists. 
     * @param vec The vector from which the data will be copied.
     * @param _shape Shape of the tensor to build. Shape and #of elements in vector should match. 
     */
    Tensor(const std::vector<T>& vec, const std::array<size_t, N>& _shape)
    : data(new std::vector<T>(vec)), desc(_shape) {}

    /**
     * @brief Move version of the above constructor.
     */
    Tensor(std::vector<T>&& vec, const std::array<size_t, N>& _shape)
    : data(new std::vector<T>(vec)), desc(_shape) {}

    Tensor(TL::Range, const std::array<size_t, N>&);

    /**
     * @brief Default copy constructor that just take a reference from the given Tensor object. 
     */
    Tensor(const Tensor&) = default;

    /**
     * @brief Equality operator that increments the reference counter for rhs's and 
     * decrement for this's (Handled by shared_ptr). 
     */
    Tensor& operator=(const Tensor&) = default;

    /**
     * @brief Make and return a copy for the tensor. Note that this is a memory 
     * intensive operation.
     */
    Tensor copy();

    ~Tensor() = default;

    /* ---------- Access Operators ---------- */

    /**
     * @brief Access tensor elements from indices. Returns a lvalue reference.
     * @param Dims... should all be convertible to size_t.
     * @throws @a std::out_of_range when the given indices are out of range of tensor shape.
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
     * @param Slice An object that takes two kinds of params, 1) any type 
     * convertible to size_t 2) @a TL::Range. @a Range(low,high) access elements as [low, high).
     * Any one of the param should be @a Range. Otherwise won't be compiled.
     * @throws @a std::logic_error when Slice and tensor dimensions mismatch.
     * @throws @a std::out_of_range when Indices passed to slices goes out of range.
     */
    Tensor operator()(const Slice&);

    /**
     * @brief Constant version of accessing from slices.
     */
    const Tensor operator()(const Slice&) const;

    /** 
     * @brief Subscript operator. It also returns a reference to the tensor along
     * its 0th dimension.
     * @param idx The index to be chosen.
     * @throw std::out_of_range when index goes out of range.
     */
    Tensor<T, N-1> operator[](size_t);

    /** 
     * @brief Constant version of subscript indexing. 
     */
    const Tensor<T, N-1> operator[](size_t) const;

    /**
     * @brief An utility function that applies @a F to every element in the tensor.
     * @param F A functor type.
     */
    template <typename F>
    Tensor& _apply(F);

    /**
     * @brief An utility function that applies @a F to every element in the tensor
     * given other tensor.
     * @param Tensor A tensor.
     * @param F A functor type.
     */
    template <typename F>
    Tensor& _apply(const Tensor&, F);

    /* ---------- Arithmetic operators ---------- */

    /* Assign to a scalar */
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

    void print(std::ostream&) const;
    
    template <typename U, size_t M>
    friend std::ostream& operator<<(std::ostream&, const Tensor<U, M>&);

    friend class TensorIterator<T, N>;
    
    TensorFormatter format;
    
private:
    TL::internal::TensorDescriptor<N> desc;
    std::shared_ptr<std::vector<T>> data;
};

}   // namespace TL

#endif