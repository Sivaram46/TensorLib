#ifndef LA_TENSOR_HEADER
#define LA_TENSOR_HEADER

#include <memory>
#include <type_traits>
#include <vector>
#include <initializer_list>
#include <iostream>

#include "tensor_descriptor.hpp"
#include "utils.cpp"
#include "slice.cpp"

namespace LA {

using std::size_t;
// Forward declaration of TensorDescriptor
template <size_t N> class TensorDescriptor; 

/**
 * A compile time tensor of type @a T and dimension @a N
 */
template <typename T, size_t N>
class Tensor 
{
public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using value_type = T;

    static const size_t n_dim = N;

    constexpr size_t ndim() const { return n_dim; }
    constexpr size_t size() const { return desc.size(); }

    size_t shape(size_t dim) const { return desc.shape[dim]; }
    const std::array<size_t, N>& shape() const { return desc.shape; }
    constexpr bool empty() const { return !size(); }

    const TensorDescriptor<N>& descriptor() const { return desc; }
    const std::array<size_t, N>& get_stride() const { return desc.stride; }

    /* ---------- Iterators over the Tensor ---------- */
    /**
     * Returns a forward iterator over the data. Iterator of the underlying std::vector.
     * It is not advisable to call this on sub-matrices.
     */
    iterator begin() { return data->begin(); }
    iterator end() { return data->end(); }
    const_iterator begin() const { return data->begin(); }
    const_iterator end() const { return data->end(); }
    
    const_iterator cbegin() const { return data->cbegin(); }
    const_iterator cend() const { return data->cend(); }

    /* ---------- Constructors ---------- */
    /* The default constructor */
    Tensor() : data(new std::vector<T>()) {}

    Tensor(std::shared_ptr<std::vector<T>> _data, const TensorDescriptor<N>& _desc)
    : data(_data), desc(_desc) {}

    /**
     * @brief Constructor that build Tensor from shapes. This constructs an empty tensor.
     * @param Shapes... should all be convertible to size_t, otherwise the constructor
     * won't be enabled at compile time.
     */
    template <typename... Shapes,
        typename = std::enable_if_t<All(Is_convertible<Shapes, size_t>()...)>
    >
    Tensor(Shapes... shapes)
    : desc(shapes...) {
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
    : data(new std::vector<T>(vec)), desc(vec.size(), _shape) {}

    /**
     * @brief Move version of the above constructor.
     */
    Tensor(std::vector<T>&& vec, const std::array<size_t, N>& _shape)
    : data(new std::vector<T>(vec)), desc(vec.size(), _shape) {}

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
     * convertible to size_t 2) @a LA::Range. @a Range(low,high) access elements as [low, high).
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

    // Operate with scalars
    Tensor& operator+=(const T&);
    Tensor& operator-=(const T&);
    Tensor& operator*=(const T&);
    Tensor& operator/=(const T&);
    Tensor& operator%=(const T&);

    // Operate with Tensors
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

    /* --------- Debug ------------ */
    template <typename U, size_t M>
    friend std::ostream& operator<<(std::ostream&, const Tensor<U, M>&);
    
private:
    TensorDescriptor<N> desc;
    std::shared_ptr<std::vector<T>> data;
};

}

#endif