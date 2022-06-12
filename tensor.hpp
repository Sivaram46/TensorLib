#ifndef LA_TENSOR_HEADER
#define LA_TENSOR_HEADER

#include <memory>
#include <type_traits>
#include <vector>
#include <initializer_list>

#include "tensor_slice.hpp"
#include "utils.cpp"

#include <iostream>

namespace LA {

// Forward declaration of TensorSlice
template <size_t N> class TensorSlice; 

template <typename T, size_t N>
class Tensor 
{
public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;
    using value_type = T;

    static const size_t n_dim = N;
    constexpr size_t ndim() { return n_dim; }

    constexpr size_t size() { return data->size(); }
    size_t shape(size_t dim) const { return desc.shape[dim]; }
    const std::array<size_t, N>& shape() const { return desc.shape; }
    const TensorSlice<N>& descriptor() { return desc; }

    // Iterators over the Tensor
    iterator begin() { return data->begin(); }
    iterator end() { return data->end(); }
    iterator begin() const { return data->begin(); }
    iterator end() const { return data->end(); }
    
    const_iterator cbegin() const { return data->cbegin(); }
    const_iterator cend() const { return data->cend(); }

    /* ----------- Constructors ----------- */
    // Default constructor
    Tensor() = default;
    // Tensor(const std::shared_ptr<std::vector<T>>& _data, const TensorSlice<N>& _desc)
    // : data(_data), desc(_desc) {}

    // Constructor that build Tensor from shapes
    template <typename... Shapes,
        typename = std::enable_if_t<All(Is_convertible<Shapes, size_t>()...)>
    >
    Tensor(Shapes... shapes)
    : desc(shapes...) {
        data = std::make_shared<std::vector<T>>(desc.size());
    }

    // Constructor that took data from a vector and shapes from a init list
    Tensor(const std::vector<T>& vec, const std::array<size_t, N>& _shape)
    : data(new std::vector<T>(vec)), desc(size(), _shape) {}

    Tensor(std::vector<T>&& vec, const std::array<size_t, N>& _shape)
    : data(new std::vector<T>(vec)), desc(vec.size(), _shape) {}

    // Constructs from init list
    Tensor(std::initializer_list<T> list, std::initializer_list<size_t> _shape)
    : data(new std::vector<T>(list)), desc(list.size(), _shape) {}

    // Copy constructor that just take a reference from the given Tensor object.
    Tensor(const Tensor&) = default;


    // Equality operator that increments the reference counter for rhs's and 
    // decrement for this's (Handled by shared_ptr).
    Tensor& operator=(const Tensor&) = default;

    // Make and return a copy for the tensor
    Tensor copy();

    ~Tensor() = default;

    /* ----- Arithmetic operators --------- */
    // An apply utility function that applies F to every element in Tensor
    template <typename F>
    Tensor& _apply(F);

    template <typename F>
    Tensor& _apply(const Tensor&, F);

    // Assign to a scalar
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

    /* ------- Binary Operators ---------- */
    Tensor operator+(const Tensor&);

    /* ------- Access Operators --------- */
    // Access with indices, returns T&
    template <typename... Indices>
    T& operator()(Indices...);

    /* --------- Debug ------------ */
    template <typename U, size_t M>
    friend std::ostream& operator<<(std::ostream&, const Tensor<U, M>&);

    // template <typename U>
    // friend std::ostream& operator<< <2> (std::ostream&, const Tensor<U, 2>&);
    
private:
    TensorSlice<N> desc;
    std::shared_ptr<std::vector<T>> data;
};

}

#endif