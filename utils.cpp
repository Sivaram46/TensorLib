#ifndef LA_TENSOR_UTILS
#define LA_TENSOR_UTILS

#include <type_traits>
#include "range.cpp"

/*
TODOS:
    Tensor specialization for Matrix
        matmul()
        transpose() or T()

    squeeze(), expand_dims(), reshape(), resize(), transpose()

    Binary operation on type different tensors

    Running time tensors
 */

namespace LA {

using std::size_t;

constexpr bool All() { return true; }

// A predicate function that applies the predicate to all the arguments. 
// Useful for checking whether all the arguments can be converted into particular
// type.
template <typename... Args>
constexpr bool All(bool b, Args... args) {
    return b && All(args...);
}

constexpr bool Some() { return false; }

template <typename... Args>
constexpr bool Some(bool b, Args... args) {
    return b || Some(args...);
}

template <typename From, typename To>
constexpr bool Is_convertible() {
    return std::is_convertible<From, To>::value;
}

template <typename T1, typename T2>
constexpr bool Is_same() {
    return std::is_same<T1, T2>::value;
}

template <typename... Args>
constexpr bool Slice_valid() {
    return Some(Is_same<Args, LA::Range>()...) &&
        All((Is_same<Args, LA::Range>() || Is_convertible<Args, size_t>())...);

}

template <typename... Args>
constexpr bool Element_valid() {
    return All(Is_convertible<Args, size_t>()...);
}

template <bool Cond1, bool Cond2, typename T1, typename T2>
struct enable_if2 {};

template <bool Cond2, typename T1, typename T2>
struct enable_if2<true, Cond2, T1, T2> {
    using type = T1;
};

template <bool Cond1, typename T1, typename T2>
struct enable_if2<Cond1, true, T1, T2> {
    using type = T2;
};

template <bool Cond1, bool Cond2, typename T1, typename T2>
using enable_if2_t = typename enable_if2<Cond1, Cond2, T1, T2>::type;

template <bool B, typename T>
using Enable_if = typename std::enable_if<B, T>::type;

}

#endif