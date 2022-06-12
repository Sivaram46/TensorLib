#ifndef LA_TENSOR_UTILS
#define LA_TENSOR_UTILS

#include <type_traits>

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


template <typename From, typename To>
constexpr bool Is_convertible() {
    return std::is_convertible<From, To>::value;
}

template <bool B, typename T>
using Enable_if = typename std::enable_if<B, T>::type;

template <typename... Args>
constexpr bool Are_size_convertible() {
    return All(Is_convertible<Args, size_t>()...);
}

template <bool...> struct bool_pack;
template <bool... v>
using all_true = std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>;

/* template <class... Args>
using Size_convertible = std::enable_if_t<all_true<std::is_convertible<Args, std::size_t>{}...>{}> */

}

#endif