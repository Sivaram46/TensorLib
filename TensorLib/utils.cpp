#ifndef TENSORLIB_TENSOR_UTILS
#define TENSORLIB_TENSOR_UTILS

#include <type_traits>

namespace TL {

using std::size_t;
struct Range;

constexpr bool All() { return true; }

/** @brief A predicate function that applies the predicate to all the arguments. 
 * Useful for checking whether all the arguments can be converted into particular
 * type. 
 */
template <typename... Args>
constexpr bool All(bool b, Args... args) {
    return b && All(args...);
}

constexpr bool Some() { return false; }

/** 
 * Similar to @a All, but returns whether any one of the predicate satisfies.
 */
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
    /* First, some of the arguments should be convertible to Range. Then other
    params should be convertible to size_t. That can be done by taking OR of 
    Is_same(Args, Range) and Is_convertible(Args, size_t). */
    return Some(Is_same<Args, TL::Range>()...) &&
        All((Is_same<Args, TL::Range>() || Is_convertible<Args, size_t>())...);
}

template <typename... Args>
constexpr bool Element_valid() {
    return All(Is_convertible<Args, size_t>()...);
}

template <bool B, typename T>
using Enable_if = typename std::enable_if<B, T>::type;

}   // namespace TL

#endif