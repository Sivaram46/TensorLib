#ifndef TENSORLIB_SLICE_CPP
#define TENSORLIB_SLICE_CPP

#include <type_traits>
#include <vector>

#include "range.cpp"
#include "utils.cpp"

namespace TL {

/**
 * Slice class that takes @a TL::Range and size_t and stores the ranges that will 
 * be used for tesnor slicing.
 */
struct Slice {
    template <typename... Dims,
        typename = std::enable_if_t<TL::Slice_valid<Dims...>()>
    >
    Slice(Dims... dims) { put_range(dims...); }
    std::vector<Range> ranges;
private:
    /* Utility functions that will convert the given Dims... to Ranges */

    template <typename... Dims>
    void put_range(Range r, Dims... dims) {
        ranges.push_back(r);
        put_range(dims...);
    }

    template <typename... Dims>
    void put_range(size_t s, Dims... dims) {
        ranges.push_back(Range(s, s+1));
        put_range(dims...);
    }

    void put_range(Range r) {
        ranges.push_back(r);
    }

    void put_range(size_t s) {
        ranges.push_back(Range(s, s+1));
    }
};

}   // namespace TL

#endif