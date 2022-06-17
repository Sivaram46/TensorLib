#ifndef LA_SLICE_CPP
#define LA_SLICE_CPP

#include <type_traits>
#include <vector>
#include "range.cpp"
#include "utils.cpp"

namespace LA {

struct Slice {
    template <typename... Dims,
        typename = std::enable_if_t<LA::Slice_valid<Dims...>()>
    >
    Slice(Dims... dims) { put_range(dims...); }
    std::vector<Range> ranges;
private:
    void put_range(Range);
    void put_range(size_t);

    template <typename... Dims>
    void put_range(Range, Dims...);

    template <typename... Dims>
    void put_range(size_t, Dims...);
};

void Slice::put_range(Range r) {
    ranges.push_back(r);
}

void Slice::put_range(size_t s) {
    ranges.push_back(Range(s, s+1));
}

template <typename... Dims>
void Slice::put_range(Range r, Dims... dims) {
    ranges.push_back(r);
    put_range(dims...);
}

template <typename... Dims>
void Slice::put_range(size_t s, Dims... dims) {
    ranges.push_back(Range(s, s+1));
    put_range(dims...);
}

}

#endif