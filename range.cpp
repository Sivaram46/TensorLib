#ifndef LA_RANGE_CPP
#define LA_RANGE_CPP

namespace TL {

/**
 * Range class that can be used to select elements in some range. Can be used
 * as param of @a TL::Slice.
 */
struct Range {
    size_t low, high;

    Range(size_t _low, size_t _high) : low(_low), high(_high) {}
    explicit Range(size_t _high) : low(0), high(_high) {}
    bool single() const { return low + 1 == high; }
};

}

#endif