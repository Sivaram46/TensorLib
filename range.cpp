#ifndef LA_RANGE_CPP
#define LA_RANGE_CPP


namespace LA {

struct Range {
    size_t low, high;

    Range(size_t _low, size_t _high) : low(_low), high(_high) {}
    explicit Range(size_t _high) : low(0), high(_high) {}
    bool single() const { return low + 1 == high; }
};

}

#endif