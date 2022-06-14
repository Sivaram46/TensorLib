#ifndef LA_SLICE_CPP
#define LA_SLICE_CPP

#include <type_traits>

namespace LA {

using std::size_t;
struct Slice
{
    size_t low, high;
    
    Slice(size_t i, size_t j) : low(i), high(j) {}
    Slice(size_t i) : low(i), high(i + 1) {}
    operator size_t ();
};

}

#endif