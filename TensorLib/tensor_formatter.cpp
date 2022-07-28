#ifndef TENSORLIB_TENSOR_FORMATTER_CPP
#define TENSORLIB_TENSOR_FORMATTER_CPP

#include <string>

namespace TL {

struct TensorFormatter
{
    /* Floating point precision. -1 for default value. */
    int precision;
    /* Max elements to be print. -1 for printing all elements. */
    int max_elements;
    /* Maximum linewidth to be print. -1 for no restriction on linewidth. */
    int linewidth;  
    /* Whether to print sign character or not. */
    bool sign;
    /* seperator for successive elements in tensor. Default to ", " */
    std::string sep;

    /**
     * @brief Float printing modes.
     * @a Scientific - Prints floats in scientific format
     * @a Fixed - Prints floats in fixed format
     * @a Default - Default float print mode.
     */
    enum class FloatMode { 
        Scientific, 
        Fixed, 
        Default
    } float_mode;

    TensorFormatter(
        int _precision = -1, int _max_elements = -1, int _linewidth = -1,
        bool _sign = false, const std::string& _sep = ", ", 
        FloatMode _float_mode = FloatMode::Default
    ) : precision(_precision), max_elements(_max_elements), linewidth(_linewidth),
    sign(_sign), sep(_sep), float_mode(_float_mode) {}
};

}   // namespace TL

#endif