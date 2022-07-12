#include <string>

namespace TL {

struct TensorFormatter
{
    int precision;
    int max_elements;
    int linewidth;  
    bool sign;
    std::string sep;

    enum class FloatMode { 
        scientific, fixed 
    } float_mode;

    TensorFormatter(
        int _precision = -1, int _max_elements = -1, int _linewidth = -1,
        bool _sign = false, const std::string& _sep = ", ", 
        FloatMode _float_mode = FloatMode::fixed
    ) : precision(_precision), max_elements(_max_elements), linewidth(_linewidth),
    sign(_sign), sep(_sep), float_mode(_float_mode) {}
};

}   // namespace TL