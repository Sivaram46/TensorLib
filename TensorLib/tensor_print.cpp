#include <iostream>
#include <array>
#include <sstream>

namespace TL {
    
namespace internal {

template <typename T, size_t N>
class TensorPrint
{
public:
    TensorPrint(std::ostream& _out, const Tensor<T, N>& _tensor)
    : out(_out), tensor(_tensor) {}

    void print() const;

private:
    std::ostream& out;
    const Tensor<T, N>& tensor;

    size_t calculate_width() const;
};

template <typename T, size_t N>
size_t TensorPrint<T, N>::calculate_width() const {

    std::stringstream element;
    size_t max_width = 1;

    typename Tensor<T, N>::iterator it = tensor.begin();
    for (; it != tensor.end(); ++it) {
        element.str(std::string());
        element << *it;
        max_width = std::max(max_width, static_cast<size_t>(element.tellp()));
    }
    
    return max_width;
}

template <typename T, size_t N>
void TensorPrint<T, N>::print() const {
    std::array<size_t, N> stride;
    stride[N - 1] = 1;
    /* Calculate temporary stride for the tensor */
    for (long i = N - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * tensor.shape(i);
    }

    auto format = tensor.get_format();

    std::stringstream element;
    size_t size = tensor.size();
    typename Tensor<T, N>::iterator it = tensor.begin();
    for (size_t i = 0; i < size; ++i, ++it) {
        element.str(std::string());
        element.width(calculate_width());

        long matched = 0;
        /* Logic:
            If an element at index i divides tensor's stride k times, had to put 
            k close brackets and newline character. 
        */
        for (long j = N - 2; j >= 0; --j) {
            if ((i + 1) % stride[j] == 0) {
                matched++;
            }
            else {
                break;
            }
        }

        element << *it;

        // Base case having only one element
        if (size == 1) {
            out << std::string(N, '[')
                << element.str()
                << std::string(N, ']')
                << '\n';
            return;
        }

        // Special cases for first and last elements
        if (i == 0) {
            out << std::string(N, '[')
                << element.str()
                << format.sep;
            continue;
        }
        if (i == size - 1) {
            out << element.str()
                << std::string(N, ']')
                << '\n';
            break;
        }

        // Print sep and close and open brackets
        out << element.str();
        if (!matched) {
            out << format.sep;
        }
        else {
            out << std::string(matched, ']')
                << std::string(matched, '\n')
                << std::string(N - matched, ' ')
                << std::string(matched, '[');    
        }
    }
}

}   // namespace internal

}   // namespace TL
