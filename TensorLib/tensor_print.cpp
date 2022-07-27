#ifndef TENSORLIB_TENSOR_PRINT_CPP
#define TENSORLIB_TENSOR_PRINT_CPP

#include <iostream>
#include <vector>
#include <sstream>

#include "tensor_formatter.cpp"

namespace TL {
    
namespace internal {

template <typename T>
class TensorPrint
{
public:
    TensorPrint(std::ostream& _out, const Tensor<T>& _tensor)
    : out(_out), tensor(_tensor) {}

    void print() const;

private:
    std::ostream& out;
    const Tensor<T>& tensor;

    size_t calculate_width() const;
    void basic_print(std::stringstream&) const;
};

template <typename T>
size_t TensorPrint<T>::calculate_width() const {

    std::stringstream element;
    size_t max_width = 1;

    typename Tensor<T>::const_iterator it = tensor.begin();
    for (; it != tensor.end(); ++it) {
        element.str(std::string());
        element << *it;
        max_width = std::max(max_width, static_cast<size_t>(element.tellp()));
    }
    
    max_width = std::min(max_width, static_cast<size_t> (tensor.format.precision));
    return max_width;
}

template <typename T>
void TensorPrint<T>::basic_print(std::stringstream& element) const {
    auto N = tensor.ndim();
    std::vector<size_t> stride(N, 1);
    /* Calculate temporary stride for the tensor */
    for (long i = N - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * tensor.shape()[i + 1];
    }

    auto format = tensor.format;
    size_t size = tensor.size();
    auto width = calculate_width();

    typename Tensor<T>::const_iterator it = tensor.begin();
    for (size_t i = 0; i < size; ++i, ++it) {
        element.str(std::string());
        element.width(width);

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
            out << std::string(N, '[');
        }
        else if (i == size - 1) {
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

template <typename T>
void TensorPrint<T>::print() const {
    std::stringstream element;
    auto format = tensor.format;

    // Set floating precision
    if (format.precision >= 0) {
        element.precision(format.precision);
    }

    // Set float modes
    switch (format.float_mode)
    {
    case TensorFormatter::FloatMode::Default:
        element << std::defaultfloat;
        break;

    case TensorFormatter::FloatMode::Fixed:
        element << std::fixed;
        break;

    case TensorFormatter::FloatMode::Scientific:
        element << std::scientific;
        break;
    }

    // TODO: Function to print only summary

    if (format.linewidth < 0) {
        basic_print(element);
        return;
    }
    else {
        // TODO: Logic for printing given linewidth
    }
}

}   // namespace internal

}   // namespace TL

#endif