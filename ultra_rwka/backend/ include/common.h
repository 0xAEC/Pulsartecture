// ultra_rwka/backend/include/common.h

#pragma once 

#include <cmath>       // For std::exp, std::sqrt, etc.
#include <cstdint>     // For fixed-width integer types like int64_t
#include <stdexcept>   // For standard exceptions like std::runtime_error
#include <string>      // For using std::string in error messages
#include <sstream>     // For formatting error messages
#include <iostream>    // For basic IO (e.g., debug prints)

//----------------------------------------------------------------------------
// Error Handling Macros
//----------------------------------------------------------------------------

// Macro to check CUDA calls for errors
// Only active when compiled with NVCC (__CUDACC__ is defined by nvcc)
#ifdef __CUDACC__
#include <cuda_runtime.h> // Include CUDA runtime API

// Define a helper function to avoid macro complexity within the macro itself
inline void __cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error in " << file << ":" << line
           << ": " << cudaGetErrorString(err) << " (" << err << ")";
        // Consider logging instead of throwing if used in destructors etc.
        throw std::runtime_error(ss.str());
    }
}

#define CUDA_CHECK(call) __cudaCheck((call), __FILE__, __LINE__)

#else
// Define a dummy CUDA_CHECK for CPU-only compilation to avoid errors
#define CUDA_CHECK(call) call
#endif // __CUDACC__


// General assertion macro for internal checks
// Throws std::runtime_error on failure
#define URWKA_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::stringstream ss; \
            ss << "Assertion failed in " << __FILE__ << ":" << __LINE__ \
               << ": (" << #condition << "). Message: " << (message); \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)

//----------------------------------------------------------------------------
// Debugging Macros (Optional)
//----------------------------------------------------------------------------

// Example: Conditional debug printing macro
// Define URWKA_DEBUG_PRINT during compilation (e.g., -DURWKA_DEBUG_PRINT) to enable
#ifdef URWKA_DEBUG_PRINT
#define DEBUG_PRINT(...) \
    do { \
        fprintf(stderr, "DEBUG [%s:%d] ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } while(0)
#else
#define DEBUG_PRINT(...) do {} while(0) // Compiles to nothing if not defined
#endif


//----------------------------------------------------------------------------
// Constants (Optional)
//----------------------------------------------------------------------------

// Example: Define constants if needed frequently across kernels
// namespace ultra_rwka {
// namespace backend {
//     namespace constants {
//         constexpr double PI = 3.14159265358979323846;
//         constexpr float PI_F = 3.1415926535f;
//     } // namespace constants
// } // namespace backend
// } // namespace ultra_rwka


//----------------------------------------------------------------------------
// Utility Functions (Optional - Keep minimal in common.h)
//----------------------------------------------------------------------------

// Example: Basic clamp function (though std::clamp exists in C++17)
// template<typename T>
// inline T clamp(T val, T min_val, T max_val) {
//     return std::max(min_val, std::min(val, max_val));
// }


// Note: Avoid including heavy dependencies like PyTorch headers here
// unless absolutely necessary and common across ALL backend code.
// Typically, PyTorch headers are included directly in the .cpp/.cu files
// that need them (e.g., binding files or kernel implementations).
