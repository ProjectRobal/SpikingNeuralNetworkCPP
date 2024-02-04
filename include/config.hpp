#pragma once

#include <cstdint>
#include <experimental/simd>

// specific number types used by neurons
typedef long double number;

#define MAX_SIMD_VECTOR_SIZE std::experimental::simd_abi::max_fixed_size<number>

typedef std::experimental::fixed_size_simd<long double, MAX_SIMD_VECTOR_SIZE> SIMD;