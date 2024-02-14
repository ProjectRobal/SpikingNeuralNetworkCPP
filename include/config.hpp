#pragma once

#include <cstdint>
#include <experimental/simd>

// specific number types used by neurons
typedef long double number;

#define MAX_SIMD_VECTOR_SIZE std::experimental::simd_abi::max_fixed_size<number>

typedef std::experimental::fixed_size_simd<number , MAX_SIMD_VECTOR_SIZE> SIMD;

typedef std::experimental::fixed_size_simd_mask<number , MAX_SIMD_VECTOR_SIZE> SIMD_MASK;

#define MAITING_THRESHOLD 0.4f

#define AMOUNT_THAT_PASS 0.4f

#define USESES_TO_MAITING 10

#define MAX_THREAD_POOL 8

#define NEURON_THRESHOLD_LEVEL 50.f