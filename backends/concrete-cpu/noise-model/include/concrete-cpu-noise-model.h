// Copyright © 2022 ZAMA.
// All rights reserved.

#ifndef CONCRETE_CPU_NOISE_MODEL_FFI_H
#define CONCRETE_CPU_NOISE_MODEL_FFI_H

// Warning, this file is autogenerated by cbindgen. Do not modify this manually.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


#define FFT_SCALING_WEIGHT -2.57722494

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

double concrete_cpu_estimate_modulus_switching_noise_with_binary_key(uint64_t internal_ks_output_lwe_dimension,
                                                                     uint64_t glwe_log2_polynomial_size,
                                                                     uint32_t ciphertext_modulus_log);

double concrete_cpu_variance_blind_rotate(uint64_t in_lwe_dimension,
                                          uint64_t out_glwe_dimension,
                                          uint64_t out_polynomial_size,
                                          uint64_t log2_base,
                                          uint64_t level,
                                          uint32_t ciphertext_modulus_log,
                                          double variance_bsk);

double concrete_cpu_variance_keyswitch(uint64_t input_lwe_dimension,
                                       uint64_t log2_base,
                                       uint64_t level,
                                       uint32_t ciphertext_modulus_log,
                                       double variance_ksk);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* CONCRETE_CPU_NOISE_MODEL_FFI_H */
