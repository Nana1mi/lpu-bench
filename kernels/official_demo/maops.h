#ifndef MAOPS_KERNEL_H_
#define MAOPS_KERNEL_H_
#include <stddef.h>
#include <stdint.h>

#define MAOPS_DIM_MAX 8

typedef enum {
  MAOPS_DTYPE_INVALID = 0,        /*!< An invalid data type. */
  MAOPS_DTYPE_HALF = 1,           /*!< A 16-bit floating-point data type. */
  MAOPS_DTYPE_FLOAT = 2,          /*!< A 32-bit floating-point data type. */
  MAOPS_DTYPE_DOUBLE = 14,        /*!< A 64-bit floating-point data type. */
  MAOPS_DTYPE_INT8 = 3,           /*!< An 8-bit signed integer data type. */
  MAOPS_DTYPE_INT16 = 4,          /*!< A 16-bit signed integer data type. */
  MAOPS_DTYPE_INT32 = 6,          /*!< A 32-bit signed integer data type. */
  MAOPS_DTYPE_INT64 = 9,          /*!< A 64-bit signed integer data type. */
  MAOPS_DTYPE_UINT8 = 7,          /*!< An 8-bit unsigned integer data type. */
  MAOPS_DTYPE_UINT16 = 13,        /*!< A 16-bit unsigned integer data type. */
  MAOPS_DTYPE_UINT32 = 11,        /*!< A 32-bit unsigned integer data type. */
  MAOPS_DTYPE_UINT64 = 12,        /*!< A 64-bit unsigned integer data type. */
  MAOPS_DTYPE_BOOL = 8,           /*!< A boolean data type. */
  MAOPS_DTYPE_COMPLEX_HALF = 15,  /*!< A 32-bit complex number of two fp16. */
  MAOPS_DTYPE_COMPLEX_FLOAT = 16, /*!< A 64-bit complex number of two fp32. */
  MAOPS_DTYPE_BF16 = 17,          /*!< A 16-bit B-floating-point data type. */
} maopsDataType_t;

// define DM SIZE
#define __DM_SIZE_1BYTES (262144)
#define __DM_SIZE_2BYTES (131072)
#define __DM_SIZE_4BYTES (65536)
#define __DM_SIZE_8BYTES (32768)

// table elements count
#define TANH_TABLE_0_D_COUNT (8192)
#define TANH_TABLE_1_D_COUNT (8192)
#define TB_DIVHP_TABLE_COUNT (16384)
#define TB_EXP_HEX_128_COUNT (2048)
#define TB_SIGMOID_0_COUNT (4128)
#define TB_SIGMOID_1_COUNT (8224)
#define TB_LOG2H_257_COUNT (4128)
#define TB_RCPLOG2L_257_COUNT (8224)
#define TB_LOGH_257_COUNT (4128)
#define TB_RCPLOGL_257_COUNT (8224)
#define TB_POW_EXP_COUNT (2048)
#define TB_POW_LOG2H_COUNT (4128)
#define TB_POW_RCPLOG2L_COUNT (8256)
#define TB_RSQRT_FP32_0_COUNT (8224)
#define TB_RSQRT_FP32_1_COUNT (4128)
#define TB_SIN_K02_256_COUNT (8192)
#define TB_SIN_K13_256_COUNT (8192)
#define TB_SQRT_TABLE_0_COUNT (8192)
#define TB_SQRT_TABLE_1_COUNT (8192)
#define TANH_TABLE_0_D_OLD_COUNT (32768)
#define TANH_TABLE_1_NO_D_OLD_COUNT (16384)
#define TB_POW_EXP_OLD_COUNT (4096)

// table size
#define TANH_TABLE_0_D_SIZE (32768)
#define TANH_TABLE_1_D_SIZE (32768)
#define TB_DIVHP_TABLE_SIZE (65536)
#define TB_EXP_HEX_128_SIZE (8192)
#define TB_SIGMOID_0_SIZE (16512)
#define TB_SIGMOID_1_SIZE (32896)
#define TB_LOG2H_257_SIZE (16512)
#define TB_RCPLOG2L_257_SIZE (32896)
#define TB_LOGH_257_SIZE (16512)
#define TB_RCPLOGL_257_SIZE (32896)
#define TB_POW_EXP_SIZE (8192)
#define TB_POW_LOG2H_SIZE (16512)
#define TB_POW_RCPLOG2L_SIZE (33024)
#define TB_RSQRT_FP32_0_SIZE (32896)
#define TB_RSQRT_FP32_1_SIZE (16512)
#define TB_SIN_K02_256_SIZE (32768)
#define TB_SIN_K13_256_SIZE (32768)
#define TB_SQRT_TABLE_0_SIZE (32768)
#define TB_SQRT_TABLE_1_SIZE (32768)
#define TANH_TABLE_0_D_OLD_SIZE (131072)
#define TANH_TABLE_1_NO_D_OLD_SIZE (65536)
#define TB_POW_EXP_OLD_SIZE (16384)

// table addr
extern __device__ long long MATH_TABLE_ADDR;
#define TANH_TABLE_0_D_ADDR (&MATH_TABLE_ADDR)
#define TANH_TABLE_1_D_ADDR ((&MATH_TABLE_ADDR) + 0x00001000)
#define TB_DIVHP_TABLE_ADDR ((&MATH_TABLE_ADDR) + 0x00002000)
#define TB_EXP_HEX_128_ADDR ((&MATH_TABLE_ADDR) + 0x00004000)
#define TB_SIGMOID_0_ADDR ((&MATH_TABLE_ADDR) + 0x00004400)
#define TB_SIGMOID_1_ADDR ((&MATH_TABLE_ADDR) + 0x00004C10)
#define TB_LOG2H_257_ADDR ((&MATH_TABLE_ADDR) + 0x00005C20)
#define TB_RCPLOG2L_257_ADDR ((&MATH_TABLE_ADDR) + 0x00006430)
#define TB_LOGH_257_ADDR ((&MATH_TABLE_ADDR) + 0x00007440)
#define TB_RCPLOGL_257_ADDR ((&MATH_TABLE_ADDR) + 0x00007C50)
#define TB_POW_EXP_ADDR ((&MATH_TABLE_ADDR) + 0x00008C60)
#define TB_POW_LOG2H_ADDR ((&MATH_TABLE_ADDR) + 0x00009060)
#define TB_POW_RCPLOG2L_ADDR ((&MATH_TABLE_ADDR) + 0x00009870)
#define TB_RSQRT_FP32_0_ADDR ((&MATH_TABLE_ADDR) + 0x0000A890)
#define TB_RSQRT_FP32_1_ADDR ((&MATH_TABLE_ADDR) + 0x0000B8A0)
#define TB_SIN_K02_256_ADDR ((&MATH_TABLE_ADDR) + 0x0000C0B0)
#define TB_SIN_K13_256_ADDR ((&MATH_TABLE_ADDR) + 0x0000D0B0)
#define TB_SQRT_TABLE_0_ADDR ((&MATH_TABLE_ADDR) + 0x0000E0B0)
#define TB_SQRT_TABLE_1_ADDR ((&MATH_TABLE_ADDR) + 0x0000F0B0)
#define TANH_TABLE_0_D_OLD_ADDR ((&MATH_TABLE_ADDR) + 0x000100B0)
#define TANH_TABLE_1_NO_D_OLD_ADDR ((&MATH_TABLE_ADDR) + 0x000140B0)
#define TB_POW_EXP_OLD_ADDR ((&MATH_TABLE_ADDR) + 0x000160B0)
#endif