#ifndef LPU_BENCH_ARCH_H
#define LPU_BENCH_ARCH_H

#define LPU_STR_IMPL(x) #x
#define LPU_STR(x) LPU_STR_IMPL(x)
#define LPU_CAT_IMPL(a, b) a##b
#define LPU_CAT(a, b) LPU_CAT_IMPL(a, b)
#define LPU_CAT3_IMPL(a, b, c) a##b##c
#define LPU_CAT3(a, b, c) LPU_CAT3_IMPL(a, b, c)

#ifdef LPU_ARCH_TOKEN
#include LPU_STR(LPU_CAT(LPU_ARCH_TOKEN, .h))
#ifndef LPU_NO_S_INTRIN
#include LPU_STR(LPU_CAT(LPU_ARCH_TOKEN, _s-intrin.h))
#endif
#define LPU_IS_SPU (LPU_CAT3(__, LPU_ARCH_TOKEN, _s__) + 0)
#define lpu_s_callm_counter LPU_CAT3(__, LPU_ARCH_TOKEN, _s_callmCounter)
#define lpu_s_set_mreg_var  LPU_CAT3(__, LPU_ARCH_TOKEN, _s_setMReg_var)
#define lpu_s_get_statb     LPU_CAT3(__, LPU_ARCH_TOKEN, _s_getStatB)
#define lpu_s_fence_i       LPU_CAT3(__, LPU_ARCH_TOKEN, _s_fenceI)
#else
#define LPU_IS_SPU 0
#endif

#endif // LPU_BENCH_ARCH_H
