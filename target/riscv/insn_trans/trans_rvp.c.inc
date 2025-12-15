/*
 * RISC-V translation routines for the P Standard Extensions.
 *
 * Copyright (c) 2020-2023 PLCT Lab
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */


/* Save a 64 bit data in src to dst and dst + 1 */
static void set_pair_regs(DisasContext *ctx, int dst, TCGv_i64 src)
{
#if defined(TARGET_RISCV32)
    TCGv_i64 tl_64 = tcg_temp_new_i64();
    TCGv_i64 th_64 = tcg_temp_new_i64();
    TCGv_i32 tl_32 = tcg_temp_new_i32();
    TCGv_i32 th_32 = tcg_temp_new_i32();
    tcg_gen_extract_i64(tl_64, src, 0, 32);
    tcg_gen_extract_i64(th_64, src, 32, 32);
    tcg_gen_trunc_i64_tl(tl_32, tl_64);
    tcg_gen_trunc_i64_tl(th_32, th_64);
    gen_set_gpr(ctx, dst, tl_32);
    gen_set_gpr(ctx, dst + 1, th_32);
# else
    gen_set_gpr(ctx, dst, src);
#endif
}

/* Concat two 32 bit data in src and src + 1 to dst */
static void get_pair_regs(DisasContext *ctx, TCGv_i64 dst, int src)
{
#if defined(TARGET_RISCV32)
    TCGv t1 = get_gpr(ctx, src, EXT_NONE);
    TCGv t2 = get_gpr(ctx, src + 1, EXT_NONE);
    tcg_gen_concat_i32_i64(dst, t1, t2);
#else
    TCGv t1 = get_gpr(ctx, src, EXT_NONE);
    tcg_gen_mov_tl(dst, t1);
#endif
}

#define GEN_SIMD_TRANS(NAME)                                \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
   TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);              \
   TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);              \
   TCGv dest = dest_gpr(ctx, a->rd);                        \
   gen_helper_##NAME(dest, tcg_env, src1, src2);            \
   return true;                                             \
}

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_32(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, src2);           \
    return true;                                            \
}                                                           
#else
#define GEN_SIMD_TRANS_32(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    return true;                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_64(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
   REQUIRE_64BIT(ctx);                                      \
   return true;                                             \
}                                                           
#else
#define GEN_SIMD_TRANS_64(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, src2);           \
    return true;                                            \
}
#endif

#define GEN_SIMD_TRANS_RD(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    TCGv t = tcg_temp_new();                                \
    gen_helper_##NAME(t, tcg_env, src1, src2, dest);        \
    gen_set_gpr(ctx, a->rd, t);                             \
    return true;                                            \
}

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_RD_32(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    TCGv t = tcg_temp_new();                                \
    gen_helper_##NAME(t, tcg_env, src1, src2, dest);        \
    gen_set_gpr(ctx, a->rd, t);                             \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_RD_32(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    return true;                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_RD_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_RD_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    TCGv t = tcg_temp_new();                                \
    gen_helper_##NAME(t, tcg_env, src1, src2, dest);        \
    gen_set_gpr(ctx, a->rd, t);                             \
    return true;                                            \
}
#endif

#define GEN_SIMD_TRANS_R1(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1);                 \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_R1_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_R1_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1);                 \
    return true;                                            \
}
#endif

#define GEN_SIMD_TRANS_IMM(NAME)                            \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv imm = tcg_constant_tl(a->imm);                     \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, imm);            \
    return true;                                            \
}

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_IMM_32(NAME)                         \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv imm = tcg_constant_tl(a->imm);                     \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, imm);            \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_IMM_32(NAME)                         \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    return true;                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_IMM_64(NAME)                         \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_IMM_64(NAME)                         \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv imm = tcg_constant_tl(a->imm);                     \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, imm);            \
    return true;                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_RD_P(NAME)                           \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv_i64 rs1_64 = tcg_temp_new_i64();                   \
    TCGv_i64 rs2_64 = tcg_temp_new_i64();                   \
    tcg_gen_extu_tl_i64(rs1_64, src1);                      \
    tcg_gen_extu_tl_i64(rs2_64, src2);                      \
    TCGv_i64 t = tcg_temp_new_i64();                        \
    gen_helper_##NAME(t, tcg_env, rs1_64, rs2_64);          \
    set_pair_regs(ctx, (a->rd)*2, t);                       \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_RD_P(NAME)                           \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    return true;                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_IMM_RD_P(NAME)                       \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv_i64 rs1_64 = tcg_temp_new_i64();                   \
    tcg_gen_extu_tl_i64(rs1_64, src1);                      \
    TCGv_i64 imm = tcg_constant_i64(a->imm);                \
    TCGv_i64 t = tcg_temp_new_i64();                        \
    gen_helper_##NAME(t, tcg_env, rs1_64, imm);             \
    set_pair_regs(ctx, (a->rd)*2, t);                       \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_IMM_RD_P(NAME)                       \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    return true;                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_RD_RD_P(NAME)                        \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv_i64 rs1_64 = tcg_temp_new_i64();                   \
    TCGv_i64 rs2_64 = tcg_temp_new_i64();                   \
    tcg_gen_extu_tl_i64(rs1_64, src1);                      \
    tcg_gen_extu_tl_i64(rs2_64, src2);                      \
    TCGv_i64 t = tcg_temp_new_i64();                        \
    if(a->rd == 0){                                         \
        tcg_gen_movi_i64(t, 0);                             \
    }else{                                                  \
        get_pair_regs(ctx, t, (a->rd)*2);                   \
    }                                                       \
    gen_helper_##NAME(t, tcg_env, rs1_64, rs2_64, t);       \
    set_pair_regs(ctx, (a->rd)*2, t);                       \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_RD_RD_P(NAME)                        \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    return true;                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_DX_P(NAME1, NAME2, NAME3)                                                \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a) \
{                                                                                               \
    REQUIRE_32BIT(ctx);                                                                         \
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);                                           \
    TCGv src2_0 = get_gpr(ctx, (a->rs2)*2, EXT_NONE);                                           \
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);                                                     \
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);                                         \
    TCGv src2_1 = get_gpr(ctx, (a->rs2)*2+1, EXT_NONE);                                         \
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);                                                   \
    gen_helper_##NAME1##_##NAME3(dest_0, tcg_env, src1_0, src2_0);                                \
    gen_helper_##NAME1##_##NAME3(dest_1, tcg_env, src1_1, src2_1);                                \
    return true;                                                                                \
}
#else
#define GEN_SIMD_TRANS_DX_P(NAME1, NAME2, NAME3)                                                \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a) \
{                                                                                               \
    REQUIRE_32BIT(ctx);                                                                         \
    return true;                                                                                \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_DX_R1_P(NAME1, NAME2, NAME3)                                             \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a) \
{                                                                                               \
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);                                           \
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);                                                     \
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);                                         \
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);                                                   \
    gen_helper_##NAME1##_##NAME3(dest_0, tcg_env, src1_0);                                      \
    gen_set_gpr(ctx, (a->rd)*2, dest_0);                                                        \
    gen_helper_##NAME1##_##NAME3(dest_1, tcg_env, src1_1);                                      \
    gen_set_gpr(ctx, (a->rd)*2+1, dest_1);                                                      \
    return true;                                                                                \
}
#else
#define GEN_SIMD_TRANS_DX_R1_P(NAME1, NAME2, NAME3)                                             \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a) \
{                                                                                               \
    REQUIRE_32BIT(ctx);                                                                         \
    return true;                                                                                \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_DW_P(NAME)                                                           \
static bool trans_p##NAME##_dw(DisasContext *ctx, arg_p##NAME##_dw * a)                     \
{                                                                                           \
    REQUIRE_32BIT(ctx);                                                                     \
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);                                       \
    TCGv src2_0 = get_gpr(ctx, (a->rs2)*2, EXT_NONE);                                       \
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);                                                 \
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);                                     \
    TCGv src2_1 = get_gpr(ctx, (a->rs2)*2+1, EXT_NONE);                                     \
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);                                               \
    gen_helper_##NAME(dest_0, tcg_env, src1_0, src2_0);                                     \
    gen_helper_##NAME(dest_1, tcg_env, src1_1, src2_1);                                     \
    return true;                                                                            \
}
#else
#define GEN_SIMD_TRANS_DW_P(NAME)                                                           \
static bool trans_p##NAME##_dw(DisasContext *ctx, arg_p##NAME##_dw * a)                     \
{                                                                                           \
    REQUIRE_32BIT(ctx);                                                                     \
    return true;                                                                            \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_IMM_32_P(NAME1, NAME2, NAME3)                                             \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a)  \
{                                                                                                \
    REQUIRE_32BIT(ctx);                                                                          \
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);                                            \
    TCGv imm_0 = tcg_constant_tl(a->imm);                                                        \
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);                                                      \
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);                                          \
    TCGv imm_1 = tcg_constant_tl(a->imm);                                                        \
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);                                                    \
    gen_helper_##NAME1##_##NAME3(dest_0, tcg_env, src1_0, imm_0);                                \
    gen_helper_##NAME1##_##NAME3(dest_1, tcg_env, src1_1, imm_1);                                \
    return true;                                                                                 \
}
#else
#define GEN_SIMD_TRANS_IMM_32_P(NAME1, NAME2, NAME3)                                             \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a)  \
{                                                                                                \
    REQUIRE_32BIT(ctx);                                                                          \
    return true;                                                                                 \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_R1_P(NAME1, NAME2, NAME3)                                                 \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a)  \
{                                                                                                \
    REQUIRE_32BIT(ctx);                                                                          \
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);                                            \
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);                                                      \
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);                                          \
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);                                                    \
    gen_helper_##NAME1##_##NAME3(dest_0, tcg_env, src1_0);                                       \
    gen_helper_##NAME1##_##NAME3(dest_1, tcg_env, src1_1);                                       \
    return true;                                                                                 \
}
#else
#define GEN_SIMD_TRANS_R1_P(NAME1, NAME2, NAME3)                                                 \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a)  \
{                                                                                                \
    REQUIRE_32BIT(ctx);                                                                          \
    return true;                                                                                 \
}
#endif

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_DXS_P(NAME1, NAME2, NAME3)                                                \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a) \
{                                                                                               \
    REQUIRE_32BIT(ctx);                                                                         \
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);                                           \
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);                                                     \
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);                                         \
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);                                                   \
    TCGv src2   = get_gpr(ctx, a->rs2, EXT_NONE);                                               \
    gen_helper_##NAME1##_##NAME3(dest_0, tcg_env, src1_0, src2);                                \
    gen_helper_##NAME1##_##NAME3(dest_1, tcg_env, src1_1, src2);                                \
    return true;                                                                                \
}
#else
#define GEN_SIMD_TRANS_DXS_P(NAME1, NAME2, NAME3)                                               \
static bool trans_##NAME1##_##NAME2##NAME3(DisasContext *ctx, arg_##NAME1##_##NAME2##NAME3 * a) \
{                                                                                               \
    REQUIRE_32BIT(ctx);                                                                         \
    return true;                                                                                \
}
#endif


GEN_SIMD_TRANS(padd_b)
GEN_SIMD_TRANS(padd_bs)
GEN_SIMD_TRANS(psub_b)
GEN_SIMD_TRANS(psadd_b)
GEN_SIMD_TRANS(psaddu_b)
GEN_SIMD_TRANS(pssub_b)
GEN_SIMD_TRANS(pssubu_b)
GEN_SIMD_TRANS(paadd_b)
GEN_SIMD_TRANS(paaddu_b)
GEN_SIMD_TRANS(pasub_b)
GEN_SIMD_TRANS(pasubu_b)
GEN_SIMD_TRANS(pdifsumu_b)
GEN_SIMD_TRANS_RD(pdifsumau_b)
GEN_SIMD_TRANS_R1(psabs_b)
GEN_SIMD_TRANS(pdif_b)
GEN_SIMD_TRANS(pdifu_b)
GEN_SIMD_TRANS(predsum_bs)
GEN_SIMD_TRANS(predsumu_bs)

GEN_SIMD_TRANS(padd_h)
GEN_SIMD_TRANS(padd_hs)
GEN_SIMD_TRANS(psub_h)
GEN_SIMD_TRANS(psadd_h)
GEN_SIMD_TRANS(psaddu_h)
GEN_SIMD_TRANS(pssub_h)
GEN_SIMD_TRANS(pssubu_h)
GEN_SIMD_TRANS(paadd_h)
GEN_SIMD_TRANS(paaddu_h)
GEN_SIMD_TRANS(pasub_h)
GEN_SIMD_TRANS(pasubu_h)

GEN_SIMD_TRANS(psh1add_h)
GEN_SIMD_TRANS(pssh1sadd_h)

GEN_SIMD_TRANS(pas_hx)
GEN_SIMD_TRANS(psa_hx)
GEN_SIMD_TRANS(psas_hx)
GEN_SIMD_TRANS(pssa_hx)
GEN_SIMD_TRANS(paas_hx)
GEN_SIMD_TRANS(pasa_hx)

GEN_SIMD_TRANS(pdif_h)
GEN_SIMD_TRANS(pdifu_h)
GEN_SIMD_TRANS_R1(psabs_h)

GEN_SIMD_TRANS(predsum_hs)
GEN_SIMD_TRANS(predsumu_hs)

GEN_SIMD_TRANS_32(sadd)
GEN_SIMD_TRANS_32(saddu)
GEN_SIMD_TRANS_32(ssub)
GEN_SIMD_TRANS_32(ssubu)
GEN_SIMD_TRANS_32(aadd)
GEN_SIMD_TRANS_32(aaddu)
GEN_SIMD_TRANS_32(asub)
GEN_SIMD_TRANS_32(asubu)
GEN_SIMD_TRANS_32(ssh1sadd)

GEN_SIMD_TRANS_64(padd_ws)
GEN_SIMD_TRANS_64(padd_w)
GEN_SIMD_TRANS_64(psub_w)
GEN_SIMD_TRANS_64(psadd_w)
GEN_SIMD_TRANS_64(psaddu_w)
GEN_SIMD_TRANS_64(pssub_w)
GEN_SIMD_TRANS_64(pssubu_w)
GEN_SIMD_TRANS_64(paadd_w)
GEN_SIMD_TRANS_64(paaddu_w)
GEN_SIMD_TRANS_64(pasub_w)
GEN_SIMD_TRANS_64(pasubu_w)
GEN_SIMD_TRANS_64(psh1add_w)
GEN_SIMD_TRANS_64(pssh1sadd_w)
GEN_SIMD_TRANS_64(pas_wx)
GEN_SIMD_TRANS_64(psa_wx)
GEN_SIMD_TRANS_64(psas_wx)
GEN_SIMD_TRANS_64(pssa_wx)
GEN_SIMD_TRANS_64(paas_wx)
GEN_SIMD_TRANS_64(pasa_wx)
GEN_SIMD_TRANS_64(predsum_ws)
GEN_SIMD_TRANS_64(predsumu_ws)

GEN_SIMD_TRANS_IMM(pslli_b)
GEN_SIMD_TRANS(psll_bs)
GEN_SIMD_TRANS_IMM(psrli_b)
GEN_SIMD_TRANS(psrl_bs)
GEN_SIMD_TRANS_IMM(psrai_b)
GEN_SIMD_TRANS(psra_bs)
GEN_SIMD_TRANS(pmin_b)
GEN_SIMD_TRANS(pminu_b)
GEN_SIMD_TRANS(pmax_b)
GEN_SIMD_TRANS(pmaxu_b)
GEN_SIMD_TRANS(pmseq_b)
GEN_SIMD_TRANS(pmslt_b)
GEN_SIMD_TRANS(pmsltu_b)

GEN_SIMD_TRANS_R1(psext_h_b)
GEN_SIMD_TRANS_IMM(psati_h)
GEN_SIMD_TRANS_IMM(pusati_h)
GEN_SIMD_TRANS_IMM(pslli_h)
GEN_SIMD_TRANS(psll_hs)
GEN_SIMD_TRANS_IMM(psrli_h)
GEN_SIMD_TRANS(psrl_hs)
GEN_SIMD_TRANS_IMM(psrai_h)
GEN_SIMD_TRANS(psra_hs)
GEN_SIMD_TRANS_IMM(psslai_h)
GEN_SIMD_TRANS_IMM(psrari_h)
GEN_SIMD_TRANS(pssha_hs)
GEN_SIMD_TRANS(psshar_hs)
GEN_SIMD_TRANS(pmin_h)
GEN_SIMD_TRANS(pminu_h)
GEN_SIMD_TRANS(pmax_h)
GEN_SIMD_TRANS(pmaxu_h)
GEN_SIMD_TRANS(pmseq_h)
GEN_SIMD_TRANS(pmslt_h)
GEN_SIMD_TRANS(pmsltu_h)

GEN_SIMD_TRANS_IMM_32(sati_32)
GEN_SIMD_TRANS_IMM_32(usati_32)
GEN_SIMD_TRANS_IMM_32(sslai)
GEN_SIMD_TRANS_IMM_32(srari_32)
GEN_SIMD_TRANS_32(ssha)
GEN_SIMD_TRANS_32(sshar)
GEN_SIMD_TRANS_32(mseq)
GEN_SIMD_TRANS_32(mslt)
GEN_SIMD_TRANS_32(msltu)

GEN_SIMD_TRANS_R1_64(psext_w_b)
GEN_SIMD_TRANS_R1_64(psext_w_h)
GEN_SIMD_TRANS_IMM_64(psati_w)
GEN_SIMD_TRANS_IMM_64(pusati_w)
GEN_SIMD_TRANS_IMM_64(pslli_w)
GEN_SIMD_TRANS_64(psll_ws)
GEN_SIMD_TRANS_IMM_64(psrli_w)
GEN_SIMD_TRANS_64(psrl_ws)
GEN_SIMD_TRANS_IMM_64(psrai_w)
GEN_SIMD_TRANS_64(psra_ws)
GEN_SIMD_TRANS_IMM_64(psslai_w)
GEN_SIMD_TRANS_IMM_64(psrari_w)
GEN_SIMD_TRANS_64(pssha_ws)
GEN_SIMD_TRANS_64(psshar_ws)
GEN_SIMD_TRANS_64(pmin_w)
GEN_SIMD_TRANS_64(pminu_w)
GEN_SIMD_TRANS_64(pmax_w)
GEN_SIMD_TRANS_64(pmaxu_w)
GEN_SIMD_TRANS_64(pmseq_w)
GEN_SIMD_TRANS_64(pmslt_w)
GEN_SIMD_TRANS_64(pmsltu_w)

GEN_SIMD_TRANS_IMM_64(sati_64)
GEN_SIMD_TRANS_IMM_64(usati_64)
GEN_SIMD_TRANS_IMM_64(srari_64)
GEN_SIMD_TRANS_64(sha)
GEN_SIMD_TRANS_64(shar)

GEN_SIMD_TRANS(ppack_h)
GEN_SIMD_TRANS(ppackbt_h)
GEN_SIMD_TRANS(ppacktb_h)
GEN_SIMD_TRANS(ppackt_h)

GEN_SIMD_TRANS_32(packbt_32)
GEN_SIMD_TRANS_32(packtb_32)
GEN_SIMD_TRANS_32(packt_32)

GEN_SIMD_TRANS_64(ppack_w)
GEN_SIMD_TRANS_64(ppackbt_w)
GEN_SIMD_TRANS_64(ppacktb_w)
GEN_SIMD_TRANS_64(ppackt_w)

GEN_SIMD_TRANS_64(packbt_64)
GEN_SIMD_TRANS_64(packtb_64)
GEN_SIMD_TRANS_64(packt_64)

GEN_SIMD_TRANS_64(zip8p)
GEN_SIMD_TRANS_64(zip8hp)
GEN_SIMD_TRANS_64(unzip8p)
GEN_SIMD_TRANS_64(unzip8hp)

GEN_SIMD_TRANS_R1_64(rev16)
GEN_SIMD_TRANS_64(zip16p)
GEN_SIMD_TRANS_64(zip16hp)
GEN_SIMD_TRANS_64(unzip16p)
GEN_SIMD_TRANS_64(unzip16hp)

GEN_SIMD_TRANS_R1(abs)
GEN_SIMD_TRANS_R1(cls)
GEN_SIMD_TRANS_R1(rev)
GEN_SIMD_TRANS_RD(slx)
GEN_SIMD_TRANS_RD(srx)
GEN_SIMD_TRANS_RD(mvm)
GEN_SIMD_TRANS_RD(mvmn)
GEN_SIMD_TRANS_RD(merge)

GEN_SIMD_TRANS_R1_64(absw)
GEN_SIMD_TRANS_R1_64(clsw)

GEN_SIMD_TRANS_RD_P(pwadd_b)
GEN_SIMD_TRANS_RD_RD_P(pwadda_b)
GEN_SIMD_TRANS_RD_P(pwaddu_b)
GEN_SIMD_TRANS_RD_RD_P(pwaddau_b)
GEN_SIMD_TRANS_RD_P(pwsub_b)
GEN_SIMD_TRANS_RD_RD_P(pwsuba_b)
GEN_SIMD_TRANS_RD_P(pwsubu_b)
GEN_SIMD_TRANS_RD_RD_P(pwsubau_b)
GEN_SIMD_TRANS_IMM_RD_P(pwslli_b)
GEN_SIMD_TRANS_RD_P(pwsll_bs)
GEN_SIMD_TRANS_IMM_RD_P(pwslai_b)
GEN_SIMD_TRANS_RD_P(pwsla_bs)

GEN_SIMD_TRANS_RD_P(pwadd_h)
GEN_SIMD_TRANS_RD_RD_P(pwadda_h)
GEN_SIMD_TRANS_RD_P(pwaddu_h)
GEN_SIMD_TRANS_RD_RD_P(pwaddau_h)
GEN_SIMD_TRANS_RD_P(pwsub_h)
GEN_SIMD_TRANS_RD_RD_P(pwsuba_h)
GEN_SIMD_TRANS_RD_P(pwsubu_h)
GEN_SIMD_TRANS_RD_RD_P(pwsubau_h)
GEN_SIMD_TRANS_IMM_RD_P(pwslli_h)
GEN_SIMD_TRANS_RD_P(pwsll_hs)
GEN_SIMD_TRANS_IMM_RD_P(pwslai_h)
GEN_SIMD_TRANS_RD_P(pwsla_hs)

GEN_SIMD_TRANS_RD_P(wadd)
GEN_SIMD_TRANS_RD_RD_P(wadda)
GEN_SIMD_TRANS_RD_P(waddu)
GEN_SIMD_TRANS_RD_RD_P(waddau)
GEN_SIMD_TRANS_RD_P(wsub)
GEN_SIMD_TRANS_RD_RD_P(wsuba)
GEN_SIMD_TRANS_RD_P(wsubu)
GEN_SIMD_TRANS_RD_RD_P(wsubau)

GEN_SIMD_TRANS_IMM_RD_P(wslli)
GEN_SIMD_TRANS_RD_P(wsll)
GEN_SIMD_TRANS_IMM_RD_P(wslai)
GEN_SIMD_TRANS_RD_P(wsla)

GEN_SIMD_TRANS_DX_P(padd, d, b)
GEN_SIMD_TRANS_DX_P(psub, d, b)
GEN_SIMD_TRANS_DX_P(psadd, d, b)
GEN_SIMD_TRANS_DX_P(psaddu, d, b)
GEN_SIMD_TRANS_DX_P(pssub, d, b)
GEN_SIMD_TRANS_DX_P(pssubu, d, b)
GEN_SIMD_TRANS_DX_P(paadd, d, b)
GEN_SIMD_TRANS_DX_P(paaddu, d, b)
GEN_SIMD_TRANS_DX_P(pasub, d, b)
GEN_SIMD_TRANS_DX_P(pasubu, d, b)
GEN_SIMD_TRANS_DX_P(pdif, d, b)
GEN_SIMD_TRANS_DX_P(pdifu, d, b)
GEN_SIMD_TRANS_DX_R1_P(psabs, d, b)
GEN_SIMD_TRANS_DX_P(padd, d, h)
GEN_SIMD_TRANS_DX_P(psub, d, h)
GEN_SIMD_TRANS_DX_P(psadd, d, h)
GEN_SIMD_TRANS_DX_P(psaddu, d, h)
GEN_SIMD_TRANS_DX_P(pssub, d, h)
GEN_SIMD_TRANS_DX_P(pssubu, d, h)
GEN_SIMD_TRANS_DX_P(paadd, d, h)
GEN_SIMD_TRANS_DX_P(paaddu, d, h)
GEN_SIMD_TRANS_DX_P(pasub, d, h)
GEN_SIMD_TRANS_DX_P(pasubu, d, h)
GEN_SIMD_TRANS_DX_P(psh1add, d, h)
GEN_SIMD_TRANS_DX_P(pssh1sadd, d, h)
GEN_SIMD_TRANS_DX_P(pas, d, hx)
GEN_SIMD_TRANS_DX_P(psa, d, hx)
GEN_SIMD_TRANS_DX_P(psas, d, hx)
GEN_SIMD_TRANS_DX_P(pssa, d, hx)
GEN_SIMD_TRANS_DX_P(paas, d, hx)
GEN_SIMD_TRANS_DX_P(pasa, d, hx)
GEN_SIMD_TRANS_DX_P(pdif, d, h)
GEN_SIMD_TRANS_DX_P(pdifu, d, h)
GEN_SIMD_TRANS_DX_R1_P(psabs, d, h)
GEN_SIMD_TRANS_DW_P(sadd)
GEN_SIMD_TRANS_DW_P(saddu)
GEN_SIMD_TRANS_DW_P(ssub)
GEN_SIMD_TRANS_DW_P(ssubu)
GEN_SIMD_TRANS_DW_P(aadd)
GEN_SIMD_TRANS_DW_P(aaddu)
GEN_SIMD_TRANS_DW_P(asub)
GEN_SIMD_TRANS_DW_P(asubu)
GEN_SIMD_TRANS_DW_P(ssh1sadd)

GEN_SIMD_TRANS_IMM_32_P(pslli, d, b)
GEN_SIMD_TRANS_IMM_32_P(psrli, d, b)
GEN_SIMD_TRANS_IMM_32_P(psrai, d, b)
GEN_SIMD_TRANS_DX_P(pmin, d, b)
GEN_SIMD_TRANS_DX_P(pminu, d, b)
GEN_SIMD_TRANS_DX_P(pmax, d, b)
GEN_SIMD_TRANS_DX_P(pmaxu, d, b)
GEN_SIMD_TRANS_DX_P(pmseq, d, b)
GEN_SIMD_TRANS_DX_P(pmslt, d, b)
GEN_SIMD_TRANS_DX_P(pmsltu, d, b)
GEN_SIMD_TRANS_R1_P(psext, d, h_b)
GEN_SIMD_TRANS_IMM_32_P(psati, d, h)
GEN_SIMD_TRANS_IMM_32_P(pusati, d, h)
GEN_SIMD_TRANS_IMM_32_P(pslli, d, h)
GEN_SIMD_TRANS_IMM_32_P(psrli, d, h)
GEN_SIMD_TRANS_IMM_32_P(psrai, d, h)
GEN_SIMD_TRANS_IMM_32_P(psslai, d, h)
GEN_SIMD_TRANS_IMM_32_P(psrari, d, h)
GEN_SIMD_TRANS_DX_P(pmin, d, h)
GEN_SIMD_TRANS_DX_P(pminu, d, h)
GEN_SIMD_TRANS_DX_P(pmax, d, h)
GEN_SIMD_TRANS_DX_P(pmaxu, d, h)
GEN_SIMD_TRANS_DX_P(pmseq, d, h)
GEN_SIMD_TRANS_DX_P(pmslt, d, h)
GEN_SIMD_TRANS_DX_P(pmsltu, d, h)
GEN_SIMD_TRANS_R1_P(psext, d, w_b)
GEN_SIMD_TRANS_R1_P(psext, d, w_h)
GEN_SIMD_TRANS_IMM_32_P(psati, d, w)
GEN_SIMD_TRANS_IMM_32_P(pusati, d, w)
GEN_SIMD_TRANS_IMM_32_P(pslli, d, w)
GEN_SIMD_TRANS_IMM_32_P(psrli, d, w)
GEN_SIMD_TRANS_IMM_32_P(psrai, d, w)
GEN_SIMD_TRANS_IMM_32_P(psslai, d, w)
GEN_SIMD_TRANS_IMM_32_P(psrari, d, w)
//#pmin_dw
//#pminu_dw
//#pmax_dw
//#pmaxu_dw
GEN_SIMD_TRANS_DX_P(pmseq, d, w)
GEN_SIMD_TRANS_DX_P(pmslt, d, w)
GEN_SIMD_TRANS_DX_P(pmsltu, d, w)

GEN_SIMD_TRANS_DXS_P(padd, d, bs)
GEN_SIMD_TRANS_DXS_P(psll, d, bs)
GEN_SIMD_TRANS_DXS_P(psra, d, bs)
GEN_SIMD_TRANS_DXS_P(padd, d, hs)
GEN_SIMD_TRANS_DXS_P(psll, d, hs)
GEN_SIMD_TRANS_DXS_P(psrl, d, hs)
GEN_SIMD_TRANS_DXS_P(psra, d, hs)
GEN_SIMD_TRANS_DXS_P(pssha, d, hs)
GEN_SIMD_TRANS_DXS_P(psshar, d, hs)
GEN_SIMD_TRANS_DXS_P(padd, d, ws)
GEN_SIMD_TRANS_DXS_P(psll, d, ws)
GEN_SIMD_TRANS_DXS_P(psrl, d, ws)
GEN_SIMD_TRANS_DXS_P(psra, d, ws)
GEN_SIMD_TRANS_DXS_P(pssha, d, ws)
GEN_SIMD_TRANS_DXS_P(psshar, d, ws)

GEN_SIMD_TRANS_DX_P(ppack, d, h)
GEN_SIMD_TRANS_DX_P(ppackbt, d, h)
//GEN_SIMD_TRANS_DX_P(ppacktb, d, h)
GEN_SIMD_TRANS_DX_P(ppacktb, d, h)
GEN_SIMD_TRANS_DX_P(ppack, d, w)
GEN_SIMD_TRANS_DX_P(ppackbt, d, w)
GEN_SIMD_TRANS_DX_P(ppacktb, d, w)
GEN_SIMD_TRANS_DX_P(ppackt, d, w)

static bool trans_padd_dw(DisasContext *ctx, arg_padd_dw * a)
{
    REQUIRE_32BIT(ctx);
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);
    TCGv src2_0 = get_gpr(ctx, (a->rs2)*2, EXT_NONE);
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);
    TCGv src2_1 = get_gpr(ctx, (a->rs2)*2+1, EXT_NONE);
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);
    tcg_gen_add_tl(dest_0, src1_0, src2_0);
    tcg_gen_add_tl(dest_1, src1_1, src2_1);
    gen_set_gpr(ctx, (a->rd)*2, dest_0);
    gen_set_gpr(ctx, (a->rd)*2+1, dest_1);
    return true;
}

static bool trans_psub_dw(DisasContext *ctx, arg_psub_dw * a)
{
    REQUIRE_32BIT(ctx);
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);
    TCGv src2_0 = get_gpr(ctx, (a->rs2)*2, EXT_NONE);
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);
    TCGv src2_1 = get_gpr(ctx, (a->rs2)*2+1, EXT_NONE);
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);
    tcg_gen_sub_tl(dest_0, src1_0, src2_0);
    tcg_gen_sub_tl(dest_1, src1_1, src2_1);
    gen_set_gpr(ctx, (a->rd)*2, dest_0);
    gen_set_gpr(ctx, (a->rd)*2+1, dest_1);
    return true;
}

static bool trans_psh1add_dw(DisasContext *ctx, arg_psh1add_dw * a)
{
    REQUIRE_32BIT(ctx);
    TCGv src1_0 = get_gpr(ctx, (a->rs1)*2, EXT_NONE);
    TCGv src2_0 = get_gpr(ctx, (a->rs2)*2, EXT_NONE);
    TCGv dest_0 = dest_gpr(ctx, (a->rd)*2);
    TCGv src1_1 = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);
    TCGv src2_1 = get_gpr(ctx, (a->rs2)*2+1, EXT_NONE);
    TCGv dest_1 = dest_gpr(ctx, (a->rd)*2+1);
    gen_sh1add(dest_0, src1_0, src2_0);
    gen_sh1add(dest_1, src1_1, src2_1);
    gen_set_gpr(ctx, (a->rd)*2, dest_0);
    gen_set_gpr(ctx, (a->rd)*2+1, dest_1);
    return true;
}

/* wzip8p and wzip16p need to verify rd is not zero register in trans functions */
static bool trans_wzip8p(DisasContext *ctx, arg_wzip8p * a)
{
    REQUIRE_32BIT(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_64 = tcg_temp_new_i64();
    tcg_gen_extu_tl_i64(rs1_64, src1);
    tcg_gen_extu_tl_i64(rs2_64, src2);
    TCGv_i64 t = tcg_temp_new_i64();
    if(a->rd == 0){
        return true;
    }else{
        get_pair_regs(ctx, t, (a->rd)*2);
    }
    gen_helper_wzip8p(t, tcg_env, rs1_64, rs2_64);
    set_pair_regs(ctx, (a->rd)*2, t);
    return true;
}

static bool trans_wzip16p(DisasContext *ctx, arg_wzip16p * a)
{
    REQUIRE_32BIT(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_64 = tcg_temp_new_i64();
    tcg_gen_extu_tl_i64(rs1_64, src1);
    tcg_gen_extu_tl_i64(rs2_64, src2);
    TCGv_i64 t = tcg_temp_new_i64();
    if(a->rd == 0){
        return true;
    }else{
        get_pair_regs(ctx, t, (a->rd)*2);
    }
    gen_helper_wzip16p(t, tcg_env, rs1_64, rs2_64);
    set_pair_regs(ctx, (a->rd)*2, t);
    return true;
}

static bool trans_pli_b(DisasContext *ctx, arg_pli_b *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE){
        imm = (( imm << 8 ) + a->imm);
	    i++;
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}

static bool trans_pli_h(DisasContext *ctx, arg_pli_h *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE / 2){
        imm =( imm << 16 ) + (a->imm & 0xFFFF);
	    i++;
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}

static bool trans_plui_h(DisasContext *ctx, arg_plui_h *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE / 2){
        imm =( imm << 16 ) + (a->imm & 0xFFFF);
	    i++;
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}

static bool trans_pli_w(DisasContext *ctx, arg_pli_w *a)
{
    REQUIRE_64BIT(ctx);
    int i = 1;
    int64_t imm = a->imm;
    while (i < TARGET_LONG_SIZE / 4){
        imm =( imm << 32 ) + (a->imm & 0xFFFFFFFF);
	    i++;
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}

static bool trans_plui_w(DisasContext *ctx, arg_plui_w *a)
{
    REQUIRE_64BIT(ctx);
    int i = 1;
    int64_t imm = a->imm;
    while (i < TARGET_LONG_SIZE / 4){
        imm =( imm << 32 ) + (a->imm & 0xFFFFFFFF);
	    i++;
    }
    gen_set_gpri(ctx, a->rd, imm);
    return true;
}

static bool trans_pli_db(DisasContext *ctx, arg_pli_db *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE){
        imm = (( imm << 8 ) + a->imm);
	    i++;
    }
    gen_set_gpri(ctx, (a->rd)*2, imm);
    gen_set_gpri(ctx, (a->rd)*2+1, imm);
    return true;
}

static bool trans_pli_dh(DisasContext *ctx, arg_pli_dh *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE / 2){
        imm =( imm << 16 ) + (a->imm & 0xFFFF);
	    i++;
    }
    gen_set_gpri(ctx, (a->rd)*2, imm);
    gen_set_gpri(ctx, (a->rd)*2+1, imm);
    return true;
}

static bool trans_plui_dh(DisasContext *ctx, arg_plui_dh *a)
{
    int i = 1;
    target_long imm = a->imm;
    while (i < TARGET_LONG_SIZE / 2){
        imm =( imm << 16 ) + (a->imm & 0xFFFF);
	    i++;
    }
    gen_set_gpri(ctx, (a->rd)*2, imm);
    gen_set_gpri(ctx, (a->rd)*2+1, imm);
    return true;
}

static bool trans_addd_p(DisasContext *ctx, arg_addd_p * a)
{
    REQUIRE_32BIT(ctx);
    TCGv_i64 src1 = tcg_temp_new_i64();
    TCGv_i64 src2 = tcg_temp_new_i64();
    TCGv_i64 dest = tcg_temp_new_i64();
    get_pair_regs(ctx, src1, (a->rs1)*2);
    get_pair_regs(ctx, src2, (a->rs2)*2);
    get_pair_regs(ctx, dest, (a->rd)*2);
    //gen_helper_add(dest, tcg_env, src1, src2);
    tcg_gen_add_i64(dest, src1, src2);
    set_pair_regs(ctx, (a->rd)*2, dest);

    return true; 
}

static bool trans_subd_p(DisasContext *ctx, arg_subd_p * a)
{
    REQUIRE_32BIT(ctx);
    TCGv_i64 src1 = tcg_temp_new_i64();
    TCGv_i64 src2 = tcg_temp_new_i64();
    TCGv_i64 dest = tcg_temp_new_i64();
    get_pair_regs(ctx, src1, (a->rs1)*2);
    get_pair_regs(ctx, src2, (a->rs2)*2);
    get_pair_regs(ctx, dest, (a->rd)*2);
    //gen_helper_sub(dest, tcg_env, src1, src2);
    tcg_gen_sub_i64(dest, src1, src2);
    set_pair_regs(ctx, (a->rd)*2, dest);

    return true; 
}

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_PREDSUM_DX_P(NAME)                   \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    TCGv_i32 src1_l;                                        \
    TCGv_i32 src1_h;                                        \
    TCGv_i32 src2 = get_gpr(ctx, a->rs2, EXT_NONE);         \
    TCGv_i32 dest = dest_gpr(ctx, a->rd);                   \
    if(a->rs1 == 0){                                        \
        src1_l = tcg_temp_new_i32();                        \
        src1_h = tcg_temp_new_i32();                        \
        tcg_gen_movi_i32(src1_l, 0);                        \
        tcg_gen_movi_i32(src1_h, 0);                        \
    }else{                                                  \
        src1_l = get_gpr(ctx, (a->rs1)*2, EXT_NONE);        \
        src1_h = get_gpr(ctx, (a->rs1)*2+1, EXT_NONE);      \
    }                                                       \
    gen_helper_##NAME(dest, tcg_env, src1_l, src1_h, src2); \
    return true;                                            \
}
#else
#define GEN_SIMD_TRANS_PREDSUM_DX_P(NAME)                   \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME * a) \
{                                                           \
    REQUIRE_32BIT(ctx);                                     \
    return true;                                            \
}
#endif

GEN_SIMD_TRANS_PREDSUM_DX_P(predsum_dbs)
GEN_SIMD_TRANS_PREDSUM_DX_P(predsumu_dbs)
GEN_SIMD_TRANS_PREDSUM_DX_P(predsum_dhs)
GEN_SIMD_TRANS_PREDSUM_DX_P(predsumu_dhs)

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_PN_OP_IMM(NAME)                     \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a) \
{                                                          \
    REQUIRE_32BIT(ctx);                                    \
    TCGv_i64 s1 = tcg_temp_new_i64();                      \
    if (a->rs1 == 0) {                                     \
        tcg_gen_mov_i64(s1, 0);                            \
    } else {                                               \
        get_pair_regs(ctx, s1, a->rs1 * 2);                \
    }                                                      \
    TCGv shamt = tcg_constant_tl(a->imm);                  \
    TCGv_i32 dest = dest_gpr(ctx, a->rd);                  \
    gen_helper_##NAME(dest, tcg_env, s1, shamt);           \
    return true;                                           \
}
#else
#define GEN_SIMD_TRANS_PN_OP_IMM(NAME)                     \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a) \
{                                                          \
    REQUIRE_32BIT(ctx);                                    \
    return true;                                           \
}
#endif

GEN_SIMD_TRANS_PN_OP_IMM(pnsrli_b)
GEN_SIMD_TRANS_PN_OP_IMM(pnsrai_b)
GEN_SIMD_TRANS_PN_OP_IMM(pnsrari_b)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipi_b)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipri_b)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipiu_b)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipriu_b)

GEN_SIMD_TRANS_PN_OP_IMM(pnsrli_h)
GEN_SIMD_TRANS_PN_OP_IMM(pnsrai_h)
GEN_SIMD_TRANS_PN_OP_IMM(pnsrari_h)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipi_h)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipri_h)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipiu_h)
GEN_SIMD_TRANS_PN_OP_IMM(pnclipriu_h)

GEN_SIMD_TRANS_PN_OP_IMM(nsrli)
GEN_SIMD_TRANS_PN_OP_IMM(nsrai)
GEN_SIMD_TRANS_PN_OP_IMM(nsrari)
GEN_SIMD_TRANS_PN_OP_IMM(nclipi)
GEN_SIMD_TRANS_PN_OP_IMM(nclipri)
GEN_SIMD_TRANS_PN_OP_IMM(nclipiu)
GEN_SIMD_TRANS_PN_OP_IMM(nclipriu)

#if defined(TARGET_RISCV32)
#define GEN_SIMD_TRANS_PN_OP_REG(NAME)                     \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a) \
{                                                          \
    REQUIRE_32BIT(ctx);                                    \
    TCGv_i64 s1 = tcg_temp_new_i64();                      \
    if (a->rs1 == 0) {                                     \
        tcg_gen_mov_i64(s1, 0);                            \
    } else {                                               \
        get_pair_regs(ctx, s1, a->rs1 * 2);                \
    }                                                      \
    TCGv_i32 shamt = tcg_temp_new_i32();                   \
    TCGv_i32 rs2 = get_gpr(ctx, a->rs2, EXT_NONE);         \
    TCGv_i32 dest = dest_gpr(ctx, a->rd);                  \
    gen_helper_##NAME(dest, tcg_env, s1, rs2);             \
    return true;                                           \
}
#else
#define GEN_SIMD_TRANS_PN_OP_REG(NAME)                     \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a) \
{                                                          \
    REQUIRE_32BIT(ctx);                                    \
    return true;                                           \
}
#endif

GEN_SIMD_TRANS_PN_OP_REG(pnsrl_bs)
GEN_SIMD_TRANS_PN_OP_REG(pnsra_bs)
GEN_SIMD_TRANS_PN_OP_REG(pnsrar_bs)
GEN_SIMD_TRANS_PN_OP_REG(pnclip_bs)
GEN_SIMD_TRANS_PN_OP_REG(pnclipr_bs)
GEN_SIMD_TRANS_PN_OP_REG(pnclipu_bs)
GEN_SIMD_TRANS_PN_OP_REG(pnclipru_bs)

GEN_SIMD_TRANS_PN_OP_REG(pnsrl_hs)
GEN_SIMD_TRANS_PN_OP_REG(pnsra_hs)
GEN_SIMD_TRANS_PN_OP_REG(pnsrar_hs)
GEN_SIMD_TRANS_PN_OP_REG(pnclip_hs)
GEN_SIMD_TRANS_PN_OP_REG(pnclipr_hs)
GEN_SIMD_TRANS_PN_OP_REG(pnclipu_hs)
GEN_SIMD_TRANS_PN_OP_REG(pnclipru_hs)

GEN_SIMD_TRANS_PN_OP_REG(nsrl)
GEN_SIMD_TRANS_PN_OP_REG(nsra)
GEN_SIMD_TRANS_PN_OP_REG(nsrar)
GEN_SIMD_TRANS_PN_OP_REG(nclip)
GEN_SIMD_TRANS_PN_OP_REG(nclipr)
GEN_SIMD_TRANS_PN_OP_REG(nclipu)
GEN_SIMD_TRANS_PN_OP_REG(nclipru)

GEN_SIMD_TRANS(pmulh_h)
GEN_SIMD_TRANS(pmulhsu_h)
GEN_SIMD_TRANS(pmulhu_h)
GEN_SIMD_TRANS(pmulq_h)
GEN_SIMD_TRANS(pmulhr_h)
GEN_SIMD_TRANS(pmulhrsu_h)
GEN_SIMD_TRANS(pmulhru_h)
GEN_SIMD_TRANS(pmulqr_h)

GEN_SIMD_TRANS_RD(pmhacc_h)
GEN_SIMD_TRANS_RD(pmhaccsu_h)
GEN_SIMD_TRANS_RD(pmhaccu_h)
GEN_SIMD_TRANS_RD(pmhracc_h)
GEN_SIMD_TRANS_RD(pmhraccsu_h)
GEN_SIMD_TRANS_RD(pmhraccu_h)

GEN_SIMD_TRANS_32(mulhr)
GEN_SIMD_TRANS_32(mulhrsu)
GEN_SIMD_TRANS_32(mulhru)
GEN_SIMD_TRANS_32(mulq)
GEN_SIMD_TRANS_32(mulqr)

GEN_SIMD_TRANS_64(pmulhr_w)
GEN_SIMD_TRANS_64(pmulhrsu_w)
GEN_SIMD_TRANS_64(pmulhru_w)
GEN_SIMD_TRANS_64(pmulq_w)
GEN_SIMD_TRANS_64(pmulqr_w)
GEN_SIMD_TRANS_64(pmulh_w)
GEN_SIMD_TRANS_64(pmulhsu_w)
GEN_SIMD_TRANS_64(pmulhu_w)

GEN_SIMD_TRANS_RD_32(mhacc)
GEN_SIMD_TRANS_RD_32(mhracc)
GEN_SIMD_TRANS_RD_32(mhaccsu)
GEN_SIMD_TRANS_RD_32(mhraccsu)
GEN_SIMD_TRANS_RD_32(mhaccu)
GEN_SIMD_TRANS_RD_32(mhraccu)

GEN_SIMD_TRANS_RD_64(pmhacc_w)
GEN_SIMD_TRANS_RD_64(pmhracc_w)
GEN_SIMD_TRANS_RD_64(pmhaccsu_w)
GEN_SIMD_TRANS_RD_64(pmhraccsu_w)
GEN_SIMD_TRANS_RD_64(pmhaccu_w)
GEN_SIMD_TRANS_RD_64(pmhraccu_w)

GEN_SIMD_TRANS_RD_32(mqacc_h00)
GEN_SIMD_TRANS_RD_32(mqacc_h01)
GEN_SIMD_TRANS_RD_32(mqacc_h11)
GEN_SIMD_TRANS_RD_32(mqracc_h00)
GEN_SIMD_TRANS_RD_32(mqracc_h01)
GEN_SIMD_TRANS_RD_32(mqracc_h11)

GEN_SIMD_TRANS_RD_64(pmqacc_w_h00)
GEN_SIMD_TRANS_RD_64(pmqacc_w_h01)
GEN_SIMD_TRANS_RD_64(pmqacc_w_h11)
GEN_SIMD_TRANS_RD_64(pmqracc_w_h00)
GEN_SIMD_TRANS_RD_64(pmqracc_w_h01)
GEN_SIMD_TRANS_RD_64(pmqracc_w_h11)

GEN_SIMD_TRANS(pmq2add_h)
GEN_SIMD_TRANS(pmqr2add_h)
GEN_SIMD_TRANS_RD(pmq2adda_h)
GEN_SIMD_TRANS_RD(pmqr2adda_h)

GEN_SIMD_TRANS_RD_64(mqacc_w00)
GEN_SIMD_TRANS_RD_64(mqacc_w01)
GEN_SIMD_TRANS_RD_64(mqacc_w11)
GEN_SIMD_TRANS_RD_64(mqracc_w00)
GEN_SIMD_TRANS_RD_64(mqracc_w01)
GEN_SIMD_TRANS_RD_64(mqracc_w11)

GEN_SIMD_TRANS_64(pmq2add_w)
GEN_SIMD_TRANS_64(pmqr2add_w)
GEN_SIMD_TRANS_RD_64(pmq2adda_w)
GEN_SIMD_TRANS_RD_64(pmqr2adda_w)

GEN_SIMD_TRANS(pmul_h_b00)
GEN_SIMD_TRANS(pmul_h_b01)
GEN_SIMD_TRANS(pmul_h_b11)
GEN_SIMD_TRANS(pmulsu_h_b00)
GEN_SIMD_TRANS(pmulsu_h_b11)
GEN_SIMD_TRANS(pmulu_h_b00)
GEN_SIMD_TRANS(pmulu_h_b01)
GEN_SIMD_TRANS(pmulu_h_b11)

GEN_SIMD_TRANS_32(mul_h00)
GEN_SIMD_TRANS_32(mul_h01)
GEN_SIMD_TRANS_32(mul_h11)
GEN_SIMD_TRANS_32(mulsu_h00)
GEN_SIMD_TRANS_32(mulsu_h11)
GEN_SIMD_TRANS_32(mulu_h00)
GEN_SIMD_TRANS_32(mulu_h01)
GEN_SIMD_TRANS_32(mulu_h11)

GEN_SIMD_TRANS_64(pmul_w_h00)
GEN_SIMD_TRANS_64(pmul_w_h01)
GEN_SIMD_TRANS_64(pmul_w_h11)
GEN_SIMD_TRANS_64(pmulsu_w_h00)
GEN_SIMD_TRANS_64(pmulsu_w_h11)
GEN_SIMD_TRANS_64(pmulu_w_h00)
GEN_SIMD_TRANS_64(pmulu_w_h01)
GEN_SIMD_TRANS_64(pmulu_w_h11)

GEN_SIMD_TRANS_RD_32(macc_h00)
GEN_SIMD_TRANS_RD_32(macc_h01)
GEN_SIMD_TRANS_RD_32(macc_h11)
GEN_SIMD_TRANS_RD_32(maccsu_h00)
GEN_SIMD_TRANS_RD_32(maccsu_h11)
GEN_SIMD_TRANS_RD_32(maccu_h00)
GEN_SIMD_TRANS_RD_32(maccu_h01)
GEN_SIMD_TRANS_RD_32(maccu_h11)

GEN_SIMD_TRANS_RD_64(pmacc_w_h00)
GEN_SIMD_TRANS_RD_64(pmacc_w_h01)
GEN_SIMD_TRANS_RD_64(pmacc_w_h11)
GEN_SIMD_TRANS_RD_64(pmaccsu_w_h00)
GEN_SIMD_TRANS_RD_64(pmaccsu_w_h11)
GEN_SIMD_TRANS_RD_64(pmaccu_w_h00)
GEN_SIMD_TRANS_RD_64(pmaccu_w_h01)
GEN_SIMD_TRANS_RD_64(pmaccu_w_h11)

GEN_SIMD_TRANS(pm2add_h)
GEN_SIMD_TRANS(pm2addsu_h)
GEN_SIMD_TRANS(pm2addu_h)
GEN_SIMD_TRANS(pm2add_hx)
GEN_SIMD_TRANS(pm2sadd_h)
GEN_SIMD_TRANS(pm2sadd_hx)
GEN_SIMD_TRANS(pm2sub_h)
GEN_SIMD_TRANS(pm2sub_hx)

GEN_SIMD_TRANS_RD(pm2adda_h)
GEN_SIMD_TRANS_RD(pm2addasu_h)
GEN_SIMD_TRANS_RD(pm2addau_h)
GEN_SIMD_TRANS_RD(pm2adda_hx)
GEN_SIMD_TRANS_RD(pm2suba_h)
GEN_SIMD_TRANS_RD(pm2suba_hx)

GEN_SIMD_TRANS_64(mul_w00)
GEN_SIMD_TRANS_64(mul_w01)
GEN_SIMD_TRANS_64(mul_w11)
GEN_SIMD_TRANS_64(mulsu_w00)
GEN_SIMD_TRANS_64(mulsu_w11)
GEN_SIMD_TRANS_64(mulu_w00)
GEN_SIMD_TRANS_64(mulu_w01)
GEN_SIMD_TRANS_64(mulu_w11)

GEN_SIMD_TRANS_RD_64(macc_w00)
GEN_SIMD_TRANS_RD_64(macc_w01)
GEN_SIMD_TRANS_RD_64(macc_w11)
GEN_SIMD_TRANS_RD_64(maccsu_w00)
GEN_SIMD_TRANS_RD_64(maccsu_w11)
GEN_SIMD_TRANS_RD_64(maccu_w00)
GEN_SIMD_TRANS_RD_64(maccu_w01)
GEN_SIMD_TRANS_RD_64(maccu_w11)

GEN_SIMD_TRANS_64(pm2add_w)
GEN_SIMD_TRANS_64(pm2addsu_w)
GEN_SIMD_TRANS_64(pm2addu_w)
GEN_SIMD_TRANS_64(pm2add_wx)
GEN_SIMD_TRANS_64(pm2sub_w)
GEN_SIMD_TRANS_64(pm2sub_wx)

GEN_SIMD_TRANS_RD_64(pm2adda_w)
GEN_SIMD_TRANS_RD_64(pm2addasu_w)
GEN_SIMD_TRANS_RD_64(pm2addau_w)
GEN_SIMD_TRANS_RD_64(pm2adda_wx)
GEN_SIMD_TRANS_RD_64(pm2suba_w)
GEN_SIMD_TRANS_RD_64(pm2suba_wx)

GEN_SIMD_TRANS(pm4add_b)
GEN_SIMD_TRANS(pm4addsu_b)
GEN_SIMD_TRANS(pm4addu_b)

GEN_SIMD_TRANS_RD(pm4adda_b)
GEN_SIMD_TRANS_RD(pm4addasu_b)
GEN_SIMD_TRANS_RD(pm4addau_b)

GEN_SIMD_TRANS_64(pm4add_h)
GEN_SIMD_TRANS_64(pm4addsu_h)
GEN_SIMD_TRANS_64(pm4addu_h)

GEN_SIMD_TRANS_RD_64(pm4adda_h)
GEN_SIMD_TRANS_RD_64(pm4addasu_h)
GEN_SIMD_TRANS_RD_64(pm4addau_h)

GEN_SIMD_TRANS(pmulh_h_b0)
GEN_SIMD_TRANS(pmulh_h_b1)
GEN_SIMD_TRANS(pmulhsu_h_b0)
GEN_SIMD_TRANS(pmulhsu_h_b1)

GEN_SIMD_TRANS_RD(pmhacc_h_b0)
GEN_SIMD_TRANS_RD(pmhacc_h_b1)
GEN_SIMD_TRANS_RD(pmhaccsu_h_b0)
GEN_SIMD_TRANS_RD(pmhaccsu_h_b1)

GEN_SIMD_TRANS_32(mulh_h0)
GEN_SIMD_TRANS_32(mulh_h1)
GEN_SIMD_TRANS_32(mulhsu_h0)
GEN_SIMD_TRANS_32(mulhsu_h1)

GEN_SIMD_TRANS_64(pmulh_w_h0)
GEN_SIMD_TRANS_64(pmulh_w_h1)
GEN_SIMD_TRANS_64(pmulhsu_w_h0)
GEN_SIMD_TRANS_64(pmulhsu_w_h1)

GEN_SIMD_TRANS_RD_32(mhacc_h0)
GEN_SIMD_TRANS_RD_32(mhacc_h1)
GEN_SIMD_TRANS_RD_32(mhaccsu_h0)
GEN_SIMD_TRANS_RD_32(mhaccsu_h1)

GEN_SIMD_TRANS_RD_64(pmhacc_w_h0)
GEN_SIMD_TRANS_RD_64(pmhacc_w_h1)
GEN_SIMD_TRANS_RD_64(pmhaccsu_w_h0)
GEN_SIMD_TRANS_RD_64(pmhaccsu_w_h1)

GEN_SIMD_TRANS_RD_RD_P(pmqwacc_h)
GEN_SIMD_TRANS_RD_RD_P(pmqrwacc_h)
GEN_SIMD_TRANS_RD_RD_P(mqwacc)
GEN_SIMD_TRANS_RD_RD_P(mqrwacc)

GEN_SIMD_TRANS_RD_P(pwmul_b)
GEN_SIMD_TRANS_RD_P(pwmulsu_b)
GEN_SIMD_TRANS_RD_P(pwmulu_b)

GEN_SIMD_TRANS_RD_P(pwmul_h)
GEN_SIMD_TRANS_RD_P(pwmulsu_h)
GEN_SIMD_TRANS_RD_P(pwmulu_h)
GEN_SIMD_TRANS_RD_RD_P(pwmacc_h)
GEN_SIMD_TRANS_RD_RD_P(pwmaccsu_h)
GEN_SIMD_TRANS_RD_RD_P(pwmaccu_h)

GEN_SIMD_TRANS_RD_P(wmul)
GEN_SIMD_TRANS_RD_P(wmulsu)
GEN_SIMD_TRANS_RD_P(wmulu)

GEN_SIMD_TRANS_RD_RD_P(wmacc)
GEN_SIMD_TRANS_RD_RD_P(wmaccsu)
GEN_SIMD_TRANS_RD_RD_P(wmaccu)

GEN_SIMD_TRANS_RD_P(pm2wadd_h)
GEN_SIMD_TRANS_RD_P(pm2waddsu_h)
GEN_SIMD_TRANS_RD_P(pm2waddu_h)
GEN_SIMD_TRANS_RD_P(pm2wadd_hx)

GEN_SIMD_TRANS_RD_RD_P(pm2wadda_h)
GEN_SIMD_TRANS_RD_RD_P(pm2waddasu_h)
GEN_SIMD_TRANS_RD_RD_P(pm2waddau_h)
GEN_SIMD_TRANS_RD_RD_P(pm2wadda_hx)

GEN_SIMD_TRANS_RD_P(pm2wsub_h)
GEN_SIMD_TRANS_RD_P(pm2wsub_hx)
GEN_SIMD_TRANS_RD_RD_P(pm2wsuba_h)
GEN_SIMD_TRANS_RD_RD_P(pm2wsuba_hx)

