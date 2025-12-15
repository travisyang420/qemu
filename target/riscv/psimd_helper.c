/*
 * RISC-V Packed SIMD Extension Helpers for QEMU.
 *
 * Copyright (C) 2024 PLCT Lab.
 * Written by Codethink Ltd and SiFive.
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

#include "qemu/osdep.h"
#include "cpu.h"
#include "qemu/host-utils.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include "internals.h"

static int64_t signed_saturate(CPURISCVState *env, int64_t arg, int range)
{
    int64_t max = (1ull << (range - 1)) - 1;
    int64_t min = -(1 << (range - 1));
    if (arg > max) {
        arg = max;
        env->vxsat = 0x1;
    } else if (arg < min) {
        arg = min;
        env->vxsat = 0x1;
    }

    return arg;
}

/*
static int64_t signed_saturate64(CPURISCVState *env, Int128 arg)
{
    Int128 max = INT64_MAX;
    Int128 min = INT64_MIN;

    if (arg > max) {
        arg = max;
        env->vxsat = 0x1;
    } else if (arg < min) {
        arg = min;
        env->vxsat = 0x1;
    }
                                                                                                                                                                                                      return arg;
}
*/

static uint64_t unsigned_saturate(CPURISCVState *env, uint64_t arg, int range)
{
    uint64_t max = (1ull << range) - 1;

    if (arg > max) {
        arg = max;
        env->vxsat = 0x1;
    }
                                                                                                                                                                                                      return arg;
}

/*
static uint64_t unsigned_saturate64(CPURISCVState *env, __uint128_t arg)
{
    __uint128_t max = UINT64_MAX; 

    if (arg > max){
        arg = max;
        env->vxsat = 0x1;
    }

    return arg;
}
*/

target_ulong HELPER(padd_h)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE / 2; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(padd_hs)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = rs2_p[0];

    for(int i=0; i < TARGET_LONG_SIZE / 2; i++){
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(padd_b)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(padd_bs)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = rs2_p[0];

    for(int i=0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(psub_b)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(psub_h)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i=0; i < TARGET_LONG_SIZE / 2; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(psadd_b)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i =0; i < TARGET_LONG_SIZE; i++){
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)signed_saturate(env, v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(psadd_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(psaddu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)unsigned_saturate(env, v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(psaddu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(pssub_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)signed_saturate(env, v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(pssub_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(pssubu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)unsigned_saturate(env, v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(pssubu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(paadd_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(paadd_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(paaddu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(paaddu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasub_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasub_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasubu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(pasubu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(psabs_b)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];

        if (v1 == INT8_MIN) {
            v1 = INT8_MAX;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = -v1;
        }

        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(pdifsumu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    target_long t = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = (target_long)(v1 - v2);
        t = t > 0 ? t : -t;
        rd = rd + t;
    }

    return rd;
}

target_ulong HELPER(pdifsumau_b)(CPURISCVState *env, target_ulong rs1,
                            target_ulong rs2, target_ulong rd)
{
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    target_long t = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = (target_long)(v1 -v2);
        t = t > 0 ? t : -t;
        rd = rd + t;
    }

    return rd;
}

target_ulong HELPER(pas_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psa_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psas_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(pssa_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)signed_saturate(env, v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(paas_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(pasa_hx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(psabs_h)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];

        if (v1 == INT16_MIN) {
            v1 = INT16_MAX;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = -v1;
        }

        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(psh1add_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)((v1 << 1) + v2);
    }

    return rd;
}

target_ulong HELPER(pssh1sadd_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p  = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
	if ( v1 < (int16_t)0xC000 ){
	    v1 = (int16_t)0x8000;
        env->vxsat = 0x1;
	}else if ( v1 >= (int16_t)0x4000){
	    v1 = (int16_t)0x7FFF;
	    env->vxsat = 0x1;
	}else{
	    v1 = v1 << 1;
	} 
        rd_p[i] = (int16_t)signed_saturate(env,v1+v2, 16);
    }

    return rd;
}

target_ulong HELPER(pdif_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(pdifu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(pdif_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(pdifu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)(( v1 < v2) ? ( v2 - v1 ) : ( v1 - v2 ));
    }

    return rd;
}

target_ulong HELPER(predsum_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_long rd = rs2;
    int16_t *rs1_p = (int16_t *)&rs1;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (target_ulong)rd;
}

target_ulong HELPER(predsumu_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = rs2;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    target_ulong v1 = 0;
    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

target_ulong HELPER(predsum_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_long rd = rs2;
    int8_t *rs1_p = (int8_t *)&rs1;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (target_ulong)rd;
}

target_ulong HELPER(predsumu_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = rs2;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

uint32_t HELPER(sadd)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) + (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

uint32_t HELPER(saddu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) + (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

uint32_t HELPER(ssub)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) - (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

uint32_t HELPER(ssubu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) - (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

uint32_t HELPER(aadd)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

uint32_t HELPER(aaddu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

uint32_t HELPER(asub)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];
    
    return (v1 - v2) >> 1;
}

uint32_t HELPER(asubu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 - v2) >> 1;
}

uint32_t HELPER(ssh1sadd)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    target_long v1 = rs1_p[0];
    target_long v2 = rs2_p[0];

	if ( v1 < (int32_t)0xC0000000 ){
	    v1 = (int32_t)0x80000000;
        env->vxsat = 0x1;
	}else if ( v1 >= (int32_t)0x40000000){
	    v1 = (int32_t)0x7FFFFFFF;
	    env->vxsat = 0x1;
	}else{
	    v1 = v1 < 1;
	}
    
    return (int32_t)signed_saturate(env, v1 + v2, 32);
}

uint64_t HELPER(padd_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = rs2_p[0];

    for(int i=0; i < TARGET_LONG_SIZE / 4; i++){
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 + v2);
    }

    return rd;
}

uint64_t HELPER(padd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 + v2);
    }

    return rd;
}

uint64_t HELPER(psub_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 - v2);
    }

    return rd;
}

uint64_t HELPER(psadd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 + v2, 32);
    }

    return rd;
}

uint64_t HELPER(psaddu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 + v2, 32);
    }

    return rd;
}

uint64_t HELPER(pssub_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 - v2, 32);
    }

    return rd;
}

uint64_t HELPER(pssubu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 - v2, 32);
    }

    return rd;
}

uint64_t HELPER(paadd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)((v1 + v2) >> 1);
    }

    return rd;
}

uint64_t HELPER(paaddu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 + v2) >> 1);
    }

    return rd;
}

uint64_t HELPER(pasub_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)((v1 - v2) >> 1);
    }

    return rd;
}

uint64_t HELPER(pasubu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 - v2) >> 1);
    }

    return rd;
}

uint64_t HELPER(psh1add_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 << 1) + v2);
    }

    return rd;
}

uint64_t HELPER(pssh1sadd_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p  = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
	if ( v1 < (int32_t)0xC0000000 ){
	    v1 = 0x80000000;
            env->vxsat = 0x1;
	}else if ( v1 >= (int32_t)0x40000000){
	    v1 = 0x7FFFFFFF;
	    env->vxsat = 0x1;
	}else{
	    v1 = v1 << 1;
	}
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 32);
    }

    return rd;
}

uint64_t HELPER(pas_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

uint64_t HELPER(psa_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

uint64_t HELPER(psas_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)signed_saturate(env, v1 + v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 - v2, 32);
        i = i + 2;
    }

    return rd;
}

uint64_t HELPER(pssa_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)signed_saturate(env, v1 - v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 + v2, 32);
        i = i + 2;
    }

    return rd;
}

uint64_t HELPER(paas_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)((v1 + v2) >> 1); 
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

uint64_t HELPER(pasa_wx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int i = 1;

    while (i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)((v1 - v2) >> 1); 
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

uint64_t HELPER(predsum_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = rs2;
    int32_t *rs1_p = (int32_t *)&rs1;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (uint64_t)rd;
}

uint64_t HELPER(predsumu_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = rs2;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

target_ulong HELPER(pslli_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 << imm);
    }

    return rd;
}

target_ulong HELPER(psll_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 << imm);
    }

    return rd;   
}

target_ulong HELPER(psrli_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint8_t)(v1 >> imm);
    }

    return rd;
}

target_ulong HELPER(psrl_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint8_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psrai_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 >> imm);
    }

    return rd;
}

target_ulong HELPER(psra_bs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 >> imm);
    }

    return rd;    
}

target_ulong HELPER(pmin_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(pminu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;   
}

target_ulong HELPER(pmax_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(pmaxu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmseq_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 == v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(pmslt_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int8_t *rs2_p = (int8_t *)&rs2;
    int8_t *rd_p = (int8_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFF : 0;
    }

    return rd;   
}

target_ulong HELPER(pmsltu_b)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint8_t *rs2_p = (uint8_t *)&rs2;
    uint8_t *rd_p = (uint8_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(psext_h_b)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = sextract32(v1, 16, 8);
        t2 = sextract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(psati_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;     
}

target_ulong HELPER(pusati_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << imm) - 1;
        if (v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(pslli_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 << imm);
    }

    return rd;     
}

target_ulong HELPER(psll_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 << imm);
    }

    return rd;      
}

target_ulong HELPER(psrli_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint16_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psrl_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint16_t)(v1 >> imm);
    }

    return rd;      
}

target_ulong HELPER(psrai_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psra_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_ulong imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 >> imm);
    }

    return rd;     
}

target_ulong HELPER(psslai_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 << imm, 16);
    }

    return rd;     
}

target_ulong HELPER(psrari_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong imm)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;

    if (imm == 0) {
        rd = rs1;
    } else {
        for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(((v1 >> (imm - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(pssha_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        if ( imm <= -16){
            rd_p[i] = (int16_t) ( v1 >> 16 );
        }else if( imm > -16 && imm < 0) {
            rd_p[i] = (int16_t) ( v1 >> (-imm) );
        }else if ( imm >= 0 && imm < 16){
            rd_p[i] = (int16_t)signed_saturate(env, v1 << imm, 16);
        }else{
            rd_p[i] = (int16_t) 0x0000;
        }
    }

    return rd;
}

target_ulong HELPER(psshar_hs)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    int32_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++){
        int32_t xrs1 = (int32_t)rs1_p[i];
        if ( imm <= -16){
            xrd = (int32_t) ( xrs1 >> 16 );
            rd_p[i] = (int16_t) (( xrd + 1 ) >> 1);
        }else if( imm > -16 && imm < 0) {
            xrd = (int32_t) ( xrs1 >> (-imm-1) );
            rd_p[i] = (int16_t) (( xrd + 1 ) >> 1);
        }else if ( imm >= 0 && imm < 16){
            rd_p[i] = (int16_t)signed_saturate(env, xrs1 << imm, 16);
        }else{
            rd_p[i] = (int16_t) 0x0000;
        }
    }

    return rd;  
}

target_ulong HELPER(pmin_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;    
}

target_ulong HELPER(pminu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmax_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;     
}

target_ulong HELPER(pmaxu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(pmseq_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 == v2 ? 0xFFFF : 0;
    }

    return rd;     
}

target_ulong HELPER(pmslt_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int16_t *rs2_p = (int16_t *)&rs2;
    int16_t *rd_p = (int16_t *)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFFFF : 0;
    }

    return rd;    
}

target_ulong HELPER(pmsltu_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFFFF : 0;
    }

    return rd;   
}

uint32_t HELPER(sati_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{
    return (uint32_t)signed_saturate(env, (int32_t)rs1, imm + 1);
}

uint32_t HELPER(usati_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{
    uint32_t rd = 0;
    int32_t v1 = rs1;
    int32_t max = (1 << imm) - 1;

    if (v1 > max) {
        v1 = max;
        env->vxsat = 0x1;
    } else if (v1 < 0) {
        v1 = 0;
        env->vxsat = 0x1;
    }
    rd = v1;
    return rd;     
}

uint32_t HELPER(sslai)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{
    uint32_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int64_t v1 = rs1_p[0];

    rd = signed_saturate(env, v1 << imm, 32);

    return rd;     
}

uint32_t HELPER(srari_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t imm)
{
    uint32_t rd = 0;
    int32_t v1 = rs1;

    if (imm == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (imm - 1)) + 1) >> 1;
    }

    return rd;     
}

uint32_t HELPER(ssha)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    int64_t xrs1 = (int64_t)(int32_t)rs1;

    if ( imm <= -32){
        rd = (int32_t) ( xrs1 >> 32 );
    }else if( imm > -32 && imm < 0) {
        rd = (int32_t) ( xrs1 >> (-imm) );
    }else if ( imm >= 0 && imm < 16){
        rd = (int32_t)signed_saturate(env, xrs1 << imm, 32);
    }else{
        rd = (int32_t) 0x00000000;
    }
    return rd;     
}

//saturating SHA with rounding
uint32_t HELPER(sshar)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    int32_t rd = 0;
    int64_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    int64_t xrs1 = (int64_t)(int32_t)rs1;

    if ( imm <= -32){
        xrd = (int64_t) ( xrs1 >> 32 );
        rd = (int32_t) (( xrd + 1 ) >> 1);
    }else if( imm > -32 && imm < 0) {
        xrd = (int64_t) ( xrs1 >> (-imm-1) );
        rd = (int32_t) (( xrd + 1 ) >> 1);
    }else if ( imm >= 0 && imm < 16){
        rd = (int32_t)signed_saturate(env, xrs1 << imm, 32);
    }else{
        rd = (int32_t) 0x00000000;
    }

    return (uint32_t)rd;
}

uint32_t HELPER(mseq)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    return rs1 == rs2 ? 0xFFFFFFFF : 0x00000000;
}

uint32_t HELPER(mslt)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    return (int32_t)rs1 < (int32_t)rs2 ? 0xFFFFFFFF : 0x00000000;
}

uint32_t HELPER(msltu)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    return (uint32_t)rs1 < (uint32_t)rs2 ? 0xFFFFFFFF : 0x00000000;
}

uint64_t HELPER(psext_w_b)(CPURISCVState *env, uint64_t rs1)
{
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    t1 = sextract64(rs1, 32, 8);
    t2 = sextract64(rs1, 0, 8);

    return (uint64_t)((t1 << 32) | (t2 & 0xFFFFFFFF));
}

uint64_t HELPER(psext_w_h)(CPURISCVState *env, uint64_t rs1)
{
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    t1 = sextract64(rs1, 32, 16);
    t2 = sextract64(rs1, 0, 16);

    return (uint64_t)((t1 << 32) | (t2 & 0xFFFFFFFF));
}

uint64_t HELPER(psati_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;
}

uint64_t HELPER(pusati_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        int64_t max = (1 << imm) - 1;
        if (v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;     
}

uint64_t HELPER(pslli_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 << imm);
    }

    return rd;  
}

uint64_t HELPER(psll_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 << (imm & 0x1F));
    }

    return rd;     
}

uint64_t HELPER(psrli_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint32_t)(v1 >> imm);
    }

    return rd;     
}

uint64_t HELPER(psrl_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint32_t)(v1 >> (imm & 0x1F));
    }

    return rd;     
}

uint64_t HELPER(psrai_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 >> imm);
    }

    return rd;     
}

uint64_t HELPER(psra_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 >> (imm & 0x1F));
    }

    return rd;     
}

uint64_t HELPER(psslai_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 << imm, 32);
    }

    return rd;     
}

uint64_t HELPER(psrari_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    if (imm == 0) {
        rd = rs1;
    } else {
        for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)(((v1 >> (imm - 1)) + 1) >> 1);
        }
    }

    return rd;     
}

uint64_t HELPER(pssha_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        if ( imm <= -32){
            rd_p[i] = (int32_t) ( v1 >> 32 );
        }else if( imm > -32 && imm < 0) {
            rd_p[i] = (int32_t) ( v1 >> (-imm) );
        }else if ( imm >= 0 && imm < 32){
            rd_p[i] = (int32_t)signed_saturate(env, v1 << imm, 32);
        }else{
            rd_p[i] = (int32_t) 0x0000;
        }
    }

    return rd;     
}

uint64_t HELPER(psshar_ws)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++){
        int64_t xrs1 = (int64_t)rs1_p[i];
        if ( imm <= -32){
            xrd = (int64_t) ( xrs1 >> 32 );
            rd_p[i] = (int32_t) (( xrd + 1 ) >> 1);
        }else if( imm > -32 && imm < 0) {
            xrd = (int64_t) ( xrs1 >> (-imm-1) );
            rd_p[i] = (int32_t) (( xrd + 1 ) >> 1);
        }else if ( imm >= 0 && imm < 16){
            rd_p[i] = (int32_t)signed_saturate(env, xrs1 << imm, 32);
        }else{
            rd_p[i] = (int32_t) 0x00000000;
        }
    }

    return rd;
}

uint64_t HELPER(pmin_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;     
}

uint64_t HELPER(pminu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;     
}

uint64_t HELPER(pmax_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

uint64_t HELPER(pmaxu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;     
}

uint64_t HELPER(pmseq_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (v1 == v2) ? 0xFFFFFFFF : 0x00000000;
    }

    return rd;
}

uint64_t HELPER(pmslt_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rs2_p = (int32_t *)&rs2;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (v1 < v2) ? 0xFFFFFFFF : 0x00000000;
    }

    return rd;     
}

uint64_t HELPER(pmsltu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint64_t v2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (v1 < v2) ? 0xFFFFFFFF : 0x00000000;
    }

    return rd;      
}

uint64_t HELPER(sati_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, imm + 1);
    }

    return rd;
}

uint64_t HELPER(usati_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int32_t *rs1_p = (int32_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        int64_t max = (1 << imm) - 1;
        if (v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if (v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;  
}

uint64_t HELPER(srari_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int64_t v1 = rs1;

    if (imm == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (imm - 1)) + 1) >> 1;
    }
    return rd;  
}

uint64_t HELPER(sha)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    __int128_t xrs1 = (__int128_t)(int64_t)rs1;
    if ( imm <= -64){
        rd = (int64_t) ( xrs1 >> 64 );
    }else if( imm > -64 && imm < 0) {
        rd = (int64_t) ( xrs1 >> (-imm) );
    }else{
        rd = rs1 << imm;
    }
    return rd;
}

uint64_t HELPER(shar)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    __int128_t xrd = 0;
    int8_t imm = rs2 & 0xFF; //extract rs2[7..0]
    __int128_t xrs1 = (__int128_t)(int64_t)rs1;

    if ( imm <= -64){
        xrd = (__int128_t) ( xrs1 >> 64 );
        rd = (int64_t) (( xrd + 1 ) >> 1);
    }else if( imm > -64 && imm < 0) {
        xrd = (__int128_t) ( xrs1 >> (-imm-1) );
        rd = (int64_t) (( xrd + 1 ) >> 1);
    }else{
        rd = rs1 << imm;
    }

    return (uint64_t)rd;    
}

target_ulong HELPER(ppack_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 0, 8);
        t2 = extract16(rs2_p[i], 0, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

target_ulong HELPER(ppackbt_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 0, 8);
        t2 = extract16(rs2_p[i], 8, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

target_ulong HELPER(ppacktb_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 8, 8);
        t2 = extract16(rs2_p[i], 0, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

target_ulong HELPER(ppackt_h)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint16_t *rs2_p = (uint16_t *)&rs2;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t t1 = 0;
    uint16_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        t1 = extract16(rs1_p[i], 8, 8);
        t2 = extract16(rs2_p[i], 8, 8);
        rd_p[i] = (t2 << 8) | (t1 & 0xFF);
    }

    return rd;  
}

uint32_t HELPER(packbt_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
    uint32_t t1 = extract32(rs1, 0, 16);
    uint32_t t2 = extract32(rs2, 16, 16);
    rd = (t2 << 16) | (t1 & 0xFFFF);
    
    return rd;
}

uint32_t HELPER(packtb_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
    uint32_t t1 = extract32(rs1, 16, 16);
    uint32_t t2 = extract32(rs2, 0, 16);
    rd = (t2 << 16) | (t1 & 0xFFFF);
    
    return rd;
}

uint32_t HELPER(packt_32)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
    uint32_t rd = 0;
    uint32_t t1 = extract32(rs1, 16, 16);
    uint32_t t2 = extract32(rs2, 16, 16);
    rd = (t2 << 16) | (t1 & 0xFFFF);
    
    return rd;
}

uint64_t HELPER(ppack_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16(rs1_p[i], 0, 16);
        t2 = extract16(rs2_p[i], 0, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(ppackbt_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16(rs1_p[i], 0, 16);
        t2 = extract16((rs2_p[i] >> 16), 0, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(ppacktb_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16((rs1_p[i] >> 16), 0, 16);
        t2 = extract16(rs2_p[i], 0, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(ppackt_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t *rs1_p = (uint32_t *)&rs1;
    uint32_t *rs2_p = (uint32_t *)&rs2;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for (int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t1 = extract16((rs1_p[i] >> 16), 0, 16);
        t2 = extract16((rs2_p[i] >> 16), 0, 16);
        rd_p[i] = (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint64_t HELPER(packbt_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = extract64(rs1, 0, 32);
    uint64_t t2 = extract64(rs2, 32, 32);
    rd = (t2 << 32) | (t1 & 0xFFFFFFFF);
    
    return rd;
}

uint64_t HELPER(packtb_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = extract64(rs1, 32, 32);
    uint64_t t2 = extract64(rs2, 0, 32);
    rd = (t2 << 32) | (t1 & 0xFFFFFFFF);
    
    return rd;
}

uint64_t HELPER(packt_64)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = extract64(rs1, 32, 32);
    uint64_t t2 = extract64(rs2, 32, 32);
    rd = (t2 << 32) | (t1 & 0xFFFFFFFF);
    
    return rd;
}

uint64_t HELPER(zip8p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 8*(3-i), 8);
        t2 = extract64(rs2, 8*(3-i), 8);
        rd = (rd <<16) | (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(zip8hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 8*(7-i), 8);
        t2 = extract64(rs2, 8*(7-i), 8);
        rd = (rd <<16) | (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(unzip8p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 16*(3-i), 8);
        t2 = extract64(rs2, 16*(3-i), 8);
        rd = (rd <<8) | (t2 << 32) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(unzip8hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 16*(3-i)+8, 8);
        t2 = extract64(rs2, 16*(3-i)+8, 8);
        rd = (rd <<8) | (t2 << 32) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(zip16p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 16*(1-i), 16);
        t2 = extract64(rs2, 16*(1-i), 16);
        rd = (rd <<32) | (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;   
}

uint64_t HELPER(zip16hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 16*(3-i), 16);
        t2 = extract64(rs2, 16*(3-i), 16);
        rd = (rd <<32) | (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;   
}

uint64_t HELPER(unzip16p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 32*(1-i), 16);
        t2 = extract64(rs2, 32*(1-i), 16);
        rd = (rd <<16) | (t2 << 32) | (t1 & 0xFFFF);
    }

    return rd;   
}

uint64_t HELPER(unzip16hp)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 32*(1-i)+16, 16);
        t2 = extract64(rs2, 32*(1-i)+16, 16);
        rd = (rd << 16) | (t2 << 32) | (t1 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(abs)(CPURISCVState *env, target_ulong rs1)
{
    target_long signed_rs1 = (target_long)rs1;
    return (signed_rs1 < 0) ? (0 - signed_rs1) : signed_rs1;
}

target_ulong HELPER(cls)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    target_long v = (target_long)rs1;
    #if defined(TARGET_RISCV64)
    int64_t lo_bound = 0xC000000000000000;
    int64_t hi_bound = 0x3FFFFFFFFFFFFFFF;
    #elif defined(TARGET_RISCV32)
    int32_t lo_bound = 0xC0000000;
    int32_t hi_bound = 0x3FFFFFFF;
    #endif

    while ( rd < TARGET_LONG_BITS - 1 && v >= lo_bound && v <= hi_bound ) {
        rd = rd + 1;
        v = v << 1;
    }

    return rd;
}

target_ulong HELPER(rev)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    for (int i = 0; i < TARGET_LONG_BITS; i++){
        rd = (rd << 1) | (rs1 & 1);
        rs1 >>= 1;
    }

    return rd;
}

uint64_t HELPER(rev16)(CPURISCVState *env, uint64_t rs1)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;

    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 16*i, 16);
        rd = (rd << 16) | (t1 & 0xFFFF);        
    }

    return rd;
}

target_ulong HELPER(slx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    int shamt = (TARGET_LONG_BITS == 32) ? (rs2 & 0x1F) : (rs2 & 0x3F);
    target_ulong xrs1 = 0; 
    target_ulong xrd = 0; 
    if(shamt <= TARGET_LONG_BITS){
        xrs1 = rs1 >> (TARGET_LONG_BITS - shamt);
        xrd = (rd << shamt) + xrs1;
    }else{
        xrd = rs1 << (shamt - TARGET_LONG_BITS);
    }

    return xrd;
}

target_ulong HELPER(srx)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    int shamt = (TARGET_LONG_BITS == 32) ? (rs2 & 0x1F) : (rs2 & 0x3F);
    target_ulong xrs1 = 0; 
    target_ulong xrd = 0; 
    if(shamt <= TARGET_LONG_BITS){
        xrs1 = rs1 << (TARGET_LONG_BITS - shamt);
        xrd = (rd >> shamt) + xrs1;
    }else{
        xrd = rs1 >> (shamt - TARGET_LONG_BITS);
    }

    return xrd;
}

target_ulong HELPER(mvm)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    return (~rs2 & rd) | (rs2 & rs1);
}

target_ulong HELPER(mvmn)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    return (~rs2 & rs1) | (rs2 & rd);
}

target_ulong HELPER(merge)(CPURISCVState *env, target_ulong rs1,
    target_ulong rs2, target_ulong rd)
{
    return (~rd & rs1) | (rd & rs2);
}

uint64_t HELPER(absw)(CPURISCVState *env, uint64_t rs1)
{
    int32_t rs1_w = rs1 & 0xFFFFFFFF;
    return (rs1_w < 0) ? (0 - rs1_w) : rs1_w;
}

uint64_t HELPER(clsw)(CPURISCVState *env, uint64_t rs1)
{
    int32_t rs1_w = rs1 & 0xFFFFFFFF;
    int c = 0;
    while( c < 31 && rs1_w >= (int32_t)0xC0000000 && rs1_w <= (int32_t)0x3FFFFFFF){
        c = c + 1;
        rs1_w = rs1_w << 1;
    }

    return (uint64_t) c;
}

uint64_t HELPER(pwadd_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwadda_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwaddu_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwaddau_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwsub_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsuba_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int16_t v1 = 0;
    int16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubu_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubau_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    uint16_t v1 = 0;
    uint16_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint16_t)rs1_p[i];
        v2 = (uint16_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwslli_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint16_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsll_bs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint8_t *rs1_p = (uint8_t *)&rs1;
    uint16_t *rd_p = (uint16_t *)&rd;
    uint32_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 4; i++) {
        v1 = (uint32_t) rs1_p[i];
        rd_p[i] = (uint16_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(pwslai_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    int16_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsla_bs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int8_t *rs1_p = (int8_t *)&rs1;
    int16_t *rd_p = (int16_t *)&rd;
    int32_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 4; i++) {
        v1 = (int32_t) rs1_p[i];
        rd_p[i] = (int16_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(pwadd_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwadda_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwaddu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwaddau_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + v1 + v2;
    }

    return rd;
}

uint64_t HELPER(pwsub_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    int64_t rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsuba_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (int32_t)rs1_p[i];
        v2 = (int32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 2; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = v1 - v2;
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwsubau_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t v1 = 0;
    uint32_t v2 = 0;

    for(int i=0; i < 4; i++){
        v1 = (uint32_t)rs1_p[i];
        v2 = (uint32_t)rs2_p[i];
        rd_p[i] = rd_p[i] + (v1 - v2);
    }

    return (uint64_t)rd;
}

uint64_t HELPER(pwslli_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint32_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsll_hs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint16_t *rs1_p = (uint16_t *)&rs1;
    uint32_t *rd_p = (uint32_t *)&rd;
    uint64_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 2; i++) {
        v1 = (uint64_t) rs1_p[i];
        rd_p[i] = (uint32_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(pwslai_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    uint64_t rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int32_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = v1 << imm;
    }

    return rd;
}

uint64_t HELPER(pwsla_hs)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int16_t *rs1_p = (int16_t *)&rs1;
    int32_t *rd_p = (int32_t *)&rd;
    int64_t v1 = 0;
    uint8_t imm = rs2 & 0x1F; //extract rs2[4..0]

    for (int i = 0; i < 2; i++) {
        v1 = (int64_t) rs1_p[i];
        rd_p[i] = (int32_t)(v1 << imm);
    }

    return rd;
}

uint64_t HELPER(wadd)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)((int64_t)(int32_t)rs1 + (int64_t)(int32_t)rs2);
}

uint64_t HELPER(wadda)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return (uint64_t)((int64_t)rd+(int64_t)(int32_t)rs1 + (int64_t)(int32_t)rs2);
}

uint64_t HELPER(waddu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)rs1 + (uint64_t)rs2;
}

uint64_t HELPER(waddau)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return rd + rs1 + rs2;
}

uint64_t HELPER(wsub)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)((int64_t)(int32_t)rs1 - (int64_t)(int32_t)rs2);
}

uint64_t HELPER(wsuba)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return (uint64_t)( (int64_t)rd + (int64_t)(int32_t)rs1 - (int64_t)(int32_t)rs2 );
}

uint64_t HELPER(wsubu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)rs1 - (uint64_t)rs2;
}

uint64_t HELPER(wsubau)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t rd)
{
    return rd + rs1 - rs2;
}

uint64_t HELPER(wslli)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    return rs1 << imm;
}

uint64_t HELPER(wsll)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint8_t imm = rs2 & 0x3F; //extract rs2[5..0]
    return rs1 << imm;
}

uint64_t HELPER(wslai)(CPURISCVState *env, uint64_t rs1,
    uint64_t imm)
{
    return (int64_t)(int32_t)rs1 << imm;
}

uint64_t HELPER(wsla)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint8_t imm = rs2 & 0x3F; //extract rs2[5..0]
    return (int64_t)(int32_t)rs1 << imm;
}

uint64_t HELPER(wzip8p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 4; i++){
        t1 = extract64(rs1, 8*(3-i), 8);
        t2 = extract64(rs2, 8*(3-i), 8);
        rd = (rd <<16) | (t2 << 8) | (t1 & 0xFF);
    }

    return rd;
}

uint64_t HELPER(wzip16p)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    
    for(int i = 0; i < 2; i++){
        t1 = extract64(rs1, 16*(1-i), 16);
        t2 = extract64(rs2, 16*(1-i), 16);
        rd = (rd <<32) | (t2 << 16) | (t1 & 0xFFFF);
    }

    return rd;
}

uint32_t HELPER(predsum_dbs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    int32_t rd = rs2;
    int8_t *rs1_p = (int8_t *)&rs1_l;
    int32_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (int8_t *)&rs1_h;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (uint32_t)rd;
}

uint32_t HELPER(predsumu_dbs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    uint32_t rd = rs2;
    uint8_t *rs1_p = (uint8_t *)&rs1_l;
    uint32_t v1 = 0;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (uint8_t *)&rs1_h;

    for (int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

uint32_t HELPER(predsum_dhs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    int32_t rd = rs2;
    int16_t *rs1_p = (int16_t *)&rs1_l;
    int16_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (int16_t *)&rs1_h;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return (uint32_t)rd;
}

uint32_t HELPER(predsumu_dhs)(CPURISCVState *env, uint32_t rs1_l,
    uint32_t rs1_h, uint32_t rs2)
{
    uint32_t rd = rs2;
    uint16_t *rs1_p = (uint16_t *)&rs1_l;
    uint32_t v1 = 0;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }

    rs1_p = (uint16_t *)&rs1_h;

    for (int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        rd += v1;
    }
    return rd;
}

uint32_t HELPER(pnsrli_b)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        uint8_t result = (s1_h >> (shamt & 0xF)) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsrai_b)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int32_t s1_h_s32 = (int32_t)(int16_t)s1_h;
        int32_t s1_h_s24 = (s1_h_s32 << 8) >> 8;
        uint8_t result = s1_h_s24 >> (shamt & 0xF) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsrari_b)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
     uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int32_t s1_h_s32 = (int32_t)(int16_t)s1_h;
        int32_t s1_h_s24 = (s1_h_s32 << 8) >> 8;
        uint32_t shx_25bit = ((uint32_t)s1_h_s24 << 1);
        uint32_t shx = (shx_25bit >> (shamt & 0xF)) & 0x1FF;
        uint8_t result = ((shx + 1) >> 1) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnclipi_b)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;

	for (int i = 0; i < 4; i++) {
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		int32_t s1_h_s32 = (int32_t)(int16_t)s1_h;
		int16_t shx = (int16_t)(s1_h_s32 >> (shamt & 0xF));
        uint8_t result = 0;
        if (shx < -128) {
            env->vxsat = 1;
            result = 0x80; // -128
        } else if (shx > 127) {
            env->vxsat = 1;
            result = 0x7F; // 127
        } else {
            result = (uint8_t)shx;
        }
        rd |= ((uint32_t)result << (i * 8));
    }

    return rd;
}

uint32_t HELPER(pnclipri_b)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 4; i++) {
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		int32_t s1_h_s32 = (int32_t)(int16_t)s1_h;
        uint64_t shx_33bit = ((uint32_t)s1_h_s32 << 1);
        uint32_t shx = (shx_33bit >> (shamt & 0xF)) & 0x1FFFF;
		uint16_t round_shx = (uint16_t)((shx + 1) >> 1);
		int16_t round_shx_s = (int16_t)round_shx;
		uint8_t result = 0;
		if (round_shx_s < -128) {
            env->vxsat = 1;
            result = 0x80; // -128
        } else if (round_shx_s > 127) {
            env->vxsat = 1;
            result = 0x7F; // 127
        } else {
            result = (uint8_t)round_shx;
        }

        rd |= ((uint32_t)result) << (i * 8);
	}

	return rd;
}

uint32_t HELPER(pnclipiu_b)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 4; i++) {
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		uint16_t shx = s1_h >> (shamt & 0xF);
		uint8_t result = 0;
		if (shx > 0x00FF) {
			env->vxsat = 1;
			result = 0xFF;
		}else{
			result = (uint8_t)(shx & 0xFF);
		}
		rd |= ((uint32_t)result) << (i * 8);
	}

	return rd;
}

uint32_t HELPER(pnclipriu_b)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 4; i++) {
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		uint32_t shx_17bit = ((uint32_t)s1_h << 1);
		uint32_t shx = shx_17bit >> (shamt & 0xF);
		uint16_t round_shx = (uint16_t)((shx + 1) >> 1);
		uint8_t result = 0;
		if (round_shx > 0x00FF) {
			env->vxsat = 1;
			result = 0xFF;
		}else{
			result = (uint8_t)(round_shx & 0xFF);
		}
		rd |= ((uint32_t)result) << (i * 8);
	}

	return rd;
}

uint32_t HELPER(pnsrl_bs)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        uint32_t s1_h_z32 = (uint32_t)s1_h;
        uint8_t result = (s1_h_z32 >> (shamt & 0x1F)) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsra_bs)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int64_t s1_h_s64 = (int64_t)(int16_t)s1_h;
        s1_h_s64 = (s1_h_s64 << 24) >> 24;
        uint8_t result = s1_h_s64 >> (shamt & 0x1F) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnsrar_bs)(CPURISCVState *env, uint64_t s1,
    uint32_t shamt)
{
    uint32_t rd = 0;

    for (int i = 0; i < 4; i++) {
        uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
        int64_t s1_h_s64 = (int64_t)(int16_t)s1_h;
        int64_t s1_h_s40 = (s1_h_s64 << 24) >> 24;
        uint64_t shx_41bit = ((uint64_t)s1_h_s40 << 1);
        uint64_t shx = (shx_41bit >> (shamt & 0x1F)) & 0x1FF;
        uint8_t result = ((shx + 1) >> 1) & 0xFF;
        rd |= ((uint32_t)result) << (i * 8);
    }

    return rd;
}

uint32_t HELPER(pnclip_bs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 4; i++)
	{
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		int64_t s1_h_s64 = (int64_t)(int16_t)s1_h;
		int64_t s1_h_s48 = (s1_h_s64 << 16) >> 16;
		int16_t shx = (int16_t)(s1_h_s48 >> (shamt & 0x1F));
		uint8_t result = 0;
		if (shx < -128) {
			env->vxsat = 1;
			result = 0x80;
		} else if (shx > 127) {
			env->vxsat = 1;
			result = 0x7F;
		}else{
			result = (uint8_t)shx;
		}
		rd |= ((uint32_t)result) << (i * 8);
	}

	return rd;
}

uint32_t HELPER(pnclipr_bs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 4; i++)
	{
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		int64_t s1_h_s64 = (int64_t)(int16_t)s1_h;
		int64_t s1_h_s48 = (s1_h_s64 << 16) >> 16;
		uint64_t shx_49bit = ((uint64_t)s1_h_s48 << 1);
		uint32_t shx = (shx_49bit >> (shamt & 0x1F)) & 0x1FFFF;
		uint16_t round_shx = (uint16_t)((shx + 1) >> 1);
		int16_t round_shx_s = (int16_t)round_shx;
		uint8_t result = 0;
		if (round_shx_s < -128) {
			env->vxsat = 1;
			result = 0x80;
		} else if (round_shx_s > 127) {
			env->vxsat = 1;
			result = 0x7F;
		}else{
			result = (uint8_t)round_shx;
		}
		rd |= ((uint32_t)result) << (i * 8);
	}

	return rd;
}

uint32_t HELPER(pnclipu_bs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 4; i++){
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		uint32_t s1_h_z32 = (uint32_t)s1_h;
		uint16_t shx = (s1_h_z32 >> (shamt & 0x1F)) & 0xFFFF;
		uint8_t result = 0;
		if (shx > 0x00FF) {
			env->vxsat = 1;
			result = 0xFF;
		}else{
			result = (uint8_t)(shx & 0xFF);
		}
		rd |= ((uint32_t)result) << (i * 8);
	}

	return rd;
}

uint32_t HELPER(pnclipru_bs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 4; i++){
		uint16_t s1_h = (s1 >> (i * 16)) & 0xFFFF;
		uint32_t s1_h_z32 = (uint32_t)s1_h;
		uint64_t shx_33bit = ((uint64_t)s1_h_z32 << 1);
		uint32_t shx = (shx_33bit >> (shamt & 0x1F)) & 0x1FFFF;
		uint16_t round_shx = (uint16_t)((shx + 1) >> 1);
		uint8_t result = 0;
		if (round_shx > 0x00FF) {
			env->vxsat = 1;
			result = 0xFF;
		}else{
			result = (uint8_t)(round_shx & 0xFF);
		}
		rd |= ((uint32_t)result) << (i * 8);
	}
	return rd;
}

uint32_t HELPER(pnsrli_h)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	uint32_t s1_low  = (uint32_t)(s1 & 0xFFFFFFFF);
    uint32_t s1_high = (uint32_t)((s1 >> 32) & 0xFFFFFFFF);

    uint16_t rd_low  = (s1_low  >> (shamt & 0x1F)) & 0xFFFF;
    uint16_t rd_high = (s1_high >> (shamt & 0x1F)) & 0xFFFF;

    rd = ((uint32_t)rd_high << 16) | rd_low;
    return rd;
}

uint32_t HELPER(pnsrai_h)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	uint32_t s1_low  = (uint32_t)(s1 & 0xFFFFFFFF);
	int64_t s1_low_s64 = (int64_t)(int32_t)s1_low;
	int64_t s1_low_s48 = (s1_low_s64  << 16) >> 16;

    uint32_t s1_high = (uint32_t)((s1 >> 32) & 0xFFFFFFFF);
	int64_t s1_high_s64 = (int64_t)(int32_t)s1_high;
	int64_t s1_high_s48 = (s1_high_s64  << 16) >> 16;

    uint16_t rd_low  = (s1_low_s48  >> (shamt & 0x1F)) & 0xFFFF;
    uint16_t rd_high = (s1_high_s48 >> (shamt & 0x1F)) & 0xFFFF;

    rd = ((uint32_t)rd_high << 16) | rd_low;
    return rd;
}

uint32_t HELPER(pnsrari_h)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		int64_t s1_w_s64 = (int64_t)(int32_t)s1_w;
		int64_t s1_w_s48 = (s1_w_s64 << 16) >> 16;
		uint64_t shx_49bit = ((uint64_t)s1_w_s48 << 1);
		uint32_t shx = (shx_49bit >> (shamt & 0x1F)) & 0x1FFFF;
		rd |= ((uint16_t)((shx + 1) >> 1)) << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclipi_h)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		int64_t s1_w_s64 = (int64_t)(int32_t)s1_w;
		int32_t shx = (int32_t)(s1_w_s64 >> (shamt & 0x1F));
		uint16_t result = 0;
		if(shx < -32768){
			env->vxsat = 1;
			result = 0x8000;
		} else if(shx > 32767){
			env->vxsat = 1;
			result = 0x7FFF;
		}else{
			result = (uint16_t)(shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclipri_h)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		int64_t s1_w_s64 = (int64_t)(int32_t)s1_w;
		__uint128_t shx_65bit = ((uint64_t)s1_w_s64 << 1);
		uint64_t shx = (shx_65bit >> (shamt & 0x1F)) & 0x1FFFFFFFF;
		int32_t round_shx = (int32_t)((shx + 1) >> 1);
		uint16_t result = 0;
		if(round_shx < -32768){
			env->vxsat = 1;
			result = 0x8000;
		} else if(round_shx > 32767){
			env->vxsat = 1;
			result = 0x7FFF;
		}else{
			result = (uint16_t)(round_shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclipiu_h)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		uint32_t shx = s1_w >> (shamt & 0x1F);
		uint16_t result = 0;
		if(shx > 65535){
			env->vxsat = 1;
			result = 0xFFFF;
		}else{
			result = (uint16_t)(shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclipriu_h)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		uint64_t shx_33bit = ((uint64_t)s1_w << 1);
		uint64_t shx = shx_33bit >> (shamt & 0x1F);
		uint32_t round_shx = (uint32_t)((shx + 1) >> 1);
		uint16_t result = 0;
		if(round_shx > 65535){
			env->vxsat = 1;
			result = 0xFFFF;
		}else{
			result = (uint16_t)(round_shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnsrl_hs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	uint32_t s1_low  = (uint32_t)(s1 & 0xFFFFFFFF);
    uint32_t s1_high = (uint32_t)((s1 >> 32) & 0xFFFFFFFF);

    uint16_t rd_low  = (s1_low  >> (shamt & 0x1F)) & 0xFFFF;
    uint16_t rd_high = (s1_high >> (shamt & 0x1F)) & 0xFFFF;

    rd = ((uint32_t)rd_high << 16) | rd_low;
    return rd;
}

uint32_t HELPER(pnsra_hs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	uint32_t s1_low  = (uint32_t)(s1 & 0xFFFFFFFF);
    uint32_t s1_high = (uint32_t)((s1 >> 32) & 0xFFFFFFFF);

    uint16_t rd_low  = (s1_low  >> (shamt & 0x1F)) & 0xFFFF;
    uint16_t rd_high = (s1_high >> (shamt & 0x1F)) & 0xFFFF;

    rd = ((uint32_t)rd_high << 16) | rd_low;
    return rd;
}

uint32_t HELPER(pnsrar_hs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		int64_t s1_w_s64 = (int64_t)(int32_t)s1_w;
		int64_t s1_w_s48 = (s1_w_s64 << 16) >> 16;
		uint64_t shx_49bit = ((uint64_t)s1_w_s48 << 1);
		uint32_t shx = (shx_49bit >> (shamt & 0x1F)) & 0x1FFFF;
		rd |= ((uint16_t)((shx + 1) >> 1)) << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclip_hs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		int64_t s1_w_s64 = (int64_t)(int32_t)s1_w;
		int32_t shx = (int32_t)(s1_w_s64 >> (shamt & 0x1F));
		uint16_t result = 0;
		if(shx < -32768){
			env->vxsat = 1;
			result = 0x8000;
		} else if(shx > 32767){
			env->vxsat = 1;
			result = 0x7FFF;
		}else{
			result = (uint16_t)(shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclipr_hs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		int64_t s1_w_s64 = (int64_t)(int32_t)s1_w;
		__uint128_t shx_65bit = ((uint64_t)s1_w_s64 << 1);
		uint64_t shx = (shx_65bit >> (shamt & 0x1F)) & 0x1FFFFFFFF;
		int32_t round_shx = (int32_t)((shx + 1) >> 1);
		uint16_t result = 0;
		if(round_shx < -32768){
			env->vxsat = 1;
			result = 0x8000;
		} else if(round_shx > 32767){
			env->vxsat = 1;
			result = 0x7FFF;
		}else{
			result = (uint16_t)(round_shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclipu_hs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		uint32_t shx = s1_w >> (shamt & 0x1F);
		uint16_t result = 0;
		if(shx > 65535){
			env->vxsat = 1;
			result = 0xFFFF;
		}else{
			result = (uint16_t)(shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(pnclipru_hs)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	for (int i = 0; i < 2; i++){
		uint32_t s1_w = (s1 >> (i * 32)) & 0xFFFFFFFF;
		uint64_t shx_33bit = ((uint64_t)s1_w << 1);
		uint64_t shx = shx_33bit >> (shamt & 0x1F);
		uint32_t round_shx = (uint32_t)((shx + 1) >> 1);
		uint16_t result = 0;
		if(round_shx > 65535){
			env->vxsat = 1;
			result = 0xFFFF;
		}else{
			result = (uint16_t)(round_shx & 0xFFFF);
		}
		rd |= result << (i * 16);
	}

	return rd;
}

uint32_t HELPER(nsrli)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = (s1 >> (shamt & 0x3F)) & 0xFFFFFFFF;
	return rd;
}

uint32_t HELPER(nsrai)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__int128_t s1_s128 = (__int128_t)((int64_t)s1);
	__int128_t s1_s96 = (s1_s128 << 32) >> 32;
	rd = (uint32_t)(s1_s96 >> (shamt & 0x3F)) & 0xFFFFFFFF;
	return rd;
}

uint32_t HELPER(nsrari)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__int128_t s1_s128 = (__int128_t)((int64_t)s1);
	__int128_t s1_s96 = (s1_s128 << 32) >> 32;
	__uint128_t shx_97bit = ((__uint128_t)s1_s96 << 1);
	uint64_t shx = (uint64_t)(shx_97bit >> (shamt & 0x3F)) & 0x1FFFFFFFF;
	rd = (uint32_t)((shx + 1) >> 1);
	return rd;
}

uint32_t HELPER(nclipi)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__int128_t s1_s128 = (__int128_t)((int64_t)s1);
	int64_t shx = (int64_t)(s1_s128 >> (shamt & 0x3F));
	if(shx < -2147483648){								//_s 0xFFFFFFFF80000000
		env->vxsat = 1;
		rd = 0x80000000;
	} else if(shx > 2147483647){						//_s 0x000000007FFFFFFF
		env->vxsat = 1;
		rd = 0x7FFFFFFF;
	}else{
		rd = (uint32_t)(shx & 0xFFFFFFFF);
	}
	return rd;
}

uint32_t HELPER(nclipri)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
    typedef struct {
        __uint128_t low;
        uint8_t high;
    } Uint129;

    Uint129 left_shift_1(__int128_t s1_s128) {
        Uint129 result;
        __uint128_t us1 = (__uint128_t)s1_s128;
        result.low = us1 << 1;
        result.high = (us1 >> 127) & 0x1;
        return result;
    }

    Uint129 right_shift(Uint129 val, uint32_t smt) {
        Uint129 result;
        if (smt == 0) {
            return val;
        } else if (smt >= 129) {
            result.low = 0;
            result.high = 0;
        } else if (smt == 128) {
            result.low = val.high;
            result.high = 0;
        } else {
            result.low = (val.low >> smt) | ((__uint128_t)val.high << (128 - smt));
            result.high = (val.high >> smt);
        }
        return result;
    }

    uint32_t rd = 0;
    __int128_t s1_s128 = (__int128_t)((int64_t)s1);
    Uint129 shx_129bit = left_shift_1(s1_s128);
    Uint129 shx = right_shift(shx_129bit, shamt & 0x3F);
    int64_t round_shx = (int64_t)((shx.low + 1) >> 1);
    if (round_shx < -2147483648) {		//_s 0xFFFFFFFF80000000
        env->vxsat = 1;
        rd = 0x80000000;
    } else if (round_shx > 2147483647) {	//_s 0x000000007FFFFFFF
        env->vxsat = 1;
        rd = 0x7FFFFFFF;
    } else {
        rd = (uint32_t)round_shx;
    }
    return rd;
}

uint32_t HELPER(nclipiu)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	uint64_t shx = s1 >> (shamt & 0x3F);
	if(shx > 4294967295){				//_u 0x00000000FFFFFFFF
		env->vxsat = 1;
		rd = 0xFFFFFFFF;
	}else{
		rd = (uint32_t)(shx & 0xFFFFFFFF);
	}

	return rd;
}

uint32_t HELPER(nclipriu)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__uint128_t shx_65bit = (s1 << 1);
	__uint128_t shx = shx_65bit >> (shamt & 0x3F);
	uint64_t round_shx = (shx + 1) >> 1;
	if(round_shx > 4294967295){				//_u 0x00000000FFFFFFFF
		env->vxsat = 1;
		rd = 0xFFFFFFFF;
	}else{
		rd = (uint32_t)(round_shx & 0xFFFFFFFF);
	}

	return rd;
}

uint32_t HELPER(nsrl)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = (s1 >> (shamt & 0x3F)) & 0xFFFFFFFF;
	return rd;
}

uint32_t HELPER(nsra)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__int128_t s1_s128 = (__int128_t)((int64_t)s1);
	__int128_t s1_s96 = (s1_s128 << 32) >> 32;
	rd = (uint32_t)(s1_s96 >> (shamt & 0x3F)) & 0xFFFFFFFF;
	return rd;
}

uint32_t HELPER(nsrar)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__int128_t s1_s128 = (__int128_t)((int64_t)s1);
	__int128_t s1_s96 = (s1_s128 << 32) >> 32;
	__uint128_t shx_97bit = ((__uint128_t)s1_s96 << 1);
	uint64_t shx = (uint64_t)(shx_97bit >> (shamt & 0x3F)) & 0x1FFFFFFFF;
	rd = (uint32_t)((shx + 1) >> 1);
	return rd;
}

uint32_t HELPER(nclip)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__int128_t s1_s128 = (__int128_t)((int64_t)s1);
	int64_t shx = (int64_t)(s1_s128 >> (shamt & 0x3F));
	if(shx < -2147483648){								//_s 0xFFFFFFFF80000000
		env->vxsat = 1;
		rd = 0x80000000;
	} else if(shx > 2147483647){						//_s 0x000000007FFFFFFF
		env->vxsat = 1;
		rd = 0x7FFFFFFF;
	}else{
		rd = (uint32_t)(shx & 0xFFFFFFFF);
	}
	return rd;
}

uint32_t HELPER(nclipr)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
    typedef struct {
        __uint128_t low;
        uint8_t high;
    } Uint129;

    Uint129 left_shift_1(__int128_t s1_s128) {
        Uint129 result;
        __uint128_t us1 = (__uint128_t)s1_s128;
        result.low = us1 << 1;
        result.high = (us1 >> 127) & 0x1;
        return result;
    }

    Uint129 right_shift(Uint129 val, uint32_t smt) {
        Uint129 result;
        if (smt == 0) {
            return val;
        } else if (smt >= 129) {
            result.low = 0;
            result.high = 0;
        } else if (smt == 128) {
            result.low = val.high;
            result.high = 0;
        } else {
            result.low = (val.low >> smt) | ((__uint128_t)val.high << (128 - smt));
            result.high = (val.high >> smt);
        }
        return result;
    }

    uint32_t rd = 0;
    __int128_t s1_s128 = (__int128_t)((int64_t)s1);
    Uint129 shx_129bit = left_shift_1(s1_s128);
    Uint129 shx = right_shift(shx_129bit, shamt & 0x3F);
    int64_t round_shx = (int64_t)((shx.low + 1) >> 1);

    if (round_shx < -2147483648) {		//_s 0xFFFFFFFF80000000
        env->vxsat = 1;
        rd = 0x80000000;
    } else if (round_shx > 2147483647) {	//_s 0x000000007FFFFFFF
        env->vxsat = 1;
        rd = 0x7FFFFFFF;
    } else {
        rd = (uint32_t)round_shx;
    }

    return rd;
}

uint32_t HELPER(nclipu)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	uint64_t shx = s1 >> (shamt & 0x3F);
	if(shx > 4294967295){				//_u 0x00000000FFFFFFFF
		env->vxsat = 1;
		rd = 0xFFFFFFFF;
	}else{
		rd = (uint32_t)(shx & 0xFFFFFFFF);
	}

	return rd;
}

uint32_t HELPER(nclipru)(CPURISCVState *env, uint64_t s1,
	uint32_t shamt)
{
	uint32_t rd = 0;
	__uint128_t shx_65bit = (s1 << 1);
	__uint128_t shx = shx_65bit >> (shamt & 0x3F);
	uint64_t round_shx = (shx + 1) >> 1;
	if(round_shx > 4294967295){				//_u 0x00000000FFFFFFFF
		env->vxsat = 1;
		rd = 0xFFFFFFFF;
	}else{
		rd = (uint32_t)(round_shx & 0xFFFFFFFF);
	}

	return rd;
}

target_ulong HELPER(pmulh_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        int16_t s2_h = (int16_t)((s2 >> (i * 16)) & 0xFFFF);

        int32_t prod = (int32_t)s1_h * (int32_t)s2_h;
        uint16_t high16 = (prod >> 16) & 0xFFFF;

        rd |= ((target_ulong)high16) << (i * 16);	
    }
    return rd;
}

target_ulong HELPER(pmulhsu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);

        int32_t prod = (int32_t)s1_h * (uint32_t)s2_h;
        uint16_t high16 = (prod >> 16) & 0xFFFF;

        rd |= ((target_ulong)high16) << (i * 16);
	 }
    return rd;
}

target_ulong HELPER(pmulhu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        uint16_t s1_h = (uint16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);

        uint32_t prod = (uint32_t)s1_h * (uint32_t)s2_h;
        uint16_t high16 = (prod >> 16) & 0xFFFF;

        rd |= ((target_ulong)high16) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulq_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;
    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        int16_t s2_h = (int16_t)((s2 >> (i * 16)) & 0xFFFF);
        uint16_t result = 0;
        if((s1_h == -32768) && (s2_h == -32768) ) {
            env->vxsat = 1;
            result = 0x7FFF;
        }else{
            int32_t prod = (int32_t)s1_h * (int32_t)s2_h;
            result = (prod >> 15) & 0xFFFF;
        }
        rd |= ((target_ulong)result) << (i * 16);
    }
    return rd;
}
	

target_ulong HELPER(pmulhr_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        int16_t s2_h = (int16_t)((s2 >> (i * 16)) & 0xFFFF);

        int32_t prod = (int32_t)s1_h * (int32_t)s2_h + (1LL << 15);
        uint16_t high16 = (prod >> 16) & 0xFFFF;

        rd |= ((target_ulong)high16) << (i * 16);
	}
    return rd;
}

target_ulong HELPER(pmulhrsu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);

        int32_t prod = (int32_t)s1_h * (uint32_t)s2_h + (1LL << 15);
        uint16_t high16 = (prod >> 16) & 0xFFFF;

        rd |= ((target_ulong)high16) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulhru_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        uint16_t s1_h = (uint16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);

        uint32_t prod = (uint32_t)s1_h * (uint32_t)s2_h + (1LL << 15);
        uint16_t high16 = (prod >> 16) & 0xFFFF;

        rd |= ((target_ulong)high16) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulqr_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        int16_t s2_h = (int16_t)((s2 >> (i * 16)) & 0xFFFF);
        uint16_t result = 0;
        if((s1_h == -32768) && (s2_h == -32768)) {
            env->vxsat = 1;
            result = 0x7FFF;
        }else{
            int32_t prod = (int32_t)s1_h * (int32_t)s2_h + (1LL << 14);
            result = (prod >> 15) & 0xFFFF;
        }
        rd |= ((target_ulong)result) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhacc_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong d)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        int16_t s2_h = (int16_t)((s2 >> (i * 16)) & 0xFFFF);
        int16_t d_h = (int16_t)((d >> (i * 16)) & 0xFFFF);
        int32_t prod = (int32_t)s1_h * (int32_t)s2_h;
        int16_t high16 = (int16_t)(prod >> 16);

        rd |= ((target_ulong)(uint16_t)(high16 + d_h)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhaccsu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong d)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);
        int16_t d_h = (int16_t)((d >> (i * 16)) & 0xFFFF);
        int32_t prod = (int32_t)s1_h * (uint32_t)s2_h;
        int16_t high16 = (int16_t)(prod >> 16);

        rd |= ((target_ulong)(uint16_t)(high16 + d_h)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhaccu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong d)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        uint16_t s1_h = (uint16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);
        uint16_t d_h = (uint16_t)((d >> (i * 16)) & 0xFFFF);
        uint32_t prod = (uint32_t)s1_h * (uint32_t)s2_h;
        uint16_t high16 = (uint16_t)(prod >> 16);

        rd |= ((target_ulong)(uint16_t)(high16 + d_h)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhracc_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong d)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        int16_t s2_h = (int16_t)((s2 >> (i * 16)) & 0xFFFF);
        int16_t d_h = (int16_t)((d >> (i * 16)) & 0xFFFF);
        int32_t prod = (int32_t)s1_h * (int32_t)s2_h + (1LL << 15);
        int16_t high16 = (int16_t)(prod >> 16);

        rd |= ((target_ulong)(uint16_t)(high16 + d_h)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhraccsu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong d)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);
        int16_t d_h = (int16_t)((d >> (i * 16)) & 0xFFFF);
        int32_t prod = (int32_t)s1_h * (uint32_t)s2_h + (1LL << 15);
        int16_t high16 = (int16_t)(prod >> 16);

        rd |= ((target_ulong)(uint16_t)(high16 + d_h)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhraccu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong d)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        uint16_t s1_h = (uint16_t)((s1 >> (i * 16)) & 0xFFFF);
        uint16_t s2_h = (uint16_t)((s2 >> (i * 16)) & 0xFFFF);
        uint16_t d_h = (uint16_t)((d >> (i * 16)) & 0xFFFF);
        uint32_t prod = (uint32_t)s1_h * (uint32_t)s2_h + (1LL << 15);
        uint16_t high16 = (uint16_t)(prod >> 16);

        rd |= ((target_ulong)(uint16_t)(high16 + d_h)) << (i * 16);
    }
    return rd;
}

uint32_t HELPER(mulhr)(CPURISCVState *env, uint32_t rs1,
                             uint32_t rs2)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    int32_t s2 = (int32_t)rs2;
    int64_t t = (int64_t)s1 * (int64_t)s2 + (1LL << 31);

    rd = (int32_t)((uint64_t)t >> 32);
    return rd;
}

uint32_t HELPER(mulhrsu)(CPURISCVState *env, uint32_t rs1,
                             uint32_t rs2)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    uint32_t s2 = (uint32_t)rs2;
    int64_t t = (int64_t)s1 * (uint64_t)s2 + (1LL << 31);

    rd = (int32_t)((uint64_t)t >> 32);
    return rd;
}

uint32_t HELPER(mulhru)(CPURISCVState *env, uint32_t rs1,
                             uint32_t rs2)
{
    uint32_t rd = 0;
    uint32_t s1 = (uint32_t)rs1;
    uint32_t s2 = (uint32_t)rs2;
    uint64_t t = (uint64_t)s1 * (uint64_t)s2 + (1LL << 31);

    rd = (uint32_t)((uint64_t)t >> 32);
    return rd;
}

uint32_t HELPER(mulq)(CPURISCVState *env, uint32_t rs1,
                             uint32_t rs2)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    int32_t s2 = (int32_t)rs2;
    if(rs1 == INT32_MIN && rs2 == INT32_MIN) {
        env->vxsat = 1;
        rd = INT32_MAX;
    }else{
        int64_t t = (int64_t)s1 * (int64_t)s2;
        rd = (int32_t)(t >> 31);
    }

    return rd;
}

uint32_t HELPER(mulqr)(CPURISCVState *env, uint32_t rs1,
                             uint32_t rs2)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    int32_t s2 = (int32_t)rs2;
    if(rs1 == INT32_MIN && rs2 == INT32_MIN) {
        env->vxsat = 1;
        rd = INT32_MAX;
    }else{
        int64_t t = (int64_t)s1 * (int64_t)s2 + (1LL << 30);
        rd = (uint32_t)(t >> 31);
    }

    return rd;
}

uint64_t HELPER(pmulh_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        int32_t rs2_w = (int32_t)(rs2 >> (i * 32));
        int64_t t = (int64_t)rs1_w * (int64_t)rs2_w;
        rd |= (uint64_t)(uint32_t)((uint64_t)t >> 32) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulhr_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        int32_t rs2_w = (int32_t)(rs2 >> (i * 32));
        int64_t t = (int64_t)rs1_w * (int64_t)rs2_w + (1LL << 31);

        rd |= (uint64_t)(uint32_t)((uint64_t)t >> 32) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulhsu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        int64_t t = (int64_t)rs1_w * (uint64_t)rs2_w;

         rd |= (uint64_t)(uint32_t)((uint64_t)t >> 32) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulhrsu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        int64_t t = (int64_t)rs1_w * (uint64_t)rs2_w + (1LL << 31);

         rd |= (uint64_t)(uint32_t)((uint64_t)t >> 32) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulhu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint32_t rs1_w = (uint32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        uint64_t t = (uint64_t)rs1_w * (uint64_t)rs2_w;

         rd |= (uint64_t)(uint32_t)((uint64_t)t >> 32) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulhru_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint32_t rs1_w = (uint32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        uint64_t t = (uint64_t)rs1_w * (uint64_t)rs2_w + (1LL << 31);

         rd |= (uint64_t)(uint32_t)((uint64_t)t >> 32) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulq_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        int32_t rs2_w = (int32_t)(rs2 >> (i * 32));
        uint32_t result = 0;
        if(rs1_w == INT32_MIN && rs2_w == INT32_MIN) {
            env->vxsat = 1;
            result = 0x7FFFFFFF;
        }else{
            int64_t t = (int64_t)rs1_w * (int64_t)rs2_w;
            result = (uint32_t)((uint64_t)t >> 31);
        }
         rd |= (uint64_t)result << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulqr_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        int32_t rs2_w = (int32_t)(rs2 >> (i * 32));
        uint32_t result = 0;
        if(rs1_w == INT32_MIN && rs2_w == INT32_MIN) {
            env->vxsat = 1;
            result = 0x7FFFFFFF;
        }else{
            int64_t t = (int64_t)rs1_w * (int64_t)rs2_w + (1LL << 30);
            result = (uint32_t)((uint64_t)t >> 31);	
        }
         rd |= (uint64_t)result << (i * 32);
    }

    return rd;
}

uint32_t HELPER(mhacc)(CPURISCVState *env, uint32_t rs1,
          uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    int32_t s2 = (int32_t)rs2;
    int32_t d = (int32_t)dest;
    int64_t t = (int64_t)s1 * (int64_t)s2;

    rd = (int32_t)(d + ((uint64_t)t >> 32));
    return rd;
}

uint32_t HELPER(mhracc)(CPURISCVState *env, uint32_t rs1,
          uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    int32_t s2 = (int32_t)rs2;
    int32_t d = (int32_t)dest;
    int64_t t = (int64_t)s1 * (int64_t)s2 + (1LL << 31);

    rd = (int32_t)(d + ((uint64_t)t >> 32));
    return rd;
}

uint32_t HELPER(mhaccsu)(CPURISCVState *env, uint32_t rs1,
          uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    uint32_t s2 = (uint32_t)rs2;
    int32_t d = (int32_t)dest;
    int64_t t = (int64_t)s1 * (uint64_t)s2;

    rd = (int32_t)(d + ((uint64_t)t >> 32));
    return rd;
}

uint32_t HELPER(mhraccsu)(CPURISCVState *env, uint32_t rs1,
          uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
    uint32_t s2 = (uint32_t)rs2;
    int32_t d = (int32_t)dest;
    int64_t t = (int64_t)s1 * (uint64_t)s2 + (1LL << 31);

    rd = (int32_t)(d + ((uint64_t)t >> 32));
    return rd;
}

uint32_t HELPER(mhaccu)(CPURISCVState *env, uint32_t rs1,
          uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    uint32_t s1 = (uint32_t)rs1;
    uint32_t s2 = (uint32_t)rs2;
    uint32_t d = (uint32_t)dest;
    uint64_t t = (uint64_t)s1 * (uint64_t)s2;

    rd = (uint32_t)(d + ((uint64_t)t >> 32));
    return rd;
}

uint32_t HELPER(mhraccu)(CPURISCVState *env, uint32_t rs1,
          uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    uint32_t s1 = (uint32_t)rs1;
    uint32_t s2 = (uint32_t)rs2;
    uint32_t d = (uint32_t)dest;
    uint64_t t = (uint64_t)s1 * (uint64_t)s2 + (1LL << 31);

    rd = (uint32_t)(d + ((uint64_t)t >> 32));
    return rd;
}


uint64_t HELPER(pmhacc_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        int32_t rs2_w = (int32_t)(rs2 >> (i * 32));
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t t = (int64_t)rs1_w * (int64_t)rs2_w;

        rd |= (uint64_t)(uint32_t)(d_w + (int32_t)((uint64_t)t >> 32)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmhracc_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        int32_t rs2_w = (int32_t)(rs2 >> (i * 32));
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t t = (int64_t)rs1_w * (int64_t)rs2_w + (1LL << 31);

        rd |= (uint64_t)(uint32_t)(d_w + (int32_t)((uint64_t)t >> 32)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmhaccsu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t t = (int64_t)rs1_w * (uint64_t)rs2_w;

        rd |= (uint64_t)(uint32_t)(d_w + (int32_t)((uint64_t)t >> 32)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmhraccsu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int32_t rs1_w = (int32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t t = (int64_t)rs1_w * (uint64_t)rs2_w + (1LL << 31);

        rd |= (uint64_t)(uint32_t)(d_w + (int32_t)((uint64_t)t >> 32)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmhaccu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint32_t rs1_w = (uint32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        uint32_t d_w = (uint32_t)(dest >> (i * 32));
        uint64_t t = (uint64_t)rs1_w * (uint64_t)rs2_w;

        rd |= (uint64_t)(uint32_t)(d_w + ((uint64_t)t >> 32)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmhraccu_w)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint32_t rs1_w = (uint32_t)(rs1 >> (i * 32));
        uint32_t rs2_w = (uint32_t)(rs2 >> (i * 32));
        uint32_t d_w = (uint32_t)(dest >> (i * 32));
        uint64_t t = (uint64_t)rs1_w * (uint64_t)rs2_w + (1LL << 31);

        rd |= (uint64_t)(uint32_t)(d_w + ((uint64_t)t >> 32)) << (i * 32);
    }

    return rd;
}

uint32_t HELPER(mqacc_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int32_t d_h0 = (int32_t)dest;

    int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h0;
    rd = (d_h0 + (int32_t)(mul >> 15));
    return rd;
}

uint32_t HELPER(mqacc_h01)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int32_t d_h0 = (int32_t)dest;

    int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h1;
    rd = (d_h0 + (int32_t)(mul >> 15));
    return rd;
}

uint32_t HELPER(mqacc_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int32_t d_h0 = (int32_t)dest;

    int64_t mul = (int64_t)s1_h1 * (int64_t)s2_h1;
    rd = (d_h0 + (int32_t)(mul >> 15));
    return rd;
}

uint32_t HELPER(mqracc_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int32_t d_h0 = (int32_t)dest;

    int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h0 + (1LL << 14);
    rd = (d_h0 + (int32_t)(mul >> 15));
    return rd;
}

uint32_t HELPER(mqracc_h01)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int32_t d_h0 = (int32_t)dest;

    int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h1 + (1LL << 14);
    rd = (d_h0 + (int32_t)(mul >> 15));
    return rd;
}

uint32_t HELPER(mqracc_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
    uint32_t rd = 0;
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int32_t d_h0 = (int32_t)dest;

    int64_t mul = (int64_t)s1_h1 * (int64_t)s2_h1 + (1LL << 14);
    rd = (d_h0 + (int32_t)(mul >> 15));
    return rd;
}

uint64_t HELPER(pmqacc_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((rs2 >> (i * 32)) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h0;

        rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmqacc_w_h01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((rs2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmqacc_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)(((rs1 >> (i * 32)) >> 16) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((rs2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmqracc_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((rs2 >> (i * 32)) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h0 + (1LL << 14);

        rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmqracc_w_h01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((rs2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h1 + (1LL << 14);

        rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmqracc_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)(((rs1 >> (i * 32)) >> 16) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((rs2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
        int64_t mul = (int64_t)s1_h0 * (int64_t)s2_h1 + (1LL << 14);

        rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
    }

    return rd;
}

target_ulong HELPER(pmq2add_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
        int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)(((s1 >> (i * 32)) >> 16) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((s2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0;
        int64_t mul_00_47bit = ((int64_t)mul_00) >> 15;
        int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1;
        int64_t mul_11_47bit = ((int64_t)mul_11) >> 15;
        rd |= ((target_ulong)(uint32_t)(int32_t)(mul_00_47bit + mul_11_47bit)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pmqr2add_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
        int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)(((s1 >> (i * 32)) >> 16) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((s2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0 + (1LL << 14);
        int64_t mul_00_47bit = ((int64_t)mul_00) >> 15;
        int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1 + (1LL << 14);
        int64_t mul_11_47bit = ((int64_t)mul_11) >> 15;
        rd |= ((target_ulong)(uint32_t)(int32_t)(mul_00_47bit + mul_11_47bit)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pmq2adda_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
        int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)(((s1 >> (i * 32)) >> 16) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((s2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t d_h = (int32_t)(dest >> (i * 32));
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0;
        int64_t mul_00_47bit = ((int64_t)mul_00) >> 15;
        int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1;
        int64_t mul_11_47bit = ((int64_t)mul_11) >> 15;
        rd |= ((target_ulong)(uint32_t)((int32_t)(mul_00_47bit + mul_11_47bit) + d_h)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pmqr2adda_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
        int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)(((s1 >> (i * 32)) >> 16) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)(((s2 >> (i * 32)) >> 16) & 0xFFFF);
        int32_t d_h = (int32_t)(dest >> (i * 32));
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0 + (1LL << 14);
        int64_t mul_00_47bit = ((int64_t)mul_00) >> 15;
        int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1 + (1LL << 14);
        int64_t mul_11_47bit = ((int64_t)mul_11) >> 15;
        rd |= ((target_ulong)(uint32_t)((int32_t)(mul_00_47bit + mul_11_47bit) + d_h)) << (i * 32);
    }
    return rd;
}

uint64_t HELPER(mqacc_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
    int32_t s2_w0 = (int32_t)(rs2 & 0xFFFFFFFF);
    int64_t d = (int64_t)dest;
    int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w0;
    __int128_t mul_95bit = ((__int128_t)mul) >> 31;

    rd = ((uint64_t)(d + (int64_t)mul_95bit));
    return rd;
}

uint64_t HELPER(mqacc_w01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
    int64_t d = (int64_t)dest;
    int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w1;
    __int128_t mul_95bit = ((__int128_t)mul) >> 31;

    rd = ((uint64_t)(d + (int64_t)mul_95bit));
    return rd;
}

uint64_t HELPER(mqacc_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
    int32_t s1_w1 = (int32_t)((rs1 >> 32) & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
    int64_t d = (int64_t)dest;
    int64_t mul = (int64_t)s1_w1 * (int64_t)s2_w1;
    __int128_t mul_95bit = ((__int128_t)mul) >> 31;

    rd = ((uint64_t)(d + (int64_t)mul_95bit));
    return rd;
}

uint64_t HELPER(mqracc_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
    int32_t s2_w0 = (int32_t)(rs2 & 0xFFFFFFFF);
    int64_t d = (int64_t)dest;
    int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w0 + (1LL << 30);
    __int128_t mul_95bit = ((__int128_t)mul) >> 31;

    rd = ((uint64_t)(d + (int64_t)mul_95bit));
    return rd;
}

uint64_t HELPER(mqracc_w01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
    int64_t d = (int64_t)dest;
    int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w1 + (1LL << 30);
    __int128_t mul_95bit = ((__int128_t)mul) >> 31;

    rd = ((uint64_t)(d + (int64_t)mul_95bit));
    return rd;
}

uint64_t HELPER(mqracc_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
    int32_t s1_w1 = (int32_t)((rs1 >> 32) & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
    int64_t d = (int64_t)dest;
    int64_t mul = (int64_t)s1_w1 * (int64_t)s2_w1 + (1LL << 30);
    __int128_t mul_95bit = ((__int128_t)mul) >> 31;

    rd = ((uint64_t)(d + (int64_t)mul_95bit));
    return rd;
}

uint64_t HELPER(pmq2add_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;

    int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
    int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2  >> 32) & 0xFFFFFFFF);
     int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0;
    __int128_t mul_00_95bit = ((__int128_t)mul_00) >> 31;
    int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1;
    __int128_t mul_11_95bit = ((__int128_t)mul_11) >> 31;

    rd = ((uint64_t)(int64_t)(mul_00_95bit + mul_11_95bit));
    return rd;
}

uint64_t HELPER(pmqr2add_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;

    int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
    int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2  >> 32) & 0xFFFFFFFF);
     int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0 + (1LL << 30);
    __int128_t mul_00_95bit = ((__int128_t)mul_00) >> 31;
    int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1 + (1LL << 30);
    __int128_t mul_11_95bit = ((__int128_t)mul_11) >> 31;

    rd = ((uint64_t)(int64_t)(mul_00_95bit + mul_11_95bit));
    return rd;
}


uint64_t HELPER(pmq2adda_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;

    int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
    int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2  >> 32) & 0xFFFFFFFF);
    int64_t d_w = (int64_t)dest;
     int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0;
    __int128_t mul_00_95bit = ((__int128_t)mul_00) >> 31;
    int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1;
    __int128_t mul_11_95bit = ((__int128_t)mul_11) >> 31;

    rd = ((uint64_t)((int64_t)(mul_00_95bit + mul_11_95bit) + d_w));
    return rd;
}

uint64_t HELPER(pmqr2adda_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;

    int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
    int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2  >> 32) & 0xFFFFFFFF);
    int64_t d_w = (int64_t)dest;
     int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0 + (1 << 30);
    __int128_t mul_00_95bit = ((__int128_t)mul_00) >> 31;
    int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1 + (1 << 30);
    __int128_t mul_11_95bit = ((__int128_t)mul_11) >> 31;

    rd = ((uint64_t)((int64_t)(mul_00_95bit + mul_11_95bit) + d_w));
    return rd;
}

target_ulong HELPER(pmul_h_b00)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int8_t s1_b0 = (int8_t)((s1 >> (i * 16)) & 0xFF);
        int8_t s2_b0 = (int8_t)((s2 >> (i * 16)) & 0xFF);
        int16_t mul = (int16_t)s1_b0 * (int16_t)s2_b0;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmul_h_b01)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int8_t s1_b0 = (int8_t)((s1 >> (i * 16)) & 0xFF);
        int8_t s2_b1 = (int8_t)((s2 >> (i * 16 + 8)) & 0xFF);
        int16_t mul = (int16_t)s1_b0 * (int16_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmul_h_b11)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int8_t s1_b1 = (int8_t)((s1 >> (i * 16 + 8)) & 0xFF);
        int8_t s2_b1 = (int8_t)((s2 >> (i * 16 + 8)) & 0xFF);
        int16_t mul = (int16_t)s1_b1 * (int16_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulsu_h_b00)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int8_t s1_b0 = (int8_t)((s1 >> (i * 16)) & 0xFF);
        uint8_t s2_b0 = (uint8_t)((s2 >> (i * 16)) & 0xFF);
        int16_t mul = (int16_t)s1_b0 * (uint16_t)s2_b0;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulsu_h_b11)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        int8_t s1_b1 = (int8_t)((s1 >> (i * 16 + 8)) & 0xFF);
        uint8_t s2_b1 = (uint8_t)((s2 >> (i * 16 + 8)) & 0xFF);
        int16_t mul = (int16_t)s1_b1 * (uint16_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulu_h_b00)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        uint8_t s1_b0 = (uint8_t)((s1 >> (i * 16)) & 0xFF);
        uint8_t s2_b0 = (uint8_t)((s2 >> (i * 16)) & 0xFF);
        uint16_t mul = (uint16_t)s1_b0 * (uint16_t)s2_b0;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulu_h_b01)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        uint8_t s1_b0 = (uint8_t)((s1 >> (i * 16)) & 0xFF);
        uint8_t s2_b1 = (uint8_t)((s2 >> (i * 16 + 8)) & 0xFF);
        uint16_t mul = (uint16_t)s1_b0 * (uint16_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulu_h_b11)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
        uint8_t s1_b1 = (uint8_t)((s1 >> (i * 16 + 8)) & 0xFF);
        uint8_t s2_b1 = (uint8_t)((s2 >> (i * 16 + 8)) & 0xFF);
        uint16_t mul = (uint16_t)s1_b1 * (uint16_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)mul) << (i * 16);
    }
    return rd;
}

uint32_t HELPER(mul_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
	int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h0;
	rd = (int32_t)mul;
    return rd;
}

uint32_t HELPER(mul_h01)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
	int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h1;
	rd = (int32_t)mul;
    return rd;
}

uint32_t HELPER(mul_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int32_t mul = (int32_t)s1_h1 * (int32_t)s2_h1;
	rd = (int32_t)mul;
    return rd;
}

uint32_t HELPER(mulsu_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
	int32_t mul = (int32_t)s1_h0 * (uint32_t)s2_h0;
	rd = (int32_t)mul;
    return rd;
}

uint32_t HELPER(mulsu_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	int32_t mul = (int32_t)s1_h1 * (uint32_t)s2_h1;
	rd = (int32_t)mul;
    return rd;
}

uint32_t HELPER(mulu_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    uint16_t s1_h0 = (uint16_t)(rs1 & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
	uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h0;
	rd = (uint32_t)mul;
    return rd;
}

uint32_t HELPER(mulu_h01)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    uint16_t s1_h0 = (uint16_t)(rs1 & 0xFFFF);
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h1;
	rd = (uint32_t)mul;
    return rd;
}

uint32_t HELPER(mulu_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    uint16_t s1_h1 = (uint16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	uint32_t mul = (uint32_t)s1_h1 * (uint32_t)s2_h1;
	rd = (uint32_t)mul;
    return rd;
}

uint64_t HELPER(pmul_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((rs2 >> (i * 32)) & 0xFFFF);
        int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h0;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmul_w_h01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmul_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h1 = (int16_t)((rs1 >> (i * 32 + 16)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul = (int32_t)s1_h1 * (int32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulsu_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h0 = (uint16_t)((rs2 >> (i * 32)) & 0xFFFF);
        int32_t mul = (int32_t)s1_h0 * (uint32_t)s2_h0;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulsu_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h1 = (int16_t)((rs1 >> (i * 32 + 16)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul = (int32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulu_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint16_t s1_h0 = (uint16_t)((rs1 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h0 = (uint16_t)((rs2 >> (i * 32)) & 0xFFFF);
        uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h0;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulu_w_h01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint16_t s1_h0 = (uint16_t)((rs1 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
        uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmulu_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint16_t s1_h1 = (uint16_t)((rs1 >> (i * 32 + 16)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
        uint32_t mul = (uint32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
    }

    return rd;
}

uint32_t HELPER(macc_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h0;
	rd = (int32_t)(d_h + mul);
    return rd;
}

uint32_t HELPER(macc_h01)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
	int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h1;
	rd = (int32_t)(d_h + mul);
    return rd;
}

uint32_t HELPER(macc_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int32_t mul = (int32_t)s1_h1 * (int32_t)s2_h1;
	rd = (int32_t)(d_h + mul);
    return rd;
}

uint32_t HELPER(maccsu_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int32_t mul = (int32_t)s1_h0 * (uint32_t)s2_h0;
	rd = (int32_t)(d_h + mul);
    return rd;
}

uint32_t HELPER(maccsu_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int32_t mul = (int32_t)s1_h1 * (uint32_t)s2_h1;
	rd = (int32_t)(d_h + mul);
    return rd;
}

uint32_t HELPER(maccu_h00)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    uint16_t s1_h0 = (uint16_t)(rs1 & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
	uint32_t d_h = (uint32_t)dest;
	uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h0;
	rd = (uint32_t)(d_h + mul);
    return rd;
}

uint32_t HELPER(maccu_h01)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    uint16_t s1_h0 = (uint16_t)(rs1 & 0xFFFF);
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	uint32_t d_h = (uint32_t)dest;
	uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h1;
	rd = (uint32_t)(d_h + mul);
    return rd;
}

uint32_t HELPER(maccu_h11)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    uint16_t s1_h1 = (uint16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	uint32_t d_h = (uint32_t)dest;
	uint32_t mul = (uint32_t)s1_h1 * (uint32_t)s2_h1;
	rd = (uint32_t)(d_h + mul);
    return rd;
}

uint64_t HELPER(pmacc_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h0 = (int16_t)((rs2 >> (i * 32)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
        int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h0;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmacc_w_h01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
        int32_t mul = (int32_t)s1_h0 * (int32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmacc_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h1 = (int16_t)((rs1 >> (i * 32 + 16)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
        int32_t mul = (int32_t)s1_h1 * (int32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmaccsu_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h0 = (int16_t)((rs1 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h0 = (uint16_t)((rs2 >> (i * 32)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
        int32_t mul = (int32_t)s1_h0 * (uint32_t)s2_h0;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmaccsu_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        int16_t s1_h1 = (int16_t)((rs1 >> (i * 32 + 16)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
        int32_t mul = (int32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmaccu_w_h00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint16_t s1_h0 = (uint16_t)((rs1 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h0 = (uint16_t)((rs2 >> (i * 32)) & 0xFFFF);
		uint32_t d_h = (uint32_t)(dest >> (i * 32));
        uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h0;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmaccu_w_h01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint16_t s1_h0 = (uint16_t)((rs1 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 +16)) & 0xFFFF);
		uint32_t d_h = (uint32_t)(dest >> (i * 32));
        uint32_t mul = (uint32_t)s1_h0 * (uint32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

uint64_t HELPER(pmaccu_w_h11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

    for (int i = 0; i < 2; i++) {
        uint16_t s1_h1 = (uint16_t)((rs1 >> (i * 32 + 16)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		uint32_t d_h = (uint32_t)(dest >> (i * 32));
        uint32_t mul = (uint32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((uint64_t)(uint32_t)(d_h + mul)) << (i * 32);
    }

    return rd;
}

target_ulong HELPER(pm2add_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0;
		int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2addsu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		uint16_t s2_h0 = (uint16_t)((s2 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul_00 = (int32_t)s1_h0 * (uint32_t)s2_h0;
		int32_t mul_11 = (int32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2addu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		uint16_t s1_h0 = (uint16_t)((s1 >> (i * 32)) & 0xFFFF);
        uint16_t s1_h1 = (uint16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		uint16_t s2_h0 = (uint16_t)((s2 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        uint32_t mul_00 = (uint32_t)s1_h0 * (uint32_t)s2_h0;
		uint32_t mul_11 = (uint32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2add_hx)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul_01 = (int32_t)s1_h0 * (int32_t)s2_h1;
		int32_t mul_10 = (int32_t)s1_h1 * (int32_t)s2_h0;
        rd |= ((target_ulong)(uint32_t)(mul_01 + mul_10)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2sadd_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int32_t s1_check = (int32_t)((s1 >> (i * 32)) & 0xFFFFFFFF);
		int32_t s2_check = (int32_t)((s2 >> (i * 32)) & 0xFFFFFFFF);
		uint32_t result = 0;
		if((s1_check == 0x80008000) && (s2_check == 0x80008000)){
			result = 0x7FFFFFFF;
			env->vxsat = 1;
		}else{
			int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        	int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
			int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        	int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        	int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0;
			int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1;
			result = (uint32_t)(mul_00 + mul_11);
		}
        rd |= ((target_ulong)result) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2sadd_hx)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int32_t s1_check = (int32_t)((s1 >> (i * 32)) & 0xFFFFFFFF);
		int32_t s2_check = (int32_t)((s2 >> (i * 32)) & 0xFFFFFFFF);
		uint32_t result = 0;
		if((s1_check == 0x80008000) && (s2_check == 0x80008000)){
			result = 0x7FFFFFFF;
			env->vxsat = 1;
		}else{
			int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        	int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
			int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        	int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        	int32_t mul_01 = (int32_t)s1_h0 * (int32_t)s2_h1;
			int32_t mul_10 = (int32_t)s1_h1 * (int32_t)s2_h0;
			result = (uint32_t)(mul_01 + mul_10);
		}
        rd |= ((target_ulong)result) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2sub_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0;
		int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 - mul_11)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2sub_hx)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
        int32_t mul_10 = (int32_t)s1_h1 * (int32_t)s2_h0;
		int32_t mul_01 = (int32_t)s1_h0 * (int32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_10 - mul_01)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2adda_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_w = (int32_t)(dest >> (i * 32));
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0;
		int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11 + d_w)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2addasu_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		uint16_t s2_h0 = (uint16_t)((s2 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_w = (int32_t)(dest >> (i * 32));
        int32_t mul_00 = (int32_t)s1_h0 * (uint32_t)s2_h0;
		int32_t mul_11 = (int32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11 + d_w)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2addau_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		uint16_t s1_h0 = (uint16_t)((s1 >> (i * 32)) & 0xFFFF);
        uint16_t s1_h1 = (uint16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		uint16_t s2_h0 = (uint16_t)((s2 >> (i * 32)) & 0xFFFF);
        uint16_t s2_h1 = (uint16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
		uint32_t d_w = (uint32_t)(dest >> (i * 32));
        uint32_t mul_00 = (uint32_t)s1_h0 * (uint32_t)s2_h0;
		uint32_t mul_11 = (uint32_t)s1_h1 * (uint32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11 + d_w)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2adda_hx)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_w = (int32_t)(dest >> (i * 32));
        int32_t mul_01 = (int32_t)s1_h0 * (int32_t)s2_h1;
		int32_t mul_10 = (int32_t)s1_h1 * (int32_t)s2_h0;
        rd |= ((target_ulong)(uint32_t)(mul_01 + mul_10 + d_w)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2suba_h)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_w = (int32_t)(dest >> (i * 32));
        int32_t mul_00 = (int32_t)s1_h0 * (int32_t)s2_h0;
		int32_t mul_11 = (int32_t)s1_h1 * (int32_t)s2_h1;
        rd |= ((target_ulong)(uint32_t)(mul_00 - mul_11 + d_w)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm2suba_hx)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int16_t s1_h0 = (int16_t)((s1 >> (i * 32)) & 0xFFFF);
        int16_t s1_h1 = (int16_t)((s1 >> (i * 32 + 16)) & 0xFFFF);
		int16_t s2_h0 = (int16_t)((s2 >> (i * 32)) & 0xFFFF);
        int16_t s2_h1 = (int16_t)((s2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_w = (int32_t)(dest >> (i * 32));
        int32_t mul_01 = (int32_t)s1_h0 * (int32_t)s2_h1;
		int32_t mul_10 = (int32_t)s1_h1 * (int32_t)s2_h0;
        rd |= ((target_ulong)(uint32_t)(mul_01 - mul_10 + d_w)) << (i * 32);
    }
    return rd;
}

uint64_t HELPER(mul_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(rs2 & 0xFFFFFFFF);
	int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w0;
	rd = (int64_t)mul;
    return rd;
}

uint64_t HELPER(mul_w01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
	int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
	int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w1;
	rd = (int64_t)mul;
    return rd;
}

uint64_t HELPER(mul_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
	uint64_t rd = 0;
    int32_t s1_w1 = (int32_t)((rs1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
	int64_t mul = (int64_t)s1_w1 * (int64_t)s2_w1;
	rd = (int64_t)mul;
    return rd;
}

uint64_t HELPER(mulsu_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(rs2 & 0xFFFFFFFF);
	int64_t mul = (int64_t)s1_w0 * (uint64_t)s2_w0;
	rd = (int64_t)mul;
    return rd;
}

uint64_t HELPER(mulsu_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
	uint64_t rd = 0;
    int32_t s1_w1 = (int32_t)((rs1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w1 = (uint32_t)((rs2 >> 32) & 0xFFFFFFFF);
	int64_t mul = (int64_t)s1_w1 * (uint64_t)s2_w1;
	rd = (int64_t)mul;
    return rd;
}

uint64_t HELPER(mulu_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t s1_w0 = (uint32_t)(rs1 & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(rs2 & 0xFFFFFFFF);
	uint64_t mul = (uint64_t)s1_w0 * (uint64_t)s2_w0;
	rd = (uint64_t)mul;
    return rd;
}

uint64_t HELPER(mulu_w01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t s1_w0 = (uint32_t)(rs1 & 0xFFFFFFFF);
	uint32_t s2_w1 = (uint32_t)((rs2 >> 32) & 0xFFFFFFFF);
	uint64_t mul = (uint64_t)s1_w0 * (uint64_t)s2_w1;
	rd = (uint64_t)mul;
    return rd;
}

uint64_t HELPER(mulu_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
    uint32_t s1_w1 = (uint32_t)((rs1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w1 = (uint32_t)((rs2 >> 32) & 0xFFFFFFFF);
	uint64_t mul = (uint64_t)s1_w1 * (uint64_t)s2_w1;
	rd = (uint64_t)mul;
    return rd;
}

uint64_t HELPER(macc_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(rs2 & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
	int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w0;
	rd = (int64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(macc_w01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
	int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
	int64_t mul = (int64_t)s1_w0 * (int64_t)s2_w1;
	rd = (int64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(macc_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    int32_t s1_w1 = (int32_t)((rs1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w1 = (int32_t)((rs2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
	int64_t mul = (int64_t)s1_w1 * (int64_t)s2_w1;
	rd = (int64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(maccsu_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    int32_t s1_w0 = (int32_t)(rs1 & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(rs2 & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
	int64_t mul = (int64_t)s1_w0 * (uint64_t)s2_w0;
	rd = (int64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(maccsu_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    int32_t s1_w1 = (int32_t)((rs1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w1 = (uint32_t)((rs2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
	int64_t mul = (int64_t)s1_w1 * (uint64_t)s2_w1;
	rd = (int64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(maccu_w00)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    uint32_t s1_w0 = (uint32_t)(rs1 & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(rs2 & 0xFFFFFFFF);
	uint64_t d_w = (uint64_t)dest;
	uint64_t mul = (uint64_t)s1_w0 * (uint64_t)s2_w0;
	rd = (uint64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(maccu_w01)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    uint32_t s1_w0 = (uint32_t)(rs1 & 0xFFFFFFFF);
	uint32_t s2_w1 = (uint32_t)((rs2 >> 32) & 0xFFFFFFFF);
	uint64_t d_w = (uint64_t)dest;
	uint64_t mul = (uint64_t)s1_w0 * (uint64_t)s2_w1;
	rd = (uint64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(maccu_w11)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
	uint64_t rd = 0;
    uint32_t s1_w1 = (uint32_t)((rs1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w1 = (uint32_t)((rs2 >> 32) & 0xFFFFFFFF);
	uint64_t d_w = (uint64_t)dest;
	uint64_t mul = (uint64_t)s1_w1 * (uint64_t)s2_w1;
	rd = (uint64_t)(d_w + mul);
    return rd;
}

uint64_t HELPER(pm2add_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
    int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0;
	int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1;
    rd = (uint64_t)(mul_00 + mul_11);
    return rd;
}

uint64_t HELPER(pm2addsu_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(s2 & 0xFFFFFFFF);
    uint32_t s2_w1 = (uint32_t)((s2 >> 32) & 0xFFFFFFFF);
    int64_t mul_00 = (int64_t)s1_w0 * (uint64_t)s2_w0;
	int64_t mul_11 = (int64_t)s1_w1 * (uint64_t)s2_w1;
    rd = (uint64_t)(mul_00 + mul_11);
    return rd;
}

uint64_t HELPER(pm2addu_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	uint32_t s1_w0 = (uint32_t)(s1 & 0xFFFFFFFF);
    uint32_t s1_w1 = (uint32_t)((s1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(s2 & 0xFFFFFFFF);
    uint32_t s2_w1 = (uint32_t)((s2 >> 32) & 0xFFFFFFFF);
    uint64_t mul_00 = (uint64_t)s1_w0 * (uint64_t)s2_w0;
	uint64_t mul_11 = (uint64_t)s1_w1 * (uint64_t)s2_w1;
    rd = (uint64_t)(mul_00 + mul_11);
    return rd;
}

uint64_t HELPER(pm2add_wx)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
    int64_t mul_01 = (int64_t)s1_w0 * (int64_t)s2_w1;
	int64_t mul_10 = (int64_t)s1_w1 * (int64_t)s2_w0;
    rd = (uint64_t)(mul_01 + mul_10);
    return rd;
}

uint64_t HELPER(pm2sub_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
    int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0;
	int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1;
    rd = (uint64_t)(mul_00 - mul_11);
    return rd;
}

uint64_t HELPER(pm2sub_wx)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
    int64_t mul_10 = (int64_t)s1_w1 * (int64_t)s2_w0;
	int64_t mul_01 = (int64_t)s1_w0 * (int64_t)s2_w1;
    rd = (uint64_t)(mul_10 - mul_01);
    return rd;
}


uint64_t HELPER(pm2adda_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
    int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0;
	int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1;
    rd = (uint64_t)(mul_00 + mul_11 + d_w);
    return rd;
}

uint64_t HELPER(pm2addasu_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(s2 & 0xFFFFFFFF);
    uint32_t s2_w1 = (uint32_t)((s2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
    int64_t mul_00 = (int64_t)s1_w0 * (uint64_t)s2_w0;
	int64_t mul_11 = (int64_t)s1_w1 * (uint64_t)s2_w1;
    rd = (uint64_t)(mul_00 + mul_11 + d_w);
    return rd;
}

uint64_t HELPER(pm2addau_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	uint32_t s1_w0 = (uint32_t)(s1 & 0xFFFFFFFF);
    uint32_t s1_w1 = (uint32_t)((s1 >> 32) & 0xFFFFFFFF);
	uint32_t s2_w0 = (uint32_t)(s2 & 0xFFFFFFFF);
    uint32_t s2_w1 = (uint32_t)((s2 >> 32) & 0xFFFFFFFF);
	uint64_t d_w = (uint64_t)dest;
    uint64_t mul_00 = (uint64_t)s1_w0 * (uint64_t)s2_w0;
	uint64_t mul_11 = (uint64_t)s1_w1 * (uint64_t)s2_w1;
    rd = (uint64_t)(mul_00 + mul_11 + d_w);
    return rd;
}

uint64_t HELPER(pm2adda_wx)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
    int64_t mul_01 = (int64_t)s1_w0 * (int64_t)s2_w1;
	int64_t mul_10 = (int64_t)s1_w1 * (int64_t)s2_w0;
    rd = (uint64_t)(mul_01 + mul_10 + d_w);
    return rd;
}

uint64_t HELPER(pm2suba_w)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
    int64_t mul_00 = (int64_t)s1_w0 * (int64_t)s2_w0;
	int64_t mul_11 = (int64_t)s1_w1 * (int64_t)s2_w1;
    rd = (uint64_t)(mul_00 - mul_11 + d_w);
    return rd;
}

uint64_t HELPER(pm2suba_wx)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	int32_t s1_w0 = (int32_t)(s1 & 0xFFFFFFFF);
    int32_t s1_w1 = (int32_t)((s1 >> 32) & 0xFFFFFFFF);
	int32_t s2_w0 = (int32_t)(s2 & 0xFFFFFFFF);
    int32_t s2_w1 = (int32_t)((s2 >> 32) & 0xFFFFFFFF);
	int64_t d_w = (int64_t)dest;
    int64_t mul_01 = (int64_t)s1_w0 * (int64_t)s2_w1;
	int64_t mul_10 = (int64_t)s1_w1 * (int64_t)s2_w0;
    rd = (uint64_t)(mul_01 - mul_10 + d_w);
    return rd;
}

target_ulong HELPER(pm4add_b)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int8_t s1_b0 = (int8_t)((s1 >> (i * 32)) & 0xFF);
        int8_t s1_b1 = (int8_t)((s1 >> (i * 32 + 8)) & 0xFF);
		int8_t s1_b2 = (int8_t)((s1 >> (i * 32 + 16)) & 0xFF);
		int8_t s1_b3 = (int8_t)((s1 >> (i * 32 + 24)) & 0xFF);
		int8_t s2_b0 = (int8_t)((s2 >> (i * 32)) & 0xFF);
		int8_t s2_b1 = (int8_t)((s2 >> (i * 32 + 8)) & 0xFF);
        int8_t s2_b2 = (int8_t)((s2 >> (i * 32 + 16)) & 0xFF);
		int8_t s2_b3 = (int8_t)((s2 >> (i * 32 + 24)) & 0xFF);
        int32_t mul_00 = (int32_t)s1_b0 * (int32_t)s2_b0;
		int32_t mul_11 = (int32_t)s1_b1 * (int32_t)s2_b1;
		int32_t mul_22 = (int32_t)s1_b2 * (int32_t)s2_b2;
		int32_t mul_33 = (int32_t)s1_b3 * (int32_t)s2_b3;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11 + mul_22 + mul_33)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm4addsu_b)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int8_t s1_b0 = (int8_t)((s1 >> (i * 32)) & 0xFF);
        int8_t s1_b1 = (int8_t)((s1 >> (i * 32 + 8)) & 0xFF);
		int8_t s1_b2 = (int8_t)((s1 >> (i * 32 + 16)) & 0xFF);
		int8_t s1_b3 = (int8_t)((s1 >> (i * 32 + 24)) & 0xFF);
		uint8_t s2_b0 = (uint8_t)((s2 >> (i * 32)) & 0xFF);
		uint8_t s2_b1 = (uint8_t)((s2 >> (i * 32 + 8)) & 0xFF);
        uint8_t s2_b2 = (uint8_t)((s2 >> (i * 32 + 16)) & 0xFF);
		uint8_t s2_b3 = (uint8_t)((s2 >> (i * 32 + 24)) & 0xFF);
        int32_t mul_00 = (int32_t)s1_b0 * (uint32_t)s2_b0;
		int32_t mul_11 = (int32_t)s1_b1 * (uint32_t)s2_b1;
		int32_t mul_22 = (int32_t)s1_b2 * (uint32_t)s2_b2;
		int32_t mul_33 = (int32_t)s1_b3 * (uint32_t)s2_b3;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11 + mul_22 + mul_33)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm4addu_b)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		uint8_t s1_b0 = (uint8_t)((s1 >> (i * 32)) & 0xFF);
        uint8_t s1_b1 = (uint8_t)((s1 >> (i * 32 + 8)) & 0xFF);
		uint8_t s1_b2 = (uint8_t)((s1 >> (i * 32 + 16)) & 0xFF);
		uint8_t s1_b3 = (uint8_t)((s1 >> (i * 32 + 24)) & 0xFF);
		uint8_t s2_b0 = (uint8_t)((s2 >> (i * 32)) & 0xFF);
		uint8_t s2_b1 = (uint8_t)((s2 >> (i * 32 + 8)) & 0xFF);
        uint8_t s2_b2 = (uint8_t)((s2 >> (i * 32 + 16)) & 0xFF);
		uint8_t s2_b3 = (uint8_t)((s2 >> (i * 32 + 24)) & 0xFF);
        uint32_t mul_00 = (uint32_t)s1_b0 * (uint32_t)s2_b0;
		uint32_t mul_11 = (uint32_t)s1_b1 * (uint32_t)s2_b1;
		uint32_t mul_22 = (uint32_t)s1_b2 * (uint32_t)s2_b2;
		uint32_t mul_33 = (uint32_t)s1_b3 * (uint32_t)s2_b3;
        rd |= ((target_ulong)(uint32_t)(mul_00 + mul_11 + mul_22 + mul_33)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm4adda_b)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int8_t s1_b0 = (int8_t)((s1 >> (i * 32)) & 0xFF);
        int8_t s1_b1 = (int8_t)((s1 >> (i * 32 + 8)) & 0xFF);
		int8_t s1_b2 = (int8_t)((s1 >> (i * 32 + 16)) & 0xFF);
		int8_t s1_b3 = (int8_t)((s1 >> (i * 32 + 24)) & 0xFF);
		int8_t s2_b0 = (int8_t)((s2 >> (i * 32)) & 0xFF);
		int8_t s2_b1 = (int8_t)((s2 >> (i * 32 + 8)) & 0xFF);
        int8_t s2_b2 = (int8_t)((s2 >> (i * 32 + 16)) & 0xFF);
		int8_t s2_b3 = (int8_t)((s2 >> (i * 32 + 24)) & 0xFF);
		int32_t	d_b = (int32_t)(dest >> (i * 32));
        int32_t mul_00 = (int32_t)s1_b0 * (int32_t)s2_b0;
		int32_t mul_11 = (int32_t)s1_b1 * (int32_t)s2_b1;
		int32_t mul_22 = (int32_t)s1_b2 * (int32_t)s2_b2;
		int32_t mul_33 = (int32_t)s1_b3 * (int32_t)s2_b3;
        rd |= ((target_ulong)(uint32_t)(d_b + mul_00 + mul_11 + mul_22 + mul_33)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm4addasu_b)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		int8_t s1_b0 = (int8_t)((s1 >> (i * 32)) & 0xFF);
        int8_t s1_b1 = (int8_t)((s1 >> (i * 32 + 8)) & 0xFF);
		int8_t s1_b2 = (int8_t)((s1 >> (i * 32 + 16)) & 0xFF);
		int8_t s1_b3 = (int8_t)((s1 >> (i * 32 + 24)) & 0xFF);
		uint8_t s2_b0 = (uint8_t)((s2 >> (i * 32)) & 0xFF);
		uint8_t s2_b1 = (uint8_t)((s2 >> (i * 32 + 8)) & 0xFF);
        uint8_t s2_b2 = (uint8_t)((s2 >> (i * 32 + 16)) & 0xFF);
		uint8_t s2_b3 = (uint8_t)((s2 >> (i * 32 + 24)) & 0xFF);
		int32_t	d_b = (int32_t)(dest >> (i * 32));
        int32_t mul_00 = (int32_t)s1_b0 * (uint32_t)s2_b0;
		int32_t mul_11 = (int32_t)s1_b1 * (uint32_t)s2_b1;
		int32_t mul_22 = (int32_t)s1_b2 * (uint32_t)s2_b2;
		int32_t mul_33 = (int32_t)s1_b3 * (uint32_t)s2_b3;
        rd |= ((target_ulong)(uint32_t)(d_b + mul_00 + mul_11 + mul_22 + mul_33)) << (i * 32);
    }
    return rd;
}

target_ulong HELPER(pm4addau_b)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 4; i++) {
		uint8_t s1_b0 = (uint8_t)((s1 >> (i * 32)) & 0xFF);
        uint8_t s1_b1 = (uint8_t)((s1 >> (i * 32 + 8)) & 0xFF);
		uint8_t s1_b2 = (uint8_t)((s1 >> (i * 32 + 16)) & 0xFF);
		uint8_t s1_b3 = (uint8_t)((s1 >> (i * 32 + 24)) & 0xFF);
		uint8_t s2_b0 = (uint8_t)((s2 >> (i * 32)) & 0xFF);
		uint8_t s2_b1 = (uint8_t)((s2 >> (i * 32 + 8)) & 0xFF);
        uint8_t s2_b2 = (uint8_t)((s2 >> (i * 32 + 16)) & 0xFF);
		uint8_t s2_b3 = (uint8_t)((s2 >> (i * 32 + 24)) & 0xFF);
		uint32_t d_b = (uint32_t)(dest >> (i * 32));
        uint32_t mul_00 = (uint32_t)s1_b0 * (uint32_t)s2_b0;
		uint32_t mul_11 = (uint32_t)s1_b1 * (uint32_t)s2_b1;
		uint32_t mul_22 = (uint32_t)s1_b2 * (uint32_t)s2_b2;
		uint32_t mul_33 = (uint32_t)s1_b3 * (uint32_t)s2_b3;
        rd |= ((target_ulong)(uint32_t)(d_b + mul_00 + mul_11 + mul_22 + mul_33)) << (i * 32);
    }
    return rd;
}

uint64_t HELPER(pm4add_h)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	int16_t s1_h0 = (int16_t)(s1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((s1 >> 16) & 0xFFFF);
	int16_t s1_h2 = (int16_t)((s1 >> 32) & 0xFFFF);
	int16_t s1_h3 = (int16_t)((s1 >> 48) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(s2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((s2 >> 16) & 0xFFFF);
    int16_t s2_h2 = (int16_t)((s2 >> 32) & 0xFFFF);
	int16_t s2_h3 = (int16_t)((s2 >> 48) & 0xFFFF);
    int64_t mul_00 = (int64_t)s1_h0 * (int64_t)s2_h0;
	int64_t mul_11 = (int64_t)s1_h1 * (int64_t)s2_h1;
	int64_t mul_22 = (int64_t)s1_h2 * (int64_t)s2_h2;
	int64_t mul_33 = (int64_t)s1_h3 * (int64_t)s2_h3;
    rd = (uint64_t)(mul_00 + mul_11 + mul_22 + mul_33);
    return rd;
}

uint64_t HELPER(pm4addsu_h)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	int16_t s1_h0 = (int16_t)(s1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((s1 >> 16) & 0xFFFF);
	int16_t s1_h2 = (int16_t)((s1 >> 32) & 0xFFFF);
	int16_t s1_h3 = (int16_t)((s1 >> 48) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(s2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((s2 >> 16) & 0xFFFF);
    uint16_t s2_h2 = (uint16_t)((s2 >> 32) & 0xFFFF);
	uint16_t s2_h3 = (uint16_t)((s2 >> 48) & 0xFFFF);
    int64_t mul_00 = (int64_t)s1_h0 * (uint64_t)s2_h0;
	int64_t mul_11 = (int64_t)s1_h1 * (uint64_t)s2_h1;
	int64_t mul_22 = (int64_t)s1_h2 * (uint64_t)s2_h2;
	int64_t mul_33 = (int64_t)s1_h3 * (uint64_t)s2_h3;
    rd = (uint64_t)(mul_00 + mul_11 + mul_22 + mul_33);
    return rd;
}

uint64_t HELPER(pm4addu_h)(CPURISCVState *env, uint64_t s1,
    uint64_t s2)
{
    uint64_t rd = 0;
	uint16_t s1_h0 = (uint16_t)(s1 & 0xFFFF);
    uint16_t s1_h1 = (uint16_t)((s1 >> 16) & 0xFFFF);
	uint16_t s1_h2 = (uint16_t)((s1 >> 32) & 0xFFFF);
	uint16_t s1_h3 = (uint16_t)((s1 >> 48) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(s2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((s2 >> 16) & 0xFFFF);
    uint16_t s2_h2 = (uint16_t)((s2 >> 32) & 0xFFFF);
	uint16_t s2_h3 = (uint16_t)((s2 >> 48) & 0xFFFF);
    uint64_t mul_00 = (uint64_t)s1_h0 * (uint64_t)s2_h0;
	uint64_t mul_11 = (uint64_t)s1_h1 * (uint64_t)s2_h1;
	uint64_t mul_22 = (uint64_t)s1_h2 * (uint64_t)s2_h2;
	uint64_t mul_33 = (uint64_t)s1_h3 * (uint64_t)s2_h3;
    rd = (uint64_t)(mul_00 + mul_11 + mul_22 + mul_33);
    return rd;
}

uint64_t HELPER(pm4adda_h)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	int16_t s1_h0 = (int16_t)(s1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((s1 >> 16) & 0xFFFF);
	int16_t s1_h2 = (int16_t)((s1 >> 32) & 0xFFFF);
	int16_t s1_h3 = (int16_t)((s1 >> 48) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(s2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((s2 >> 16) & 0xFFFF);
    int16_t s2_h2 = (int16_t)((s2 >> 32) & 0xFFFF);
	int16_t s2_h3 = (int16_t)((s2 >> 48) & 0xFFFF);
	int64_t d_h = (int64_t)dest;
    int64_t mul_00 = (int64_t)s1_h0 * (int64_t)s2_h0;
	int64_t mul_11 = (int64_t)s1_h1 * (int64_t)s2_h1;
	int64_t mul_22 = (int64_t)s1_h2 * (int64_t)s2_h2;
	int64_t mul_33 = (int64_t)s1_h3 * (int64_t)s2_h3;
    rd = (uint64_t)(mul_00 + mul_11 + mul_22 + mul_33 + d_h);
    return rd;
}

uint64_t HELPER(pm4addasu_h)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	int16_t s1_h0 = (int16_t)(s1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((s1 >> 16) & 0xFFFF);
	int16_t s1_h2 = (int16_t)((s1 >> 32) & 0xFFFF);
	int16_t s1_h3 = (int16_t)((s1 >> 48) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(s2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((s2 >> 16) & 0xFFFF);
    uint16_t s2_h2 = (uint16_t)((s2 >> 32) & 0xFFFF);
	uint16_t s2_h3 = (uint16_t)((s2 >> 48) & 0xFFFF);
	int64_t d_h = (int64_t)dest;
    int64_t mul_00 = (int64_t)s1_h0 * (uint64_t)s2_h0;
	int64_t mul_11 = (int64_t)s1_h1 * (uint64_t)s2_h1;
	int64_t mul_22 = (int64_t)s1_h2 * (uint64_t)s2_h2;
	int64_t mul_33 = (int64_t)s1_h3 * (uint64_t)s2_h3;
    rd = (uint64_t)(mul_00 + mul_11 + mul_22 + mul_33 + d_h);
    return rd;
}

uint64_t HELPER(pm4addau_h)(CPURISCVState *env, uint64_t s1,
    uint64_t s2, uint64_t dest)
{
    uint64_t rd = 0;
	uint16_t s1_h0 = (uint16_t)(s1 & 0xFFFF);
    uint16_t s1_h1 = (uint16_t)((s1 >> 16) & 0xFFFF);
	uint16_t s1_h2 = (uint16_t)((s1 >> 32) & 0xFFFF);
	uint16_t s1_h3 = (uint16_t)((s1 >> 48) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(s2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((s2 >> 16) & 0xFFFF);
    uint16_t s2_h2 = (uint16_t)((s2 >> 32) & 0xFFFF);
	uint16_t s2_h3 = (uint16_t)((s2 >> 48) & 0xFFFF);
	uint64_t d_h = (uint64_t)dest;
    uint64_t mul_00 = (uint64_t)s1_h0 * (uint64_t)s2_h0;
	uint64_t mul_11 = (uint64_t)s1_h1 * (uint64_t)s2_h1;
	uint64_t mul_22 = (uint64_t)s1_h2 * (uint64_t)s2_h2;
	uint64_t mul_33 = (uint64_t)s1_h3 * (uint64_t)s2_h3;
    rd = (uint64_t)(mul_00 + mul_11 + mul_22 + mul_33 + d_h);
    return rd;
}

target_ulong HELPER(pmulh_h_b0)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		int8_t s2_b0 = (int8_t)((s2 >> (i * 16)) & 0xFF);
		int32_t mul = (int32_t)s1_h * (int32_t)s2_b0;
        rd |= ((target_ulong)(uint16_t)(int16_t)((mul >> 8) & 0xFFFF)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulh_h_b1)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		int8_t s2_b1 = (int8_t)((s2 >> (i * 16 + 8)) & 0xFF);
		int32_t mul = (int32_t)s1_h * (int32_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)(int16_t)((mul >> 8) & 0xFFFF)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulhsu_h_b0)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		uint8_t s2_b0 = (uint8_t)((s2 >> (i * 16)) & 0xFF);
		int32_t mul = (int32_t)s1_h * (uint32_t)s2_b0;
        rd |= ((target_ulong)(uint16_t)(int16_t)((mul >> 8) & 0xFFFF)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmulhsu_h_b1)(CPURISCVState *env, target_ulong s1,
    target_ulong s2)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		uint8_t s2_b1 = (uint8_t)((s2 >> (i * 16 + 8)) & 0xFF);
		int32_t mul = (int32_t)s1_h * (uint32_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)(int16_t)((mul >> 8) & 0xFFFF)) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhacc_h_b0)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		int8_t s2_b0 = (int8_t)((s2 >> (i * 16)) & 0xFF);
		int16_t	d_h = (int16_t)((dest >> (i * 16)) & 0xFFFF);
		int32_t mul = (int32_t)s1_h * (int32_t)s2_b0;
        rd |= ((target_ulong)(uint16_t)(d_h + (int16_t)((mul >> 8) & 0xFFFF))) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhacc_h_b1)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		int8_t s2_b1 = (int8_t)((s2 >> (i * 16 + 8)) & 0xFF);
		int16_t	d_h = (int16_t)((dest >> (i * 16)) & 0xFFFF);
		int32_t mul = (int32_t)s1_h * (int32_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)(d_h + (int16_t)((mul >> 8) & 0xFFFF))) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhaccsu_h_b0)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		uint8_t s2_b0 = (uint8_t)((s2 >> (i * 16)) & 0xFF);
		int16_t	d_h = (int16_t)((dest >> (i * 16)) & 0xFFFF);
		int32_t mul = (int32_t)s1_h * (uint32_t)s2_b0;
        rd |= ((target_ulong)(uint16_t)(d_h + (int16_t)((mul >> 8) & 0xFFFF))) << (i * 16);
    }
    return rd;
}

target_ulong HELPER(pmhaccsu_h_b1)(CPURISCVState *env, target_ulong s1,
    target_ulong s2, target_ulong dest)
{
    target_ulong rd = 0;

    for (int i = 0; i < sizeof(target_ulong) / 2; i++) {
		int16_t s1_h = (int16_t)((s1 >> (i * 16)) & 0xFFFF);
		uint8_t s2_b1 = (uint8_t)((s2 >> (i * 16 + 8)) & 0xFF);
		int16_t	d_h = (int16_t)((dest >> (i * 16)) & 0xFFFF);
		int32_t mul = (int32_t)s1_h * (uint32_t)s2_b1;
        rd |= ((target_ulong)(uint16_t)(d_h + (int16_t)((mul >> 8) & 0xFFFF))) << (i * 16);
    }
    return rd;
}

uint32_t HELPER(mulh_h0)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
	int64_t mul = (int64_t)s1 * (int64_t)s2_h0;
	rd = (int32_t)(mul >> 16);
    return rd;
}

uint32_t HELPER(mulh_h1)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int64_t mul = (int64_t)s1 * (int64_t)s2_h1;
	rd = (int32_t)(mul >> 16);
    return rd;
}

uint32_t HELPER(mulhsu_h0)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
	int64_t mul = (int64_t)s1 * (uint64_t)s2_h0;
	rd = (int32_t)(mul >> 16);
    return rd;
}

uint32_t HELPER(mulhsu_h1)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	int64_t mul = (int64_t)s1 * (uint64_t)s2_h1;
	rd = (int32_t)(mul >> 16);
    return rd;
}

uint64_t HELPER(pmulh_w_h0)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		int16_t s2_h0 = (int16_t)((rs2 >> (i * 32)) & 0xFFFF);
		int64_t mul = (int64_t)s1 * (int64_t)s2_h0;
		rd |= ((uint64_t)(uint32_t)(mul >> 16)) << (i * 32);
	}
    return rd;
}

uint64_t HELPER(pmulh_w_h1)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		int16_t s2_h1 = (int16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		int64_t mul = (int64_t)s1 * (int64_t)s2_h1;
		rd |= ((uint64_t)(uint32_t)(mul >> 16)) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pmulhsu_w_h0)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		uint16_t s2_h0 = (uint16_t)((rs2 >> (i * 32)) & 0xFFFF);
		int64_t mul = (int64_t)s1 * (uint64_t)s2_h0;
		rd |= ((uint64_t)(uint32_t)(mul >> 16)) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pmulhsu_w_h1)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		int64_t mul = (int64_t)s1 * (uint64_t)s2_h1;
		rd |= ((uint64_t)(uint32_t)(mul >> 16)) << (i * 32);
	}

    return rd;
}

uint32_t HELPER(mhacc_h0)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int64_t mul = (int64_t)s1 * (int64_t)s2_h0;
	rd = (int32_t)(d_h + (int32_t)(mul >> 16));
    return rd;
}

uint32_t HELPER(mhacc_h1)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int64_t mul = (int64_t)s1 * (int64_t)s2_h1;
	rd = (int32_t)(d_h + (int32_t)(mul >> 16));
    return rd;
}

uint32_t HELPER(mhaccsu_h0)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int64_t mul = (int64_t)s1 * (uint64_t)s2_h0;
	rd = (int32_t)(d_h + (int32_t)(mul >> 16));
    return rd;
}

uint32_t HELPER(mhaccsu_h1)(CPURISCVState *env, uint32_t rs1,
    uint32_t rs2, uint32_t dest)
{
	uint32_t rd = 0;
    int32_t s1 = (int32_t)rs1;
	uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	int32_t d_h = (int32_t)dest;
	int64_t mul = (int64_t)s1 * (uint64_t)s2_h1;
	rd = (int32_t)(d_h + (int32_t)(mul >> 16));
    return rd;
}

uint64_t HELPER(pmhacc_w_h0)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		int16_t s2_h0 = (int16_t)((rs2 >> (i * 32)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
		int64_t mul = (int64_t)s1 * (int64_t)s2_h0;
		rd |= ((uint64_t)(uint32_t)(d_h + (int32_t)(mul >> 16))) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pmhacc_w_h1)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		int16_t s2_h1 = (int16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
		int64_t mul = (int64_t)s1 * (int64_t)s2_h1;
		rd |= ((uint64_t)(uint32_t)(d_h + (int32_t)(mul >> 16))) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pmhaccsu_w_h0)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		uint16_t s2_h0 = (uint16_t)((rs2 >> (i * 32)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
		int64_t mul = (int64_t)s1 * (uint64_t)s2_h0;
		rd |= ((uint64_t)(uint32_t)(d_h + (int32_t)(mul >> 16))) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pmhaccsu_w_h1)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int32_t s1 = (int32_t)(rs1 >> (i * 32));
		uint16_t s2_h1 = (uint16_t)((rs2 >> (i * 32 + 16)) & 0xFFFF);
		int32_t d_h = (int32_t)(dest >> (i * 32));
		int64_t mul = (int64_t)s1 * (uint64_t)s2_h1;
		rd |= ((uint64_t)(uint32_t)(d_h + (int32_t)(mul >> 16))) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pmqwacc_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int16_t s1_h = (int16_t)((rs1 >> (i * 16)) & 0xFFFF);
		int16_t s2_h = (int16_t)((rs2 >> (i * 16)) & 0xFFFF);
		int32_t d_w = (int32_t)(dest >> (i * 32));
		int64_t mul = (int64_t)s1_h * (int64_t)s2_h;
		rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pmqrwacc_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int16_t s1_h = (int16_t)((rs1 >> (i * 16)) & 0xFFFF);
		int16_t s2_h = (int16_t)((rs2 >> (i * 16)) & 0xFFFF);
		int32_t d_w = (int32_t)(dest >> (i * 32));
		int64_t mul = (int64_t)s1_h * (int64_t)s2_h + (1LL << 14);
		rd |= ((uint64_t)(uint32_t)(d_w + (int32_t)(mul >> 15))) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(mqwacc)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	int64_t s1 = (int64_t)(int32_t)rs1;
	int64_t s2 = (int64_t)(int32_t)rs2;
	int64_t d_w = (int64_t)dest;
	__int128_t mul = (__int128_t)s1 * (__int128_t)s2;
	rd = (uint64_t)(d_w + (int64_t)(mul >> 31));

    return rd;
}

uint64_t HELPER(mqrwacc)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	int64_t s1 = (int64_t)(int32_t)rs1;
	int64_t s2 = (int64_t)(int32_t)rs2;
	int64_t d_w = (int64_t)dest;
	__int128_t mul = (__int128_t)s1 * (__int128_t)s2 + (1LL << 30);
	rd = (uint64_t)(d_w + (int64_t)(mul >> 31));
    return rd;
}

uint64_t HELPER(pwmul_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 4; i++) {
		int8_t s1_b = (int8_t)((rs1 >> (i * 8)) & 0xFF);
		int8_t s2_b = (int8_t)((rs2 >> (i * 8)) & 0xFF);
		int32_t mul = (int32_t)s1_b * (int32_t)s2_b;
		rd |= ((uint64_t)(uint16_t)(int16_t)mul) << (i * 16);
	}

    return rd;
}

uint64_t HELPER(pwmulsu_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 4; i++) {
		int8_t s1_b = (int8_t)((rs1 >> (i * 8)) & 0xFF);
		uint8_t s2_b = (uint8_t)((rs2 >> (i * 8)) & 0xFF);
		int32_t mul = (int32_t)s1_b * (uint32_t)s2_b;
		rd |= ((uint64_t)(uint16_t)(int16_t)mul) << (i * 16);
	}

    return rd;
}

uint64_t HELPER(pwmulu_b)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 4; i++) {
		uint8_t s1_b = (uint8_t)((rs1 >> (i * 8)) & 0xFF);
		uint8_t s2_b = (uint8_t)((rs2 >> (i * 8)) & 0xFF);
		uint32_t mul = (int32_t)s1_b * (uint32_t)s2_b;
		rd |= ((uint64_t)(uint16_t)mul) << (i * 16);
	}

    return rd;
}

uint64_t HELPER(pwmul_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int16_t s1_h = (int16_t)((rs1 >> (i * 16)) & 0xFFFF);
		int16_t s2_h = (int16_t)((rs2 >> (i * 16)) & 0xFFFF);
		int32_t mul = (int32_t)s1_h * (int32_t)s2_h;
		rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pwmulsu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int16_t s1_h = (int16_t)((rs1 >> (i * 16)) & 0xFFFF);
		uint16_t s2_h = (uint16_t)((rs2 >> (i * 16)) & 0xFFFF);
		int32_t mul = (int32_t)s1_h * (uint32_t)s2_h;
		rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pwmulu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		uint16_t s1_h = (uint16_t)((rs1 >> (i * 16)) & 0xFFFF);
		uint16_t s2_h = (uint16_t)((rs2 >> (i * 16)) & 0xFFFF);
		uint32_t mul = (uint32_t)s1_h * (uint32_t)s2_h;
		rd |= ((uint64_t)(uint32_t)mul) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pwmacc_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int16_t s1_h = (int16_t)((rs1 >> (i * 16)) & 0xFFFF);
		int16_t s2_h = (int16_t)((rs2 >> (i * 16)) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
		int32_t mul = (int32_t)s1_h * (int32_t)s2_h;
		rd |= ((uint64_t)(uint32_t)(d_w + mul)) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pwmaccsu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		int16_t s1_h = (int16_t)((rs1 >> (i * 16)) & 0xFFFF);
		uint16_t s2_h = (uint16_t)((rs2 >> (i * 16)) & 0xFFFF);
        int32_t d_w = (int32_t)(dest >> (i * 32));
		int32_t mul = (int32_t)s1_h * (uint32_t)s2_h;
		rd |= ((uint64_t)(uint32_t)(d_w + mul)) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(pwmaccu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	for(int i = 0; i < 2; i++) {
		uint16_t s1_h = (uint16_t)((rs1 >> (i * 16)) & 0xFFFF);
		uint16_t s2_h = (uint16_t)((rs2 >> (i * 16)) & 0xFFFF);
        uint32_t d_w = (uint32_t)(dest >> (i * 32));
		uint32_t mul = (uint32_t)s1_h * (uint32_t)s2_h;
		rd |= ((uint64_t)(uint32_t)(d_w + mul)) << (i * 32);
	}

    return rd;
}

uint64_t HELPER(wmul)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)((int64_t)(int32_t)rs1 * (int64_t)(int32_t)rs2);
}

uint64_t HELPER(wmulsu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)((int64_t)(int32_t)rs1 * (uint64_t)rs2);
}

uint64_t HELPER(wmulu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    return (uint64_t)rs1 * (uint64_t)rs2;
}

uint64_t HELPER(wmacc)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    return (uint64_t)((int64_t)(int32_t)rs1 * (int64_t)(int32_t)rs2 + (int64_t)dest);
}

uint64_t HELPER(wmaccsu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    return (uint64_t)((int64_t)(int32_t)rs1 * (uint64_t)rs2 + (int64_t)dest);
}

uint64_t HELPER(wmaccu)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    return (uint64_t)rs1 * (uint64_t)rs2 + (uint64_t)dest;
}

uint64_t HELPER(pm2wadd_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int64_t mul_00 = (int64_t)s1_h0 * (int64_t)s2_h0;
    int64_t mul_11 = (int64_t)s1_h1 * (int64_t)s2_h1;
	rd = ((uint64_t)(mul_00 + mul_11));

    return rd;
}

uint64_t HELPER(pm2waddsu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	int64_t mul_00 = (int64_t)s1_h0 * (uint64_t)s2_h0;
    int64_t mul_11 = (int64_t)s1_h1 * (uint64_t)s2_h1;
	rd = ((uint64_t)(mul_00 + mul_11));

    return rd;
}

uint64_t HELPER(pm2waddu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

	uint16_t s1_h0 = (uint16_t)(rs1 & 0xFFFF);
    uint16_t s1_h1 = (uint16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
	uint64_t mul_00 = (uint64_t)s1_h0 * (uint64_t)s2_h0;
    uint64_t mul_11 = (uint64_t)s1_h1 * (uint64_t)s2_h1;
	rd = ((uint64_t)(mul_00 + mul_11));

    return rd;
}

uint64_t HELPER(pm2wadda_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int64_t d_h = (int64_t)dest;
	int64_t mul_00 = (int64_t)s1_h0 * (int64_t)s2_h0;
    int64_t mul_11 = (int64_t)s1_h1 * (int64_t)s2_h1;
	rd = ((uint64_t)(d_h + mul_00 + mul_11));

    return rd;
}

uint64_t HELPER(pm2waddasu_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
    int64_t d_h = (int64_t)dest;
	int64_t mul_00 = (int64_t)s1_h0 * (uint64_t)s2_h0;
    int64_t mul_11 = (int64_t)s1_h1 * (uint64_t)s2_h1;
	rd = ((uint64_t)(d_h + mul_00 + mul_11));

    return rd;
}

uint64_t HELPER(pm2waddau_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;

	uint16_t s1_h0 = (uint16_t)(rs1 & 0xFFFF);
    uint16_t s1_h1 = (uint16_t)((rs1 >> 16) & 0xFFFF);
	uint16_t s2_h0 = (uint16_t)(rs2 & 0xFFFF);
    uint16_t s2_h1 = (uint16_t)((rs2 >> 16) & 0xFFFF);
    uint64_t d_h = (uint64_t)dest;
	uint64_t mul_00 = (uint64_t)s1_h0 * (uint64_t)s2_h0;
    uint64_t mul_11 = (uint64_t)s1_h1 * (uint64_t)s2_h1;
	rd = ((uint64_t)(d_h + mul_00 + mul_11));

    return rd;
}

uint64_t HELPER(pm2wadd_hx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int64_t mul_01 = (int64_t)s1_h0 * (int64_t)s2_h1;
    int64_t mul_10 = (int64_t)s1_h1 * (int64_t)s2_h0;
	rd = ((uint64_t)(mul_01 + mul_10));

    return rd;
}

uint64_t HELPER(pm2wadda_hx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
    uint64_t rd = 0;
	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s1_h2 = (int16_t)((rs1 >> 32) & 0xFFFF);
	int16_t s1_h3 = (int16_t)((rs1 >> 48) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int16_t s2_h2 = (int16_t)((rs2 >> 32) & 0xFFFF);
	int16_t s2_h3 = (int16_t)((rs2 >> 48) & 0xFFFF);
    int64_t d_h = (int64_t)dest;
    int64_t mul_01 = (int64_t)s1_h0 * (int64_t)s2_h1;
	int64_t mul_10 = (int64_t)s1_h1 * (int64_t)s2_h0;
	int64_t mul_23 = (int64_t)s1_h2 * (int64_t)s2_h3;
	int64_t mul_32 = (int64_t)s1_h3 * (int64_t)s2_h2;
    rd = (uint64_t)(d_h + mul_01 + mul_10 + mul_23 + mul_32);

    return rd;
}

uint64_t HELPER(pm2wsub_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int64_t mul_00 = (int64_t)s1_h0 * (int64_t)s2_h0;
    int64_t mul_11 = (int64_t)s1_h1 * (int64_t)s2_h1;
	rd = ((uint64_t)(mul_00 - mul_11));

    return rd;
}

uint64_t HELPER(pm2wsub_hx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2)
{
    uint64_t rd = 0;

	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
	int64_t mul_01 = (int64_t)s1_h0 * (int64_t)s2_h1;
    int64_t mul_10 = (int64_t)s1_h1 * (int64_t)s2_h0;
	rd = ((uint64_t)(mul_01 - mul_10));

    return rd;
}

uint64_t HELPER(pm2wsuba_h)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
   uint64_t rd = 0;
	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s1_h2 = (int16_t)((rs1 >> 32) & 0xFFFF);
	int16_t s1_h3 = (int16_t)((rs1 >> 48) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int16_t s2_h2 = (int16_t)((rs2 >> 32) & 0xFFFF);
	int16_t s2_h3 = (int16_t)((rs2 >> 48) & 0xFFFF);
    int64_t d_h = (int64_t)dest;
    int64_t mul_00 = (int64_t)s1_h0 * (int64_t)s2_h0;
	int64_t mul_11 = (int64_t)s1_h1 * (int64_t)s2_h1;
	int64_t mul_22 = (int64_t)s1_h2 * (int64_t)s2_h2;
	int64_t mul_33 = (int64_t)s1_h3 * (int64_t)s2_h3;
    rd = (uint64_t)(d_h + mul_00 - mul_11 + mul_22 - mul_33);

    return rd;
}

uint64_t HELPER(pm2wsuba_hx)(CPURISCVState *env, uint64_t rs1,
    uint64_t rs2, uint64_t dest)
{
   uint64_t rd = 0;
	int16_t s1_h0 = (int16_t)(rs1 & 0xFFFF);
    int16_t s1_h1 = (int16_t)((rs1 >> 16) & 0xFFFF);
	int16_t s1_h2 = (int16_t)((rs1 >> 32) & 0xFFFF);
	int16_t s1_h3 = (int16_t)((rs1 >> 48) & 0xFFFF);
	int16_t s2_h0 = (int16_t)(rs2 & 0xFFFF);
    int16_t s2_h1 = (int16_t)((rs2 >> 16) & 0xFFFF);
    int16_t s2_h2 = (int16_t)((rs2 >> 32) & 0xFFFF);
	int16_t s2_h3 = (int16_t)((rs2 >> 48) & 0xFFFF);
    int64_t d_h = (int64_t)dest;
    int64_t mul_01 = (int64_t)s1_h0 * (int64_t)s2_h1;
	int64_t mul_10 = (int64_t)s1_h1 * (int64_t)s2_h0;
	int64_t mul_23 = (int64_t)s1_h2 * (int64_t)s2_h3;
	int64_t mul_32 = (int64_t)s1_h3 * (int64_t)s2_h2;
    rd = (uint64_t)(d_h + mul_01 - mul_10 + mul_23 - mul_32);
	
	return rd;
}
