/* Copyright(c) 2015 Mikko Juola
 *
 * This file contains code from GROMACS molecular simulation package,
 * albeit quite modified to just include what we need in LSTM.
 *
 * The version this file came from was licensed under LGPL 2.1, below is
 * their license boilerplate:
 *
 * ================================================================
 * Copyright (c) 2012,2013, by the GROMACS development team, led by
 * David van der Spoel, Berk Hess, Erik Lindahl, and including many
 * others, as listed in the AUTHORS file in the top-level source
 * directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 * =====================================================================
 */

#ifndef mj_exp_avx_h
#define mj_exp_avx_h

#include <immintrin.h>
#include <fmaintrin.h>
#include <math.h>

static inline __m256d gmx_mm256_abs_pd(__m256d x)
{
    const __m256d signmask  = _mm256_castsi256_pd( _mm256_set_epi32(0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF,
                                                                    0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF) );

    return _mm256_and_pd(x, signmask);
}

/* 1.0/x, 256 bit wide */
static inline __m256d gmx_mm256_inv_pd(__m256d x)
{
    const __m256d two  = _mm256_set1_pd(2.0);

    /* Lookup instruction only exists in single precision, convert back and forth... */
    __m256d lu = _mm256_cvtps_pd(_mm_rcp_ps( _mm256_cvtpd_ps(x)));

    /* Perform two N-R steps for double precision */
    lu         = _mm256_mul_pd(lu, _mm256_sub_pd(two, _mm256_mul_pd(x, lu)));
    return _mm256_mul_pd(lu, _mm256_sub_pd(two, _mm256_mul_pd(x, lu)));
}

/* Exponential function, 256 bit. This could be calculated from 2^x as Exp(x)=2^(y),
 * where y=log2(e)*x, but there will then be a small rounding error since we lose
 * some precision due to the multiplication. This will then be magnified a lot by
 * the exponential.
 *
 * Instead, we calculate the fractional part directly as a PadÃ© approximation of
 * Exp(z) on [-0.5,0.5]. We use extended precision arithmetics to calculate the fraction
 * remaining after 2^y, which avoids the precision-loss.
 */
static __m256d gmx_mm256_exp_pd(__m256d exparg)
{
    const __m256d cutoff = { 708.39, 708.39, 708.39, 708.39 };
    if ( exparg[0] >= cutoff[0] || exparg[1] >= cutoff[1] || exparg[2] >= cutoff[2] || exparg[3] >= cutoff[3] ) {
        __m256d result = { exp(exparg[0]), exp(exparg[1]), exp(exparg[2]), exp(exparg[3]) };
        return result;
    }

    const __m256d argscale = _mm256_set1_pd(1.4426950408889634073599);
    /* Lower bound: We do not allow numbers that would lead to an IEEE fp representation exponent smaller than -126. */
    const __m256d arglimit = _mm256_set1_pd(1022.0);
    const __m128i expbase  = _mm_set1_epi32(1023);

    const __m256d invargscale0  = _mm256_set1_pd(6.93145751953125e-1);
    const __m256d invargscale1  = _mm256_set1_pd(1.42860682030941723212e-6);

    const __m256d P2       = _mm256_set1_pd(1.26177193074810590878e-4);
    const __m256d P1       = _mm256_set1_pd(3.02994407707441961300e-2);
    /* P0 == 1.0 */
    const __m256d Q3       = _mm256_set1_pd(3.00198505138664455042E-6);
    const __m256d Q2       = _mm256_set1_pd(2.52448340349684104192E-3);
    const __m256d Q1       = _mm256_set1_pd(2.27265548208155028766E-1);
    /* Q0 == 2.0 */
    const __m256d one      = _mm256_set1_pd(1.0);
    const __m256d two      = _mm256_set1_pd(2.0);

    __m256d       valuemask;
    __m256i       iexppart;
    __m128i       iexppart128a, iexppart128b;
    __m256d       fexppart;
    __m256d       intpart;
    __m256d       x, z, z2;
    __m256d       PolyP, PolyQ;

    x             = _mm256_mul_pd(exparg, argscale);

    iexppart128a  = _mm256_cvtpd_epi32(x);
    intpart       = _mm256_round_pd(x, _MM_FROUND_TO_NEAREST_INT);

    /* Add exponent bias */
    iexppart128a   = _mm_add_epi32(iexppart128a, expbase);

    /* We now want to shift the exponent 52 positions left, but to achieve this we need
     * to separate the 128-bit register data into two registers (4x64-bit > 128bit)
     * shift them, and then merge into a single __m256d.
     * Elements 0/1 should end up in iexppart128a, and 2/3 in iexppart128b.
     * It doesnt matter what we put in the 2nd/4th position, since that data will be
     * shifted out and replaced with zeros.
     */
    iexppart128b   = _mm_shuffle_epi32(iexppart128a, _MM_SHUFFLE(3, 3, 2, 2));
    iexppart128a   = _mm_shuffle_epi32(iexppart128a, _MM_SHUFFLE(1, 1, 0, 0));

    iexppart128b   = _mm_slli_epi64(iexppart128b, 52);
    iexppart128a   = _mm_slli_epi64(iexppart128a, 52);

    iexppart  = _mm256_castsi128_si256(iexppart128a);
    iexppart  = _mm256_insertf128_si256(iexppart, iexppart128b, 0x1);

    valuemask = _mm256_cmp_pd(arglimit, gmx_mm256_abs_pd(x), _CMP_GE_OQ);
    fexppart  = _mm256_and_pd(valuemask, _mm256_castsi256_pd(iexppart));

    z         = _mm256_sub_pd(exparg, _mm256_mul_pd(invargscale0, intpart));
    z         = _mm256_sub_pd(z, _mm256_mul_pd(invargscale1, intpart));

    z2        = _mm256_mul_pd(z, z);

    PolyQ     = _mm256_mul_pd(Q3, z2);
    PolyQ     = _mm256_add_pd(PolyQ, Q2);
    PolyP     = _mm256_mul_pd(P2, z2);
    PolyQ     = _mm256_mul_pd(PolyQ, z2);
    PolyP     = _mm256_add_pd(PolyP, P1);
    PolyQ     = _mm256_add_pd(PolyQ, Q1);
    PolyP     = _mm256_mul_pd(PolyP, z2);
    PolyQ     = _mm256_mul_pd(PolyQ, z2);
    PolyP     = _mm256_add_pd(PolyP, one);
    PolyQ     = _mm256_add_pd(PolyQ, two);

    PolyP     = _mm256_mul_pd(PolyP, z);

    z         = _mm256_mul_pd(PolyP, gmx_mm256_inv_pd(_mm256_sub_pd(PolyQ, PolyP)));
    z         = _mm256_add_pd(one, _mm256_mul_pd(two, z));

    z         = _mm256_mul_pd(z, fexppart);

    return z;
}

#endif

