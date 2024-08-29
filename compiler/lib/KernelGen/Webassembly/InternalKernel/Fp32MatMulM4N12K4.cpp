#include "WebAssembly/InternalKernel/InternalKernel.h"
#include "compiler/Common/Logger.h"
#include "../Utils/StringTemplate.h"
#include "MatmulCommon.h"
using namespace megcc;
using namespace KernelGen;
using namespace WebAssembly;
namespace {
std::string transpose_1x12_4_s() {
    return R"(
static inline void transpose_1x12_4_s(const float* inptr0, float* outptr) {
#ifdef __SSE__
    __m128 xmm0 = _mm_loadu_ps(inptr0);     //A0A1A2A3
    __m128 xmm1 = _mm_loadu_ps(inptr0 + 4); //B0B1B2B3
    __m128 xmm2 = _mm_loadu_ps(inptr0 + 8); //C0C1C2C3
    __m128 xmm3 = _mm_loadu_ps(inptr0 + 12); //D0D1D2D3
    
    __m128 xmm4 = _mm_loadu_ps(inptr0 + 16); //E0E1E2E3
    __m128 xmm5 = _mm_loadu_ps(inptr0 + 20); //F0F1F2F3
    __m128 xmm6 = _mm_loadu_ps(inptr0 + 24); //G0G1G2G3
    __m128 xmm7 = _mm_loadu_ps(inptr0 + 28); //H0H1H2H3
    
    __m128 xmm8 = _mm_loadu_ps(inptr0 + 32); //A0A1A2A3
    __m128 xmm9 = _mm_loadu_ps(inptr0 + 36); //A0A1A2A3
    __m128 xmm10 = _mm_loadu_ps(inptr0 + 40); //A0A1A2A3
    __m128 xmm11 = _mm_loadu_ps(inptr0 + 44); //A0A1A2A3

    __m128 xmm12 = _mm_unpacklo_ps(xmm0, xmm2); //A0C0A1C1
    __m128 xmm13 = _mm_unpackhi_ps(xmm0, xmm2); //A2C2A3C3
    __m128 xmm14 = _mm_unpacklo_ps(xmm1, xmm3); //B0D0B1D1
    __m128 xmm15 = _mm_unpackhi_ps(xmm1, xmm3); //B2D2B3D3

    xmm0 = _mm_unpacklo_ps(xmm12, xmm14); //A0B0C0D0
    xmm1 = _mm_unpackhi_ps(xmm12, xmm14); //A1B1C1D1
    xmm2 = _mm_unpacklo_ps(xmm13, xmm15); //A2B2C2D2
    xmm3 = _mm_unpackhi_ps(xmm13, xmm15); //A3B3C3D3

    xmm12 = _mm_unpacklo_ps(xmm4, xmm6); 
    xmm13 = _mm_unpackhi_ps(xmm4, xmm6); 
    xmm14 = _mm_unpacklo_ps(xmm5, xmm7); 
    xmm15 = _mm_unpackhi_ps(xmm5, xmm7); 

    xmm4 = _mm_unpacklo_ps(xmm12, xmm14); //E
    xmm5 = _mm_unpackhi_ps(xmm12, xmm14); //F
    xmm6 = _mm_unpacklo_ps(xmm13, xmm15); //G
    xmm7 = _mm_unpackhi_ps(xmm13, xmm15); //H

    xmm12 = _mm_unpacklo_ps(xmm8, xmm10); 
    xmm13 = _mm_unpackhi_ps(xmm8, xmm10); 
    xmm14 = _mm_unpacklo_ps(xmm9, xmm11); 
    xmm15 = _mm_unpackhi_ps(xmm9, xmm11); 

    xmm8 = _mm_unpacklo_ps(xmm12, xmm14); 
    xmm9 = _mm_unpackhi_ps(xmm12, xmm14); 
    xmm10 = _mm_unpacklo_ps(xmm13, xmm15); 
    xmm11 = _mm_unpackhi_ps(xmm13, xmm15); 

    _mm_storeu_ps(outptr, xmm0);
    _mm_storeu_ps(outptr + 4, xmm4);
    _mm_storeu_ps(outptr + 8, xmm8);
    _mm_storeu_ps(outptr + 12, xmm1);
    _mm_storeu_ps(outptr + 16, xmm5);
    _mm_storeu_ps(outptr + 20, xmm9);
    _mm_storeu_ps(outptr + 24, xmm2);
    _mm_storeu_ps(outptr + 28, xmm6);
    _mm_storeu_ps(outptr + 32, xmm10);
    _mm_storeu_ps(outptr + 36, xmm3);
    _mm_storeu_ps(outptr + 40, xmm7);
    _mm_storeu_ps(outptr + 44, xmm11);
#endif
}
    )";
}

std::string transpose_1x4_4_s() {
    return R"(
static inline void transpose_1x4_4_s(const float* inptr0, float* outptr) {
#ifdef __SSE__
    __m128 xmm0 = _mm_loadu_ps(inptr0);     //A0A1A2A3
    __m128 xmm1 = _mm_loadu_ps(inptr0 + 4); //B0B1B2B3
    __m128 xmm2 = _mm_loadu_ps(inptr0 + 8); //C0C1C2C3
    __m128 xmm3 = _mm_loadu_ps(inptr0 + 12); //D0D1D2D3

    __m128 xmm4 = _mm_unpacklo_ps(xmm0, xmm2); //A0C0A1C1
    __m128 xmm5 = _mm_unpackhi_ps(xmm0, xmm2); //A2C2A3C3
    __m128 xmm6 = _mm_unpacklo_ps(xmm1, xmm3); //B0D0B1D1
    __m128 xmm7 = _mm_unpackhi_ps(xmm1, xmm3); //B2D2B3D3

    xmm0 = _mm_unpacklo_ps(xmm4, xmm6); //A0B0C0D0
    xmm1 = _mm_unpackhi_ps(xmm4, xmm6); //A1B1C1D1
    xmm2 = _mm_unpacklo_ps(xmm5, xmm7); //A2B2C2D2
    xmm3 = _mm_unpackhi_ps(xmm5, xmm7); //A3B3C3D3

    _mm_storeu_ps(outptr, xmm0);
    _mm_storeu_ps(outptr + 4, xmm1);
    _mm_storeu_ps(outptr + 8, xmm2);
    _mm_storeu_ps(outptr + 12, xmm3);
#endif
}
    )";
}

static std::string kern_4x12(TContext* ctx) {
    std::stringstream writer;
    // TODO:with bias not implemented
    writer << R"(
static inline void kern_4x12_bias_relu(const float* packA, const float* packB, int K, float* output, int LDC, const float* bias_ptr) {
#ifdef __WASM_SIMD128_H
    v128_t xmm0, xmm1, xmm2, xmm3;
    v128_t xmm4, xmm5, xmm6, xmm7;
    /*Res*/
    v128_t xmm8, xmm9, xmm10, xmm11;
    v128_t xmm12, xmm13, xmm14, xmm15;

    const float* a_ptr = packA;
    const float* b_ptr = packB;
    float* output0 = output;

    xmm0 = wasm_v128_load(a_ptr);
    //i = 0;
    print_data(a_ptr, 4);

    print_data(b_ptr, 12);
    xmm1 = wasm_f32x4_splat(*(b_ptr+0));
    xmm4 = wasm_f32x4_mul(xmm0, xmm1);

    xmm2 = wasm_f32x4_splat(*(b_ptr+1));
    xmm5 = wasm_f32x4_mul(xmm0, xmm2);

    xmm3 = wasm_f32x4_splat(*(b_ptr+2));
    xmm6 = wasm_f32x4_mul(xmm0, xmm3);

    xmm1 = wasm_f32x4_splat(*(b_ptr+3));
    xmm7 = wasm_f32x4_mul(xmm0, xmm1);
    
    xmm2 = wasm_f32x4_splat(*(b_ptr+4));
    xmm8 = wasm_f32x4_mul(xmm0, xmm2);

    xmm3 = wasm_f32x4_splat(*(b_ptr+5));
    xmm9 = wasm_f32x4_mul(xmm0, xmm3);

    xmm1 = wasm_f32x4_splat(*(b_ptr+6));
    xmm10 = wasm_f32x4_mul(xmm0, xmm1);
    
    xmm2 = wasm_f32x4_splat(*(b_ptr+7));
    xmm11 = wasm_f32x4_mul(xmm0, xmm2);

    xmm3 = wasm_f32x4_splat(*(b_ptr+8));
    xmm12 = wasm_f32x4_mul(xmm0, xmm3);

    xmm1 = wasm_f32x4_splat(*(b_ptr+9));
    xmm13 = wasm_f32x4_mul(xmm0, xmm1);
    
    xmm2 = wasm_f32x4_splat(*(b_ptr+10));
    xmm14 = wasm_f32x4_mul(xmm0, xmm2);

    xmm3 = wasm_f32x4_splat(*(b_ptr+11));
    xmm15 = wasm_f32x4_mul(xmm0, xmm3);

    b_ptr += 12;

    for (size_t i = 1; i < 4; ++i) {
        print_data(a_ptr+4*i, 4);

        print_data(b_ptr, 12);
        xmm0 = wasm_v128_load(a_ptr + 4 * i);
        //i = 0;
        xmm1 = wasm_f32x4_splat(*(b_ptr+0));
        xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
        
        xmm2 = wasm_f32x4_splat(*(b_ptr+1));
        xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));

        xmm3 = wasm_f32x4_splat(*(b_ptr+2));
        xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));

        xmm1 = wasm_f32x4_splat(*(b_ptr+3));
        xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
        
        xmm2 = wasm_f32x4_splat(*(b_ptr+4));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));

        xmm3 = wasm_f32x4_splat(*(b_ptr+5));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));

        xmm1 = wasm_f32x4_splat(*(b_ptr+6));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
        
        xmm2 = wasm_f32x4_splat(*(b_ptr+7));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));

        xmm3 = wasm_f32x4_splat(*(b_ptr+8));
        xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));

        xmm1 = wasm_f32x4_splat(*(b_ptr+9));
        xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
        
        xmm2 = wasm_f32x4_splat(*(b_ptr+10));
        xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));

        xmm3 = wasm_f32x4_splat(*(b_ptr+11));
        xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));

        b_ptr += 12;
    }


    for (size_t k = 4; k < K; k += 4) {
        // i = 0
        a_ptr += 16; 
        print_data(a_ptr, 4);

        print_data(b_ptr, 12);
        xmm0 = wasm_v128_load(a_ptr);
        xmm1 = wasm_f32x4_splat(*(b_ptr+0));
        xmm2 = wasm_f32x4_splat(*(b_ptr+1));
        xmm3 = wasm_f32x4_splat(*(b_ptr+2));

        xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+3));
        xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+4));
        xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+5));
        xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+6));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+7));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+8));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+9));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+10));
        xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+11));
        xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
        b_ptr += 12;
        xmm1 = wasm_f32x4_splat(*(b_ptr+0));
        xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+1));
        xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+2));

        // i = 1
        print_data(a_ptr+4, 4);

        print_data(b_ptr, 12);
        xmm0 = wasm_v128_load(a_ptr + 4);
        xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+3));
        xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+4));
        xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+5));
        xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+6));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+7));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+8));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+9));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+10));
        xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+11));
        xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
        b_ptr += 12;
        xmm1 = wasm_f32x4_splat(*(b_ptr+0));
        xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+1));
        xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+2));
        print_data(a_ptr+8, 4);

        print_data(b_ptr, 12);
        // i = 2
        xmm0 = wasm_v128_load(a_ptr + 8);
        xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+3));
        xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+4));
        xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+5));
        xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+6));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+7));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+8));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+9));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+10));
        xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+11));
        xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
        b_ptr += 12;
        xmm1 = wasm_f32x4_splat(*(b_ptr+0));
        xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+1));
        xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+2));
        print_data(a_ptr+12, 4);

        print_data(b_ptr, 12);
        // i = 3
        xmm0 = wasm_v128_load(a_ptr + 12);
        xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+3));
        xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+4));
        xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+5));
        xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+6));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+7));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+8));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
        xmm1 = wasm_f32x4_splat(*(b_ptr+9));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
        xmm2 = wasm_f32x4_splat(*(b_ptr+10));
        xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
        xmm3 = wasm_f32x4_splat(*(b_ptr+11));
        xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
        xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
        xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
        b_ptr += 12;
    }

    wasm_v128_store(output + 0, xmm4);
    wasm_v128_store(output + 4, xmm5);
    wasm_v128_store(output + 8, xmm6);
    wasm_v128_store(output + 12, xmm7);
    wasm_v128_store(output + 16, xmm8);
    wasm_v128_store(output + 20, xmm9);
    wasm_v128_store(output + 24, xmm10);
    wasm_v128_store(output + 28, xmm11);
    wasm_v128_store(output + 32, xmm12);
    wasm_v128_store(output + 36, xmm13);
    wasm_v128_store(output + 40, xmm14);
    wasm_v128_store(output + 44, xmm15);


#endif
}
    )";
    return writer.str();
}

static std::string kern_4x4(TContext* crx) {
    std::stringstream writer;
    writer << R"(static inline void kern_4x4_bias_relu(const float* packA, const float* packB, int K, float* output, int LDC, const float* bias_ptr, int n_remain) {
#ifdef __WASM_SIMD128_H
    v128_t xmm0, xmm1, xmm2, xmm3;
    v128_t xmm4, xmm5, xmm6, xmm7;
    /*Res*/
    v128_t xmm8, xmm9, xmm10, xmm11;

    const float* a_ptr = packA;
    const float* b_ptr = packB;
    float* output0 = output;

    xmm0 = wasm_v128_load(a_ptr);
    xmm1 = wasm_v128_load(a_ptr + 4);
    xmm2 = wasm_v128_load(a_ptr + 8);
    
    // a0
    xmm4 = wasm_f32x4_splat(*(b_ptr+0));
    xmm8 = wasm_f32x4_mul(xmm0, xmm4);
    xmm3 = wasm_v128_load(a_ptr + 12);
    
    xmm5 = wasm_f32x4_splat(*(b_ptr+1));
    xmm9 = wasm_f32x4_mul(xmm0, xmm5);

    xmm6 = wasm_f32x4_splat(*(b_ptr+2));
    xmm10 = wasm_f32x4_mul(xmm0, xmm6);

    xmm7 = wasm_f32x4_splat(*(b_ptr+3));
    xmm11 = wasm_f32x4_mul(xmm0, xmm7);

    // a1
    xmm4 = wasm_f32x4_splat(*(b_ptr+4));
    xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm1, xmm4));
    
    xmm5 = wasm_f32x4_splat(*(b_ptr+5));
    xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm1, xmm5));

    xmm6 = wasm_f32x4_splat(*(b_ptr+6));
    xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm1, xmm6));

    xmm7 = wasm_f32x4_splat(*(b_ptr+7));
    xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm1, xmm7));

    xmm4 = wasm_f32x4_splat(*(b_ptr+8));
    xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm2, xmm4));
    
    xmm5 = wasm_f32x4_splat(*(b_ptr+9));
    xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm2, xmm5));

    xmm6 = wasm_f32x4_splat(*(b_ptr+10));
    xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm2, xmm6));

    xmm7 = wasm_f32x4_splat(*(b_ptr+11));
    xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm2, xmm7));

    xmm4 = wasm_f32x4_splat(*(b_ptr+12));
    xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm3, xmm4));
    
    xmm5 = wasm_f32x4_splat(*(b_ptr+13));
    xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm3, xmm5));

    xmm6 = wasm_f32x4_splat(*(b_ptr+14));
    xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm3, xmm6));

    xmm7 = wasm_f32x4_splat(*(b_ptr+15));
    xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm3, xmm7));

    // k round
    b_ptr += 16;
    for (size_t k = 4; k < K; k+=4) {
        a_ptr += 16;
        xmm0 = wasm_v128_load(a_ptr);
        xmm1 = wasm_v128_load(a_ptr + 4);
        
        xmm4 = wasm_f32x4_splat(*(b_ptr+0));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm4));
        xmm2 = wasm_v128_load(a_ptr + 8);
    
        xmm5 = wasm_f32x4_splat(*(b_ptr+1));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm5));
        xmm3 = wasm_v128_load(a_ptr + 12);

        xmm6 = wasm_f32x4_splat(*(b_ptr+2));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm6));

        xmm7 = wasm_f32x4_splat(*(b_ptr+3));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm7));

        xmm4 = wasm_f32x4_splat(*(b_ptr+4));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm1, xmm4));
        
        xmm5 = wasm_f32x4_splat(*(b_ptr+5));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm1, xmm5));

        xmm6 = wasm_f32x4_splat(*(b_ptr+6));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm1, xmm6));

        xmm7 = wasm_f32x4_splat(*(b_ptr+7));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm1, xmm7));

        xmm4 = wasm_f32x4_splat(*(b_ptr+8));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm2, xmm4));
        
        xmm5 = wasm_f32x4_splat(*(b_ptr+9));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm2, xmm5));

        xmm6 = wasm_f32x4_splat(*(b_ptr+10));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm2, xmm6));

        xmm7 = wasm_f32x4_splat(*(b_ptr+11));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm2, xmm7));

        xmm4 = wasm_f32x4_splat(*(b_ptr+12));
        xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm3, xmm4));
        
        xmm5 = wasm_f32x4_splat(*(b_ptr+13));
        xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm3, xmm5));

        xmm6 = wasm_f32x4_splat(*(b_ptr+14));
        xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm3, xmm6));

        xmm7 = wasm_f32x4_splat(*(b_ptr+15));
        xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm3, xmm7));
        b_ptr += 16;
    }

    if (n_remain == 1){
        wasm_v128_store(output + 0, xmm8);
    }
    else if(n_remain == 2) {
        wasm_v128_store(output + 0, xmm8);
        wasm_v128_store(output + 4, xmm9);
    }
    else if(n_remain == 3){
        wasm_v128_store(output + 0, xmm8);
        wasm_v128_store(output + 4, xmm9);
        wasm_v128_store(output + 8, xmm10);
    }
    else if(n_remain == 4){
        wasm_v128_store(output + 0, xmm8);
        wasm_v128_store(output + 4, xmm9);
        wasm_v128_store(output + 8, xmm10);
        wasm_v128_store(output + 12, xmm11);
    }

    
#endif
})";
    return writer.str();
}

std::string gen_pack_a(const std::string &sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"(
    const int PACK_C_SIZE = 4;
    size_t cp_length = (kmax - k0) * PACK_C_SIZE;
    for (int m = y0; m < ymax; m += 4) {
        const float* src = inptr + (m / PACK_C_SIZE) * ldin + k0 * PACK_C_SIZE;
        memcpy(outptr, src, cp_length * sizeof(float));
        outptr += cp_length;
    }
})";
    return ss.str();
}

std::string gen_pack_b(const std::string &sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"(
    float tmpbuff[16] = {0.0f};
    const int PACK_C_SIZE = 4;
    int ksize = kmax - k0;
    int ksize12 = ksize * 12;
    int ksize4 = (ksize << 2);
    float* outptr_base = outptr;
    float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr0 = inptr + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;

        int x = x0;
        float* access_outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            float*  outptr_interleave = access_outptr;
            transpose_1x12_4_s(inptr0, outptr_interleave);
            inptr0 += 48;
            access_outptr += ksize12;
        }
        access_outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            float*  outptr_interleave = access_outptr;
            transpose_1x4_4_s(inptr0, outptr_interleave);
            inptr0 += 16;
            access_outptr += ksize4;
        }
        if (x < xmax) {
            //last
            memcpy(tmpbuff, inptr0, sizeof(float) * (xmax - x) * PACK_C_SIZE);
            float*  outptr_interleave = access_outptr;
            const float* tmp_ptr = &tmpbuff[0];
            transpose_1x4_4_s(tmp_ptr, outptr_interleave);
            access_outptr += ksize4;
        }
        outptr_base += 12 * PACK_C_SIZE;
        outptr_base4 += 4 * PACK_C_SIZE;
    }
    }
)";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        size_t res = (size_t)(kmax - k0) * (ymax - y0) * sizeof(float);
        return res;
    })";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_n = 12;
        const size_t packed_hw = (xmax - x0 + packed_n - 1) / packed_n * packed_n;
        size_t res = (size_t)(kmax - k0) * packed_hw * sizeof(float);
        return res;
    })";
    return ss.str();
}

std::string gen_kernel(
        const std::string& sig, TContext* ctx, const std::string& postprocess_call,
        const std::string& preset_str = "") {
    std::string keren_body =
            R"(
    ${kernel_sig}{
        ${preset_str}
        const int m_block = 4;
        const int n_block = 12;
        const int pack_mk = 4;
        const int K12 = K * 12;
        const int K4 = K * 4;
        size_t m = 0;        
        for (; m + m_block <= M; m += m_block) {
            float* output = C + (m / pack_mk * LDC);

            size_t n = 0;
            const float* cur_pack_b = pack_b;
            for (; n + n_block <= N; n += n_block) {
                kern_4x12_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                    bias_ptr);
                output += n_block * pack_mk;
                cur_pack_b += K12;
            }

            for (; n < N; n += 4) {                
                kern_4x4_bias_relu(pack_a, cur_pack_b, K, output, LDC,
                                   bias_ptr, N - n > 4 ? 4 : N - n);
                output += 4 * pack_mk;
                cur_pack_b += K4;
            }
            pack_a += K4;
            bias_ptr += m_block;
        }        
        ${postprocess_call}
    }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("postprocess_call", postprocess_call)
            .add("preset_str", preset_str)
            .add("kernel_sig", sig)
            .render(keren_body);
}

}

std::string MatmulM4N12MK4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "WebAssembly_fp32_m4_n12_k4_matmul";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    return ss.str();
}

std::string MatmulM4N12MK4Kernel::GetKernelBody(TContext* ctx) const {
   
    auto postprocess_pair = "";
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <xmmintrin.h>\n";
    writer << "#include <wasm_simd128.h>\n";
    writer << transpose_1x12_4_s();
    writer << transpose_1x4_4_s();
    writer << kern_4x12(ctx);
    writer << kern_4x4(ctx);
    writer << gen_pack_a(GetPackASignature(ctx));
    writer << gen_pack_b(GetPackBSignature(ctx));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));

    std::string preset_temp = R"(
        size_t pack_a_size = ${packa_workspace_sym}(0, M, 0, K);
        float* pack_a = workspace;
        float* pack_b = workspace + pack_a_size;
        ${packa_sym}(pack_a, A, LDA, 0, M, 0, K);
        ${packb_sym}(pack_b, B, LDB, 0, N, 0, K);
    )";
    std::string preset_str =
            StringTemplate::StringTemplateArgs()
                    .add("packa_workspace_sym", GetPackAWorkspaceSymbol(ctx))
                    .add("packa_sym", GetPackASymbol(ctx))
                    .add("packb_sym", GetPackBSymbol(ctx))
                    .render(preset_temp);
    writer << gen_kernel(
            GetKernelSignature(ctx), ctx, postprocess_pair, preset_str);
    return writer.str();
}


std::string MatmulM4N12MK4Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}

std::string MatmulM4N12MK4Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}