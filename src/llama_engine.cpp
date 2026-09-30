#include "llama_engine.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <dispatch/dispatch.h>
#include <arm_neon.h>
#define HAS_ARM_NEON 1
#endif

constexpr size_t MAX_SEQ_LEN = 8192;

inline float bf16_to_fp32(uint16_t val) noexcept {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

inline void enable_fast_math() noexcept {
#if defined(__aarch64__) || defined(__arm64__)
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ULL << 24); // FZ: Flush-to-zero
    fpcr |= (1ULL << 19); // DN: Default NaN
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#endif
}

// ============================================================================
// Platform Memory Mapping
// ============================================================================
struct PlatformMemoryMap {
    int fd{-1};
    size_t size{0};
    const uint8_t* data{nullptr};

    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat sb;
        fstat(fd, &sb);
        size = sb.st_size;
        data = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0));
        return data != MAP_FAILED && data != nullptr;
    }

    void close() {
        if (data && data != MAP_FAILED) { munmap(const_cast<uint8_t*>(data), size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    ~PlatformMemoryMap() { close(); }
};

// ============================================================================
// Parallel Dispatch
// ============================================================================
template <typename F>
inline void parallel_for(size_t count, F&& func) {
#if defined(__APPLE__) && defined(__MACH__)
    dispatch_apply(count, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t i) {
        func(i);
    });
#else
    for (size_t i = 0; i < count; ++i) func(i);
#endif
}

// ============================================================================
// RMS Normalization
// ============================================================================
inline void rms_norm(const float* x, const float* weight, float* out, size_t size) noexcept {
#if HAS_ARM_NEON
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < size; i += 16) {
        float32x4_t v0 = vld1q_f32(&x[i + 0]);
        float32x4_t v1 = vld1q_f32(&x[i + 4]);
        float32x4_t v2 = vld1q_f32(&x[i + 8]);
        float32x4_t v3 = vld1q_f32(&x[i + 12]);
        sum_vec = vmlaq_f32(sum_vec, v0, v0);
        sum_vec = vmlaq_f32(sum_vec, v1, v1);
        sum_vec = vmlaq_f32(sum_vec, v2, v2);
        sum_vec = vmlaq_f32(sum_vec, v3, v3);
    }
    float sum = vaddvq_f32(sum_vec);
    float inv_rms = 1.0f / std::sqrt((sum / static_cast<float>(size)) + 1e-5f);
    float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
    for (size_t i = 0; i < size; i += 16) {
        vst1q_f32(&out[i + 0],  vmulq_f32(vld1q_f32(&x[i + 0]),  vmulq_f32(vld1q_f32(&weight[i + 0]),  inv_rms_vec)));
        vst1q_f32(&out[i + 4],  vmulq_f32(vld1q_f32(&x[i + 4]),  vmulq_f32(vld1q_f32(&weight[i + 4]),  inv_rms_vec)));
        vst1q_f32(&out[i + 8],  vmulq_f32(vld1q_f32(&x[i + 8]),  vmulq_f32(vld1q_f32(&weight[i + 8]),  inv_rms_vec)));
        vst1q_f32(&out[i + 12], vmulq_f32(vld1q_f32(&x[i + 12]), vmulq_f32(vld1q_f32(&weight[i + 12]), inv_rms_vec)));
    }
#else
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) sum += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt((sum / static_cast<float>(size)) + 1e-5f);
    for (size_t i = 0; i < size; ++i) out[i] = x[i] * weight[i] * inv_rms;
#endif
}

// ============================================================================
// RoPE Rotary Positional Embedding
// ============================================================================
inline void apply_rope(float* vec, size_t n_heads, size_t head_dim, size_t pos,
                       const float* cos_tab, const float* sin_tab) noexcept {
    const float* c_ptr = &cos_tab[pos * (head_dim / 2)];
    const float* s_ptr = &sin_tab[pos * (head_dim / 2)];

    for (size_t h = 0; h < n_heads; ++h) {
        float* h_ptr = &vec[h * head_dim];
#if HAS_ARM_NEON
        for (size_t i = 0; i < head_dim / 2; i += 4) {
            float32x4_t r0 = vld1q_f32(&h_ptr[i]);
            float32x4_t r1 = vld1q_f32(&h_ptr[i + head_dim / 2]);
            float32x4_t c = vld1q_f32(&c_ptr[i]);
            float32x4_t s = vld1q_f32(&s_ptr[i]);

            float32x4_t out0 = vsubq_f32(vmulq_f32(r0, c), vmulq_f32(r1, s));
            float32x4_t out1 = vaddq_f32(vmulq_f32(r0, s), vmulq_f32(r1, c));

            vst1q_f32(&h_ptr[i], out0);
            vst1q_f32(&h_ptr[i + head_dim / 2], out1);
        }
#else
        for (size_t i = 0; i < head_dim / 2; ++i) {
            float r0 = h_ptr[i];
            float r1 = h_ptr[i + head_dim / 2];
            float c = c_ptr[i];
            float s = s_ptr[i];
            h_ptr[i] = r0 * c - r1 * s;
            h_ptr[i + head_dim / 2] = r0 * s + r1 * c;
        }
#endif
    }
}

// ============================================================================
// Signed 4-Bit Block-128 Matrix Structure
// ============================================================================
struct SignedBlock128Matrix {
    uint32_t out_dim{0};
    uint32_t in_dim{0};
    uint32_t num_blocks{0};
    const uint8_t* nibbles{nullptr};
    const uint16_t* step_bf16{nullptr};
    const uint16_t* min_bf16{nullptr};

    size_t map_from_buffer(const uint8_t* ptr) {
        size_t off = 0;
        std::memcpy(&out_dim, ptr + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&in_dim, ptr + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&num_blocks, ptr + off, sizeof(uint32_t)); off += sizeof(uint32_t);

        nibbles = ptr + off;
        off += num_blocks * 64;

        step_bf16 = reinterpret_cast<const uint16_t*>(ptr + off);
        off += num_blocks * sizeof(uint16_t);

        min_bf16 = reinterpret_cast<const uint16_t*>(ptr + off);
        off += num_blocks * sizeof(uint16_t);

        return off;
    }

    void dequantize_embedding_row(size_t token_id, float* out_x) const noexcept {
        size_t blocks_per_row = in_dim / 128;
        size_t b_start = token_id * blocks_per_row;

        for (size_t b = 0; b < blocks_per_row; ++b) {
            size_t blk = b_start + b;
            float step = bf16_to_fp32(step_bf16[blk]);
            float min_val = bf16_to_fp32(min_bf16[blk]);
#if HAS_ARM_NEON
            float32x4_t s_vec = vdupq_n_f32(step);
            float32x4_t m_vec = vdupq_n_f32(min_val);
            const uint8_t* np = &nibbles[blk * 64];

            for (size_t chunk = 0; chunk < 8; ++chunk) {
                uint8x8_t raw = vld1_u8(&np[chunk * 8]);
                uint8x8x2_t zipped = vzip_u8(vand_u8(raw, vdup_n_u8(0x0F)), vshr_n_u8(raw, 4));
                uint16x8_t q0 = vmovl_u8(zipped.val[0]);
                uint16x8_t q1 = vmovl_u8(zipped.val[1]);

                float32x4_t q_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q0)));
                float32x4_t q_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q0)));
                float32x4_t q_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q1)));
                float32x4_t q_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q1)));

                vst1q_f32(&out_x[b * 128 + chunk * 16 + 0],  vmlaq_f32(m_vec, q_f0, s_vec));
                vst1q_f32(&out_x[b * 128 + chunk * 16 + 4],  vmlaq_f32(m_vec, q_f1, s_vec));
                vst1q_f32(&out_x[b * 128 + chunk * 16 + 8],  vmlaq_f32(m_vec, q_f2, s_vec));
                vst1q_f32(&out_x[b * 128 + chunk * 16 + 12], vmlaq_f32(m_vec, q_f3, s_vec));
            }
#else
            const uint8_t* np = &nibbles[blk * 64];
            for (int j = 0; j < 64; ++j) {
                uint8_t byte = np[j];
                out_x[b * 128 + j * 2 + 0] = min_val + static_cast<float>(byte & 0x0F) * step;
                out_x[b * 128 + j * 2 + 1] = min_val + static_cast<float>(byte >> 4) * step;
            }
#endif
        }
    }
};

struct LayerBlock128 {
    const float* in_norm{nullptr};
    const float* post_norm{nullptr};
    SignedBlock128Matrix W_q, W_k, W_v, W_o, W_gate, W_up, W_down;
};

// ============================================================================
// High-Performance Vectorized Factored GEMV for Signed 4-Bit Block-128
// w_i = min + q_i * step  ==>  w dot x = min * sum(x) + step * (q dot x)
// ============================================================================
inline void signed_4bit_b128_gemv(const SignedBlock128Matrix& W, const float* x, const float* x_block_sums, float* y) noexcept {
    size_t M = W.out_dim;
    size_t K = W.in_dim;
    size_t blocks_per_row = K / 128;

    parallel_for(M / 4, [&](size_t r4) {
        size_t row0 = r4 * 4 + 0, row1 = r4 * 4 + 1, row2 = r4 * 4 + 2, row3 = r4 * 4 + 3;
        size_t b_start0 = row0 * blocks_per_row, b_start1 = row1 * blocks_per_row;
        size_t b_start2 = row2 * blocks_per_row, b_start3 = row3 * blocks_per_row;

        float r0_total = 0.0f, r1_total = 0.0f, r2_total = 0.0f, r3_total = 0.0f;

#if HAS_ARM_NEON
        uint8x8_t mask_lo = vdup_n_u8(0x0F);

        for (size_t b = 0; b < blocks_per_row; ++b) {
            size_t blk0 = b_start0 + b, blk1 = b_start1 + b;
            size_t blk2 = b_start2 + b, blk3 = b_start3 + b;

            float step0 = bf16_to_fp32(W.step_bf16[blk0]), min0 = bf16_to_fp32(W.min_bf16[blk0]);
            float step1 = bf16_to_fp32(W.step_bf16[blk1]), min1 = bf16_to_fp32(W.min_bf16[blk1]);
            float step2 = bf16_to_fp32(W.step_bf16[blk2]), min2 = bf16_to_fp32(W.min_bf16[blk2]);
            float step3 = bf16_to_fp32(W.step_bf16[blk3]), min3 = bf16_to_fp32(W.min_bf16[blk3]);

            const uint8_t* np0 = &W.nibbles[blk0 * 64];
            const uint8_t* np1 = &W.nibbles[blk1 * 64];
            const uint8_t* np2 = &W.nibbles[blk2 * 64];
            const uint8_t* np3 = &W.nibbles[blk3 * 64];
            const float* px = &x[b * 128];

            float32x4_t r0_acc0 = vdupq_n_f32(0.0f), r0_acc1 = vdupq_n_f32(0.0f);
            float32x4_t r1_acc0 = vdupq_n_f32(0.0f), r1_acc1 = vdupq_n_f32(0.0f);
            float32x4_t r2_acc0 = vdupq_n_f32(0.0f), r2_acc1 = vdupq_n_f32(0.0f);
            float32x4_t r3_acc0 = vdupq_n_f32(0.0f), r3_acc1 = vdupq_n_f32(0.0f);

            for (size_t chunk = 0; chunk < 8; ++chunk) {
                float32x4_t x0 = vld1q_f32(&px[chunk * 16 + 0]);
                float32x4_t x1 = vld1q_f32(&px[chunk * 16 + 4]);
                float32x4_t x2 = vld1q_f32(&px[chunk * 16 + 8]);
                float32x4_t x3 = vld1q_f32(&px[chunk * 16 + 12]);

                // Row 0
                uint8x8_t raw0 = vld1_u8(&np0[chunk * 8]);
                uint8x8x2_t zip0 = vzip_u8(vand_u8(raw0, mask_lo), vshr_n_u8(raw0, 4));
                uint16x8_t q0_0 = vmovl_u8(zip0.val[0]);
                uint16x8_t q0_1 = vmovl_u8(zip0.val[1]);
                r0_acc0 = vmlaq_f32(r0_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q0_0))), x0);
                r0_acc1 = vmlaq_f32(r0_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q0_0))), x1);
                r0_acc0 = vmlaq_f32(r0_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q0_1))), x2);
                r0_acc1 = vmlaq_f32(r0_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q0_1))), x3);

                // Row 1
                uint8x8_t raw1 = vld1_u8(&np1[chunk * 8]);
                uint8x8x2_t zip1 = vzip_u8(vand_u8(raw1, mask_lo), vshr_n_u8(raw1, 4));
                uint16x8_t q1_0 = vmovl_u8(zip1.val[0]);
                uint16x8_t q1_1 = vmovl_u8(zip1.val[1]);
                r1_acc0 = vmlaq_f32(r1_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q1_0))), x0);
                r1_acc1 = vmlaq_f32(r1_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q1_0))), x1);
                r1_acc0 = vmlaq_f32(r1_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q1_1))), x2);
                r1_acc1 = vmlaq_f32(r1_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q1_1))), x3);

                // Row 2
                uint8x8_t raw2 = vld1_u8(&np2[chunk * 8]);
                uint8x8x2_t zip2 = vzip_u8(vand_u8(raw2, mask_lo), vshr_n_u8(raw2, 4));
                uint16x8_t q2_0 = vmovl_u8(zip2.val[0]);
                uint16x8_t q2_1 = vmovl_u8(zip2.val[1]);
                r2_acc0 = vmlaq_f32(r2_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q2_0))), x0);
                r2_acc1 = vmlaq_f32(r2_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q2_0))), x1);
                r2_acc0 = vmlaq_f32(r2_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q2_1))), x2);
                r2_acc1 = vmlaq_f32(r2_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q2_1))), x3);

                // Row 3
                uint8x8_t raw3 = vld1_u8(&np3[chunk * 8]);
                uint8x8x2_t zip3 = vzip_u8(vand_u8(raw3, mask_lo), vshr_n_u8(raw3, 4));
                uint16x8_t q3_0 = vmovl_u8(zip3.val[0]);
                uint16x8_t q3_1 = vmovl_u8(zip3.val[1]);
                r3_acc0 = vmlaq_f32(r3_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q3_0))), x0);
                r3_acc1 = vmlaq_f32(r3_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q3_0))), x1);
                r3_acc0 = vmlaq_f32(r3_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(q3_1))), x2);
                r3_acc1 = vmlaq_f32(r3_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(q3_1))), x3);
            }

            float xs = x_block_sums[b];
            r0_total += min0 * xs + step0 * vaddvq_f32(vaddq_f32(r0_acc0, r0_acc1));
            r1_total += min1 * xs + step1 * vaddvq_f32(vaddq_f32(r1_acc0, r1_acc1));
            r2_total += min2 * xs + step2 * vaddvq_f32(vaddq_f32(r2_acc0, r2_acc1));
            r3_total += min3 * xs + step3 * vaddvq_f32(vaddq_f32(r3_acc0, r3_acc1));
        }
#else
        for (size_t b = 0; b < blocks_per_row; ++b) {
            size_t blk0 = b_start0 + b, blk1 = b_start1 + b;
            size_t blk2 = b_start2 + b, blk3 = b_start3 + b;

            float step0 = bf16_to_fp32(W.step_bf16[blk0]), min0 = bf16_to_fp32(W.min_bf16[blk0]);
            float step1 = bf16_to_fp32(W.step_bf16[blk1]), min1 = bf16_to_fp32(W.min_bf16[blk1]);
            float step2 = bf16_to_fp32(W.step_bf16[blk2]), min2 = bf16_to_fp32(W.min_bf16[blk2]);
            float step3 = bf16_to_fp32(W.step_bf16[blk3]), min3 = bf16_to_fp32(W.min_bf16[blk3]);

            const uint8_t* np0 = &W.nibbles[blk0 * 64];
            const uint8_t* np1 = &W.nibbles[blk1 * 64];
            const uint8_t* np2 = &W.nibbles[blk2 * 64];
            const uint8_t* np3 = &W.nibbles[blk3 * 64];
            const float* px = &x[b * 128];

            float q_dot0 = 0.0f, q_dot1 = 0.0f, q_dot2 = 0.0f, q_dot3 = 0.0f;
            for (int j = 0; j < 64; ++j) {
                float x0 = px[j * 2 + 0], x1 = px[j * 2 + 1];
                q_dot0 += static_cast<float>(np0[j] & 0xF) * x0 + static_cast<float>(np0[j] >> 4) * x1;
                q_dot1 += static_cast<float>(np1[j] & 0xF) * x0 + static_cast<float>(np1[j] >> 4) * x1;
                q_dot2 += static_cast<float>(np2[j] & 0xF) * x0 + static_cast<float>(np2[j] >> 4) * x1;
                q_dot3 += static_cast<float>(np3[j] & 0xF) * x0 + static_cast<float>(np3[j] >> 4) * x1;
            }
            float xs = x_block_sums[b];
            r0_total += min0 * xs + step0 * q_dot0;
            r1_total += min1 * xs + step1 * q_dot1;
            r2_total += min2 * xs + step2 * q_dot2;
            r3_total += min3 * xs + step3 * q_dot3;
        }
#endif
        y[row0] = r0_total;
        y[row1] = r1_total;
        y[row2] = r2_total;
        y[row3] = r3_total;
    });
}

// ============================================================================
// Fused SwiGLU: out = silu(W_gate * x) * (W_up * x)
// ============================================================================
inline void signed_4bit_b128_fused_swiglu(
    const SignedBlock128Matrix& W_gate,
    const SignedBlock128Matrix& W_up,
    const float* x,
    const float* x_block_sums,
    float* out_act) noexcept
{
    size_t M = W_gate.out_dim;
    size_t K = W_gate.in_dim;
    size_t blocks_per_row = K / 128;

    auto silu = [](float val) noexcept { return val / (1.0f + std::exp(-val)); };

    parallel_for(M / 2, [&](size_t r2) {
        size_t row0 = r2 * 2 + 0, row1 = r2 * 2 + 1;
        size_t b_start0 = row0 * blocks_per_row, b_start1 = row1 * blocks_per_row;

        float g0_total = 0.0f, u0_total = 0.0f;
        float g1_total = 0.0f, u1_total = 0.0f;

#if HAS_ARM_NEON
        uint8x8_t mask_lo = vdup_n_u8(0x0F);

        for (size_t b = 0; b < blocks_per_row; ++b) {
            size_t blk0 = b_start0 + b, blk1 = b_start1 + b;

            float g_step0 = bf16_to_fp32(W_gate.step_bf16[blk0]), g_min0 = bf16_to_fp32(W_gate.min_bf16[blk0]);
            float u_step0 = bf16_to_fp32(W_up.step_bf16[blk0]),   u_min0 = bf16_to_fp32(W_up.min_bf16[blk0]);

            float g_step1 = bf16_to_fp32(W_gate.step_bf16[blk1]), g_min1 = bf16_to_fp32(W_gate.min_bf16[blk1]);
            float u_step1 = bf16_to_fp32(W_up.step_bf16[blk1]),   u_min1 = bf16_to_fp32(W_up.min_bf16[blk1]);

            const uint8_t* g_np0 = &W_gate.nibbles[blk0 * 64];
            const uint8_t* u_np0 = &W_up.nibbles[blk0 * 64];
            const uint8_t* g_np1 = &W_gate.nibbles[blk1 * 64];
            const uint8_t* u_np1 = &W_up.nibbles[blk1 * 64];
            const float* px = &x[b * 128];

            float32x4_t g0_acc0 = vdupq_n_f32(0.0f), g0_acc1 = vdupq_n_f32(0.0f);
            float32x4_t u0_acc0 = vdupq_n_f32(0.0f), u0_acc1 = vdupq_n_f32(0.0f);
            float32x4_t g1_acc0 = vdupq_n_f32(0.0f), g1_acc1 = vdupq_n_f32(0.0f);
            float32x4_t u1_acc0 = vdupq_n_f32(0.0f), u1_acc1 = vdupq_n_f32(0.0f);

            for (size_t chunk = 0; chunk < 8; ++chunk) {
                float32x4_t x0 = vld1q_f32(&px[chunk * 16 + 0]);
                float32x4_t x1 = vld1q_f32(&px[chunk * 16 + 4]);
                float32x4_t x2 = vld1q_f32(&px[chunk * 16 + 8]);
                float32x4_t x3 = vld1q_f32(&px[chunk * 16 + 12]);

                // Row 0 Gate
                uint8x8_t g_raw0 = vld1_u8(&g_np0[chunk * 8]);
                uint8x8x2_t g_zip0 = vzip_u8(vand_u8(g_raw0, mask_lo), vshr_n_u8(g_raw0, 4));
                uint16x8_t g0_0 = vmovl_u8(g_zip0.val[0]);
                uint16x8_t g0_1 = vmovl_u8(g_zip0.val[1]);
                g0_acc0 = vmlaq_f32(g0_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(g0_0))), x0);
                g0_acc1 = vmlaq_f32(g0_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(g0_0))), x1);
                g0_acc0 = vmlaq_f32(g0_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(g0_1))), x2);
                g0_acc1 = vmlaq_f32(g0_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(g0_1))), x3);

                // Row 0 Up
                uint8x8_t u_raw0 = vld1_u8(&u_np0[chunk * 8]);
                uint8x8x2_t u_zip0 = vzip_u8(vand_u8(u_raw0, mask_lo), vshr_n_u8(u_raw0, 4));
                uint16x8_t u0_0 = vmovl_u8(u_zip0.val[0]);
                uint16x8_t u0_1 = vmovl_u8(u_zip0.val[1]);
                u0_acc0 = vmlaq_f32(u0_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0_0))), x0);
                u0_acc1 = vmlaq_f32(u0_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u0_0))), x1);
                u0_acc0 = vmlaq_f32(u0_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u0_1))), x2);
                u0_acc1 = vmlaq_f32(u0_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u0_1))), x3);

                // Row 1 Gate
                uint8x8_t g_raw1 = vld1_u8(&g_np1[chunk * 8]);
                uint8x8x2_t g_zip1 = vzip_u8(vand_u8(g_raw1, mask_lo), vshr_n_u8(g_raw1, 4));
                uint16x8_t g1_0 = vmovl_u8(g_zip1.val[0]);
                uint16x8_t g1_1 = vmovl_u8(g_zip1.val[1]);
                g1_acc0 = vmlaq_f32(g1_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(g1_0))), x0);
                g1_acc1 = vmlaq_f32(g1_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(g1_0))), x1);
                g1_acc0 = vmlaq_f32(g1_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(g1_1))), x2);
                g1_acc1 = vmlaq_f32(g1_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(g1_1))), x3);

                // Row 1 Up
                uint8x8_t u_raw1 = vld1_u8(&u_np1[chunk * 8]);
                uint8x8x2_t u_zip1 = vzip_u8(vand_u8(u_raw1, mask_lo), vshr_n_u8(u_raw1, 4));
                uint16x8_t u1_0 = vmovl_u8(u_zip1.val[0]);
                uint16x8_t u1_1 = vmovl_u8(u_zip1.val[1]);
                u1_acc0 = vmlaq_f32(u1_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1_0))), x0);
                u1_acc1 = vmlaq_f32(u1_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u1_0))), x1);
                u1_acc0 = vmlaq_f32(u1_acc0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(u1_1))), x2);
                u1_acc1 = vmlaq_f32(u1_acc1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(u1_1))), x3);
            }

            float xs = x_block_sums[b];
            g0_total += g_min0 * xs + g_step0 * vaddvq_f32(vaddq_f32(g0_acc0, g0_acc1));
            u0_total += u_min0 * xs + u_step0 * vaddvq_f32(vaddq_f32(u0_acc0, u0_acc1));

            g1_total += g_min1 * xs + g_step1 * vaddvq_f32(vaddq_f32(g1_acc0, g1_acc1));
            u1_total += u_min1 * xs + u_step1 * vaddvq_f32(vaddq_f32(u1_acc0, u1_acc1));
        }
#else
        for (size_t b = 0; b < blocks_per_row; ++b) {
            size_t blk0 = b_start0 + b, blk1 = b_start1 + b;
            float g_step0 = bf16_to_fp32(W_gate.step_bf16[blk0]), g_min0 = bf16_to_fp32(W_gate.min_bf16[blk0]);
            float u_step0 = bf16_to_fp32(W_up.step_bf16[blk0]),   u_min0 = bf16_to_fp32(W_up.min_bf16[blk0]);
            float g_step1 = bf16_to_fp32(W_gate.step_bf16[blk1]), g_min1 = bf16_to_fp32(W_gate.min_bf16[blk1]);
            float u_step1 = bf16_to_fp32(W_up.step_bf16[blk1]),   u_min1 = bf16_to_fp32(W_up.min_bf16[blk1]);

            const uint8_t* g_np0 = &W_gate.nibbles[blk0 * 64];
            const uint8_t* u_np0 = &W_up.nibbles[blk0 * 64];
            const uint8_t* g_np1 = &W_gate.nibbles[blk1 * 64];
            const uint8_t* u_np1 = &W_up.nibbles[blk1 * 64];
            const float* px = &x[b * 128];

            float g0_q = 0.0f, u0_q = 0.0f, g1_q = 0.0f, u1_q = 0.0f;
            for (int j = 0; j < 64; ++j) {
                float x0 = px[j * 2 + 0], x1 = px[j * 2 + 1];
                g0_q += static_cast<float>(g_np0[j] & 0xF) * x0 + static_cast<float>(g_np0[j] >> 4) * x1;
                u0_q += static_cast<float>(u_np0[j] & 0xF) * x0 + static_cast<float>(u_np0[j] >> 4) * x1;
                g1_q += static_cast<float>(g_np1[j] & 0xF) * x0 + static_cast<float>(g_np1[j] >> 4) * x1;
                u1_q += static_cast<float>(u_np1[j] & 0xF) * x0 + static_cast<float>(u_np1[j] >> 4) * x1;
            }
            float xs = x_block_sums[b];
            g0_total += g_min0 * xs + g_step0 * g0_q;
            u0_total += u_min0 * xs + u_step0 * u0_q;
            g1_total += g_min1 * xs + g_step1 * g1_q;
            u1_total += u_min1 * xs + u_step1 * u1_q;
        }
#endif
        out_act[row0] = silu(g0_total) * u0_total;
        out_act[row1] = silu(g1_total) * u1_total;
    });
}

// ============================================================================
// Precompute Block Sums
// ============================================================================
inline void compute_x_block_sums(const float* x, size_t K, float* block_sums) noexcept {
    size_t num_blocks = K / 128;
#if HAS_ARM_NEON
    for (size_t b = 0; b < num_blocks; ++b) {
        const float* px = &x[b * 128];
        float32x4_t s0 = vdupq_n_f32(0.0f);
        float32x4_t s1 = vdupq_n_f32(0.0f);
        float32x4_t s2 = vdupq_n_f32(0.0f);
        float32x4_t s3 = vdupq_n_f32(0.0f);

        for (size_t i = 0; i < 128; i += 16) {
            s0 = vaddq_f32(s0, vld1q_f32(&px[i + 0]));
            s1 = vaddq_f32(s1, vld1q_f32(&px[i + 4]));
            s2 = vaddq_f32(s2, vld1q_f32(&px[i + 8]));
            s3 = vaddq_f32(s3, vld1q_f32(&px[i + 12]));
        }
        block_sums[b] = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    }
#else
    for (size_t b = 0; b < num_blocks; ++b) {
        float sum = 0.0f;
        for (size_t i = 0; i < 128; ++i) sum += x[b * 128 + i];
        block_sums[b] = sum;
    }
#endif
}

// ============================================================================
// Tokenizer Implementation
// ============================================================================
class LLaMATokenizer {
public:
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> id_to_tok;
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks;
    std::unordered_map<uint8_t, std::string> byte_to_unicode;
    std::unordered_map<std::string, uint8_t> unicode_to_byte;
    mutable std::string last_decoded_str;

    void init_byte_mappings() {
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            uint8_t b = static_cast<uint8_t>(bs[i]);
            int code = cs[i];
            std::string u_str;
            if (code < 128) {
                u_str += static_cast<char>(code);
            } else if (code < 2048) {
                u_str += static_cast<char>(0xC0 | (code >> 6));
                u_str += static_cast<char>(0x80 | (code & 0x3F));
            } else {
                u_str += static_cast<char>(0xE0 | (code >> 12));
                u_str += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                u_str += static_cast<char>(0x80 | (code & 0x3F));
            }
            byte_to_unicode[b] = u_str;
            unicode_to_byte[u_str] = b;
        }
    }

    bool load(const std::string& path) {
        init_byte_mappings();
        std::ifstream fin(path, std::ios::binary);
        if (!fin.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());

        size_t p_vocab = content.find("\"vocab\": {");
        size_t p_merges = content.find("\"merges\": [", p_vocab);

        if (p_vocab != std::string::npos && p_merges != std::string::npos) {
            size_t cur = p_vocab + 10;
            while (cur < p_merges) {
                size_t q1 = content.find('\"', cur);
                if (q1 >= p_merges || q1 == std::string::npos) break;
                size_t q2 = q1 + 1;
                while (q2 < p_merges) {
                    if (content[q2] == '\"') {
                        size_t backslashes = 0;
                        size_t b = q2;
                        while (b > q1 && content[b - 1] == '\\') { backslashes++; b--; }
                        if (backslashes % 2 == 0) break;
                    }
                    q2++;
                }
                if (q2 >= p_merges) break;
                std::string tok = content.substr(q1 + 1, q2 - q1 - 1);
                std::string clean_tok;
                for (size_t i = 0; i < tok.size(); ++i) {
                    if (tok[i] == '\\' && i + 1 < tok.size()) {
                        if (tok[i+1] == '\"' || tok[i+1] == '\\') {
                            clean_tok += tok[i+1];
                            i++;
                            continue;
                        }
                    }
                    clean_tok += tok[i];
                }
                size_t col = content.find(':', q2);
                if (col >= p_merges || col == std::string::npos) break;
                size_t num_start = col + 1;
                while (num_start < p_merges && !std::isdigit(static_cast<unsigned char>(content[num_start]))) num_start++;
                size_t num_end = num_start;
                int32_t id = 0;
                while (num_end < p_merges && std::isdigit(static_cast<unsigned char>(content[num_end]))) {
                    id = id * 10 + (content[num_end] - '0');
                    num_end++;
                }
                if (num_end > num_start) {
                    vocab[clean_tok] = id;
                    id_to_tok[id] = clean_tok;
                }
                cur = num_end;
            }
        }

        if (p_merges != std::string::npos) {
            size_t p_end = content.rfind("]");
            size_t cur = p_merges + 11;
            int32_t rank = 0;
            while (cur < p_end) {
                size_t q1 = content.find('\"', cur);
                if (q1 >= p_end || q1 == std::string::npos) break;
                size_t q2 = q1 + 1;
                while (q2 < p_end) {
                    if (content[q2] == '\"') {
                        size_t backslashes = 0;
                        size_t b = q2;
                        while (b > q1 && content[b - 1] == '\\') { backslashes++; b--; }
                        if (backslashes % 2 == 0) break;
                    }
                    q2++;
                }
                if (q2 >= p_end) break;
                std::string m = content.substr(q1 + 1, q2 - q1 - 1);
                size_t sp = m.find(' ');
                if (sp != std::string::npos) {
                    std::string p0 = m.substr(0, sp);
                    std::string p1 = m.substr(sp + 1);
                    bpe_ranks[{p0, p1}] = rank++;
                }
                cur = q2 + 1;
            }
        }
        return true;
    }

    std::vector<std::string> bpe(const std::string& token_str) const {
        if (token_str.empty()) return {};
        std::vector<std::string> word;
        for (size_t i = 0; i < token_str.size();) {
            unsigned char c = token_str[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            word.push_back(token_str.substr(i, len));
            i += len;
        }
        if (word.size() <= 1) return word;
        while (true) {
            int32_t min_rank = 1e9;
            int min_idx = -1;
            for (size_t i = 0; i < word.size() - 1; ++i) {
                auto it = bpe_ranks.find({word[i], word[i+1]});
                if (it != bpe_ranks.end() && it->second < min_rank) {
                    min_rank = it->second;
                    min_idx = static_cast<int>(i);
                }
            }
            if (min_idx == -1) break;
            std::vector<std::string> new_word;
            for (size_t i = 0; i < word.size();) {
                if (static_cast<int>(i) == min_idx) {
                    new_word.push_back(word[i] + word[i+1]);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i++;
                }
            }
            word = new_word;
            if (word.size() <= 1) break;
        }
        return word;
    }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> tokens;
        size_t i = 0;
        while (i < text.size()) {
            if (text[i] == ' ') {
                std::string s_tok = byte_to_unicode.at(static_cast<uint8_t>(' '));
                auto it = vocab.find(s_tok);
                if (it != vocab.end()) tokens.push_back(it->second);
                i++;
                continue;
            }
            size_t j = i;
            while (j < text.size() && text[j] != ' ') j++;
            std::string sub = text.substr(i, j - i);
            std::string utf8_mapped;
            for (unsigned char c : sub) utf8_mapped += byte_to_unicode.at(c);
            auto bpe_tokens = bpe(utf8_mapped);
            for (const auto& t : bpe_tokens) {
                auto it = vocab.find(t);
                if (it != vocab.end()) tokens.push_back(it->second);
            }
            i = j;
        }
        return tokens;
    }

    std::string decode(int32_t token_id) const {
        auto it = id_to_tok.find(token_id);
        if (it == id_to_tok.end()) return "";
        std::string s = it->second;
        std::string raw;
        for (size_t i = 0; i < s.size();) {
            unsigned char c = s[i];
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            std::string u_char = s.substr(i, len);
            auto uit = unicode_to_byte.find(u_char);
            if (uit != unicode_to_byte.end()) raw += static_cast<char>(uit->second);
            else raw += u_char;
            i += len;
        }
        return raw;
    }
};

// ============================================================================
// Model State & Memory
// ============================================================================
struct KVCache {
    std::vector<float> k;
    std::vector<float> v;
};

struct ModelScratch {
    std::vector<float> x;
    std::vector<float> norm_buf;
    std::vector<float> block_sums_dim;
    std::vector<float> block_sums_ffn;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_out;
    std::vector<float> proj_o;
    std::vector<float> mlp_act;
    std::vector<float> mlp_out;
    std::vector<float> logits;

    void init(size_t dim, size_t ffn_dim, size_t kv_dim, size_t vocab_size) {
        x.assign(dim, 0.0f);
        norm_buf.assign(std::max(dim, ffn_dim), 0.0f);
        block_sums_dim.assign(dim / 128, 0.0f);
        block_sums_ffn.assign(ffn_dim / 128, 0.0f);
        q.assign(dim, 0.0f);
        k.assign(kv_dim, 0.0f);
        v.assign(kv_dim, 0.0f);
        attn_out.assign(dim, 0.0f);
        proj_o.assign(dim, 0.0f);
        mlp_act.assign(ffn_dim, 0.0f);
        mlp_out.assign(dim, 0.0f);
        logits.assign(vocab_size, 0.0f);
    }
};

struct Candidate {
    int32_t id;
    float val;
};

inline int32_t sample_top_p(float* logits, size_t vocab_size, float temperature, float top_p) noexcept {
    if (temperature <= 0.01f) {
        int32_t best_id = 0;
        float best_val = -1e30f;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] > best_val) { best_val = logits[i]; best_id = static_cast<int32_t>(i); }
        }
        return best_id;
    }

    float inv_t = 1.0f / temperature;
    for (size_t i = 0; i < vocab_size; ++i) logits[i] *= inv_t;

    float max_l = -1e30f;
    for (size_t i = 0; i < vocab_size; ++i) if (logits[i] > max_l) max_l = logits[i];

    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        logits[i] = std::exp(logits[i] - max_l);
        sum_exp += logits[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < vocab_size; ++i) logits[i] *= inv_sum;

    size_t k = std::min<size_t>(64, vocab_size);
    std::vector<Candidate> top_candidates(k);
    for (size_t i = 0; i < k; ++i) top_candidates[i] = {static_cast<int32_t>(i), logits[i]};

    for (size_t i = k; i < vocab_size; ++i) {
        float val = logits[i];
        if (val > top_candidates[k - 1].val) {
            size_t pos = k - 1;
            while (pos > 0 && val > top_candidates[pos - 1].val) {
                top_candidates[pos] = top_candidates[pos - 1];
                pos--;
            }
            top_candidates[pos] = {static_cast<int32_t>(i), val};
        }
    }

    float cum = 0.0f;
    size_t last_idx = 0;
    for (size_t i = 0; i < k; ++i) {
        cum += top_candidates[i].val;
        last_idx = i;
        if (cum >= top_p) break;
    }

    float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * cum;
    float running = 0.0f;
    for (size_t i = 0; i <= last_idx; ++i) {
        running += top_candidates[i].val;
        if (running >= r) return top_candidates[i].id;
    }
    return top_candidates[0].id;
}

// ============================================================================
// Internal Engine State Class
// ============================================================================
struct LlamaEngineState {
    PlatformMemoryMap mmap_model;
    uint32_t num_layers{0};
    uint32_t dim{0};
    uint32_t ffn_dim{0};
    uint32_t vocab_size{0};
    uint32_t num_heads{0};
    uint32_t num_kv_heads{0};
    uint32_t block_size{0};
    size_t head_dim{0};
    size_t kv_dim{0};

    const float* final_norm{nullptr};
    std::vector<LayerBlock128> layers;
    SignedBlock128Matrix W_shared_lm_head;

    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    std::vector<KVCache> kv_caches;
    ModelScratch sc;

    bool load(const std::string& model_path) {
        enable_fast_math();
        if (!mmap_model.open(model_path)) {
            return false;
        }
        const uint8_t* base = mmap_model.data;

        // Check Magic Header
        char magic[8];
        std::memcpy(magic, base, 8);
        if (std::memcmp(magic, "LLM4S128", 8) != 0) {
            return false;
        }

        size_t off = 8;
        std::memcpy(&num_layers, base + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&dim, base + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&ffn_dim, base + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&vocab_size, base + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&num_heads, base + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&num_kv_heads, base + off, sizeof(uint32_t)); off += sizeof(uint32_t);
        std::memcpy(&block_size, base + off, sizeof(uint32_t)); off += sizeof(uint32_t);

        head_dim = dim / num_heads;
        kv_dim = num_kv_heads * head_dim;

        final_norm = reinterpret_cast<const float*>(base + off); off += dim * sizeof(float);

        layers.resize(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            layers[l].in_norm = reinterpret_cast<const float*>(base + off); off += dim * sizeof(float);
            layers[l].post_norm = reinterpret_cast<const float*>(base + off); off += dim * sizeof(float);
            off += layers[l].W_q.map_from_buffer(base + off);
            off += layers[l].W_k.map_from_buffer(base + off);
            off += layers[l].W_v.map_from_buffer(base + off);
            off += layers[l].W_o.map_from_buffer(base + off);
            off += layers[l].W_gate.map_from_buffer(base + off);
            off += layers[l].W_up.map_from_buffer(base + off);
            off += layers[l].W_down.map_from_buffer(base + off);
        }

        off += W_shared_lm_head.map_from_buffer(base + off);

        // Precompute RoPE
        rope_cos.assign(MAX_SEQ_LEN * (head_dim / 2), 0.0f);
        rope_sin.assign(MAX_SEQ_LEN * (head_dim / 2), 0.0f);
        for (size_t pos = 0; pos < MAX_SEQ_LEN; ++pos) {
            for (size_t i = 0; i < head_dim / 2; ++i) {
                float theta = std::pow(500000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                rope_cos[pos * (head_dim / 2) + i] = std::cos(static_cast<float>(pos) * theta);
                rope_sin[pos * (head_dim / 2) + i] = std::sin(static_cast<float>(pos) * theta);
            }
        }

        // Allocate KV caches
        kv_caches.resize(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            kv_caches[l].k.assign(MAX_SEQ_LEN * kv_dim, 0.0f);
            kv_caches[l].v.assign(MAX_SEQ_LEN * kv_dim, 0.0f);
        }

        sc.init(dim, ffn_dim, kv_dim, vocab_size);
        return true;
    }

    void reset() {
        // KV cache positions are overwritten sequentially starting from pos = 0.
        // No memory-zeroing needed, avoiding 128 MB cache thrashing.
    }

    void forward_token(int32_t token_id, size_t pos, bool compute_logits = true) {
        if (pos >= MAX_SEQ_LEN) return;
        if (token_id < 0 || token_id >= static_cast<int32_t>(vocab_size)) token_id = 0;

        // Dequantize token embedding row
        W_shared_lm_head.dequantize_embedding_row(static_cast<size_t>(token_id), sc.x.data());

        for (size_t l = 0; l < num_layers; ++l) {
            const auto& lw = layers[l];
            auto& kv = kv_caches[l];

            rms_norm(sc.x.data(), lw.in_norm, sc.norm_buf.data(), dim);
            compute_x_block_sums(sc.norm_buf.data(), dim, sc.block_sums_dim.data());

            signed_4bit_b128_gemv(lw.W_q, sc.norm_buf.data(), sc.block_sums_dim.data(), sc.q.data());
            signed_4bit_b128_gemv(lw.W_k, sc.norm_buf.data(), sc.block_sums_dim.data(), sc.k.data());
            signed_4bit_b128_gemv(lw.W_v, sc.norm_buf.data(), sc.block_sums_dim.data(), sc.v.data());

            apply_rope(sc.q.data(), num_heads, head_dim, pos, rope_cos.data(), rope_sin.data());
            apply_rope(sc.k.data(), num_kv_heads, head_dim, pos, rope_cos.data(), rope_sin.data());
            std::memcpy(&kv.k[pos * kv_dim], sc.k.data(), kv_dim * sizeof(float));
            std::memcpy(&kv.v[pos * kv_dim], sc.v.data(), kv_dim * sizeof(float));

            // Coarsen parallel dispatch from 32 tasks to 4 worker tasks (8 heads per worker)
            // This eliminates 87.5% of GCD thread synchronization overhead per token.
            parallel_for(4, [&](size_t group) {
                size_t h_start = group * 8;
                size_t h_end = h_start + 8;
                float head_scores[MAX_SEQ_LEN];

                for (size_t h = h_start; h < h_end; ++h) {
                    size_t kv_h = h / 4;
                    const float* q_ptr = &sc.q[h * head_dim];

                    float max_score = -1e30f;
                    for (size_t t = 0; t <= pos; ++t) {
                        const float* k_ptr = &kv.k[t * kv_dim + kv_h * head_dim];
#if HAS_ARM_NEON
                        float32x4_t acc0 = vdupq_n_f32(0.0f);
                        float32x4_t acc1 = vdupq_n_f32(0.0f);
                        float32x4_t acc2 = vdupq_n_f32(0.0f);
                        float32x4_t acc3 = vdupq_n_f32(0.0f);

                        for (size_t d = 0; d < 64; d += 16) {
                            acc0 = vmlaq_f32(acc0, vld1q_f32(&q_ptr[d + 0]),  vld1q_f32(&k_ptr[d + 0]));
                            acc1 = vmlaq_f32(acc1, vld1q_f32(&q_ptr[d + 4]),  vld1q_f32(&k_ptr[d + 4]));
                            acc2 = vmlaq_f32(acc2, vld1q_f32(&q_ptr[d + 8]),  vld1q_f32(&k_ptr[d + 8]));
                            acc3 = vmlaq_f32(acc3, vld1q_f32(&q_ptr[d + 12]), vld1q_f32(&k_ptr[d + 12]));
                        }
                        float dot = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#else
                        float dot = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d) dot += q_ptr[d] * k_ptr[d];
#endif
                        float s = dot * 0.125f; // 1 / sqrt(64)
                        head_scores[t] = s;
                        if (s > max_score) max_score = s;
                    }

                    float sum_exp = 0.0f;
                    for (size_t t = 0; t <= pos; ++t) {
                        float e = std::exp(head_scores[t] - max_score);
                        head_scores[t] = e;
                        sum_exp += e;
                    }
                    float inv_sum = 1.0f / sum_exp;
                    for (size_t t = 0; t <= pos; ++t) head_scores[t] *= inv_sum;

                    float* out_ptr = &sc.attn_out[h * head_dim];
                    std::memset(out_ptr, 0, head_dim * sizeof(float));
                    for (size_t t = 0; t <= pos; ++t) {
                        const float* v_ptr = &kv.v[t * kv_dim + kv_h * head_dim];
                        float w = head_scores[t];
#if HAS_ARM_NEON
                        float32x4_t w_vec = vdupq_n_f32(w);
                        for (size_t d = 0; d < 64; d += 16) {
                            vst1q_f32(&out_ptr[d + 0],  vmlaq_f32(vld1q_f32(&out_ptr[d + 0]),  w_vec, vld1q_f32(&v_ptr[d + 0])));
                            vst1q_f32(&out_ptr[d + 4],  vmlaq_f32(vld1q_f32(&out_ptr[d + 4]),  w_vec, vld1q_f32(&v_ptr[d + 4])));
                            vst1q_f32(&out_ptr[d + 8],  vmlaq_f32(vld1q_f32(&out_ptr[d + 8]),  w_vec, vld1q_f32(&v_ptr[d + 8])));
                            vst1q_f32(&out_ptr[d + 12], vmlaq_f32(vld1q_f32(&out_ptr[d + 12]), w_vec, vld1q_f32(&v_ptr[d + 12])));
                        }
#else
                        for (size_t d = 0; d < 64; ++d) out_ptr[d] += w * v_ptr[d];
#endif
                    }
                }
            });

            compute_x_block_sums(sc.attn_out.data(), dim, sc.block_sums_dim.data());
            signed_4bit_b128_gemv(lw.W_o, sc.attn_out.data(), sc.block_sums_dim.data(), sc.proj_o.data());
            for (size_t i = 0; i < dim; ++i) sc.x[i] += sc.proj_o[i];

            rms_norm(sc.x.data(), lw.post_norm, sc.norm_buf.data(), dim);
            compute_x_block_sums(sc.norm_buf.data(), dim, sc.block_sums_dim.data());

            signed_4bit_b128_fused_swiglu(lw.W_gate, lw.W_up, sc.norm_buf.data(), sc.block_sums_dim.data(), sc.mlp_act.data());

            compute_x_block_sums(sc.mlp_act.data(), ffn_dim, sc.block_sums_ffn.data());
            signed_4bit_b128_gemv(lw.W_down, sc.mlp_act.data(), sc.block_sums_ffn.data(), sc.mlp_out.data());
            for (size_t i = 0; i < dim; ++i) sc.x[i] += sc.mlp_out[i];
        }

        if (compute_logits) {
            rms_norm(sc.x.data(), final_norm, sc.norm_buf.data(), dim);
            compute_x_block_sums(sc.norm_buf.data(), dim, sc.block_sums_dim.data());
            signed_4bit_b128_gemv(W_shared_lm_head, sc.norm_buf.data(), sc.block_sums_dim.data(), sc.logits.data());
        }
    }
};

// ============================================================================
// C API Exported Functions
// ============================================================================

extern "C" {

llama_engine_t llama_engine_create(const char* model_path) {
    if (!model_path) return nullptr;
    auto* engine = new LlamaEngineState();
    if (!engine->load(model_path)) {
        delete engine;
        return nullptr;
    }
    return static_cast<llama_engine_t>(engine);
}

void llama_engine_free(llama_engine_t engine) {
    if (engine) {
        delete static_cast<LlamaEngineState*>(engine);
    }
}

void llama_engine_reset(llama_engine_t engine) {
    if (engine) {
        static_cast<LlamaEngineState*>(engine)->reset();
    }
}

int32_t llama_engine_get_num_layers(llama_engine_t engine) {
    if (!engine) return 0;
    return static_cast<LlamaEngineState*>(engine)->num_layers;
}

int32_t llama_engine_get_dim(llama_engine_t engine) {
    if (!engine) return 0;
    return static_cast<LlamaEngineState*>(engine)->dim;
}

int32_t llama_engine_get_ffn_dim(llama_engine_t engine) {
    if (!engine) return 0;
    return static_cast<LlamaEngineState*>(engine)->ffn_dim;
}

int32_t llama_engine_get_vocab_size(llama_engine_t engine) {
    if (!engine) return 0;
    return static_cast<LlamaEngineState*>(engine)->vocab_size;
}

int32_t llama_engine_get_num_heads(llama_engine_t engine) {
    if (!engine) return 0;
    return static_cast<LlamaEngineState*>(engine)->num_heads;
}

int32_t llama_engine_get_num_kv_heads(llama_engine_t engine) {
    if (!engine) return 0;
    return static_cast<LlamaEngineState*>(engine)->num_kv_heads;
}

int32_t llama_engine_get_head_dim(llama_engine_t engine) {
    if (!engine) return 0;
    return static_cast<LlamaEngineState*>(engine)->head_dim;
}

int32_t llama_engine_get_max_seq_len(llama_engine_t engine) {
    return MAX_SEQ_LEN;
}

int llama_engine_forward_token(llama_engine_t engine, int32_t token_id, size_t pos, int compute_logits) {
    if (!engine) return -1;
    static_cast<LlamaEngineState*>(engine)->forward_token(token_id, pos, compute_logits != 0);
    return 0;
}

int llama_engine_prefill(llama_engine_t engine, const int32_t* token_ids, size_t count, size_t start_pos, int compute_last_logits) {
    if (!engine || !token_ids) return -1;
    auto* state = static_cast<LlamaEngineState*>(engine);
    for (size_t i = 0; i < count; ++i) {
        bool last = (i + 1 == count);
        bool comp = last && (compute_last_logits != 0);
        state->forward_token(token_ids[i], start_pos + i, comp);
    }
    return 0;
}

int32_t llama_engine_sample(llama_engine_t engine, float temperature, float top_p) {
    if (!engine) return 0;
    auto* state = static_cast<LlamaEngineState*>(engine);
    return sample_top_p(state->sc.logits.data(), state->vocab_size, temperature, top_p);
}

const float* llama_engine_get_logits(llama_engine_t engine) {
    if (!engine) return nullptr;
    return static_cast<LlamaEngineState*>(engine)->sc.logits.data();
}

void llama_engine_dequantize_embedding(llama_engine_t engine, size_t token_id, float* out_x) {
    if (!engine || !out_x) return;
    auto* state = static_cast<LlamaEngineState*>(engine);
    state->W_shared_lm_head.dequantize_embedding_row(token_id, out_x);
}

llama_tokenizer_t llama_tokenizer_create(const char* tokenizer_path) {
    if (!tokenizer_path) return nullptr;
    auto* tok = new LLaMATokenizer();
    if (!tok->load(tokenizer_path)) {
        delete tok;
        return nullptr;
    }
    return static_cast<llama_tokenizer_t>(tok);
}

void llama_tokenizer_free(llama_tokenizer_t tokenizer) {
    if (tokenizer) {
        delete static_cast<LLaMATokenizer*>(tokenizer);
    }
}

int llama_tokenizer_encode(llama_tokenizer_t tokenizer, const char* text, int32_t* out_tokens, int max_tokens) {
    if (!tokenizer || !text || !out_tokens || max_tokens <= 0) return 0;
    auto* tok = static_cast<LLaMATokenizer*>(tokenizer);
    auto ids = tok->encode(text);
    int count = std::min<int>(max_tokens, static_cast<int>(ids.size()));
    for (int i = 0; i < count; ++i) {
        out_tokens[i] = ids[i];
    }
    return count;
}

const char* llama_tokenizer_decode(llama_tokenizer_t tokenizer, int32_t token_id) {
    if (!tokenizer) return "";
    auto* tok = static_cast<LLaMATokenizer*>(tokenizer);
    tok->last_decoded_str = tok->decode(token_id);
    return tok->last_decoded_str.c_str();
}

} // extern "C"
