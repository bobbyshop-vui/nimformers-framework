/*
 * nimformers.h — stable C ABI for the Nimformers framework.
 *
 * Build the shared library with:
 *     make build-for-python
 *
 * ABI rules
 * ---------
 *  - Return value 0 means success, 1 means failure (or NULL / -1 for pointer
 *    and length queries).  Read the message with nimf_last_error().
 *  - Handles are opaque; create them with *_create / *_load / *_upload and
 *    release them with the matching *_destroy / *_free.
 *  - All float data is IEEE-754 binary32.  Shapes and lengths are int64_t.
 *  - Output buffers are caller-allocated and must be large enough.
 *  - Strings returned by nimf_state_dict_name(), nimf_tokenizer_decode(),
 *    nimf_sentencepiece_decode(), nimf_sentencepiece_normalizer(),
 *    nimf_vm_lower_msl(), nimf_vm_lower_glsl() and nimf_vm_key() are
 *    allocated by the library: free them with nimf_free_string().
 *    nimf_last_error(), nimf_version() and nimf_context_backend_name()
 *    return static/thread-local strings and must not be freed.
 *  - The library is not safe for concurrent GPU calls on one context.
 *    The error string is thread-local.
 *  - Payload pointers returned by nimf_quant_data() / nimf_quant_scale() /
 *    nimf_quant_zero() may point into a read-only mmap when a state dict was
 *    loaded with mapped != 0; treat them as const.
 */
#ifndef NIMFORMERS_H
#define NIMFORMERS_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#  if defined(NIMF_BUILD)
#    define NIMF_API __declspec(dllexport)
#  else
#    define NIMF_API __declspec(dllimport)
#  endif
#  define NIMF_CALL __cdecl
#else
#  define NIMF_API __attribute__((visibility("default")))
#  define NIMF_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NIMF_ABI_VERSION 2

/* Backend selector for nimf_context_create(). */
#define NIMF_BACKEND_AUTO   0
#define NIMF_BACKEND_METAL  1
#define NIMF_BACKEND_VULKAN 2
#define NIMF_BACKEND_CUDA   3  /* legacy alias; redirected to Vulkan */
#define NIMF_BACKEND_CPU    4  /* runtime-dispatched SIMD CPU backend */

/* Quantization kinds for nimf_quantize(). */
#define NIMF_QK_FP32_RAW        0
#define NIMF_QK_INT8            1
#define NIMF_QK_INT4            2
#define NIMF_QK_INT4_ASYMMETRIC 3
#define NIMF_QK_FP8_E4M3        4
#define NIMF_QK_FP8_E5M2        5
#define NIMF_QK_CUSTOM          6
#define NIMF_QK_AUTO            7
#define NIMF_QK_FP4_KM          8

typedef struct NimfContext      NimfContext;
typedef struct NimfTensor       NimfTensor;
typedef struct NimfQuantTensor  NimfQuantTensor;
typedef struct NimfStateDict    NimfStateDict;
typedef struct NimfVmProgram    NimfVmProgram;
typedef struct NimfResidentQ4   NimfResidentQ4;
typedef struct NimfResidentF32  NimfResidentF32;
typedef struct NimfTokenizer    NimfTokenizer;
typedef struct NimfSentencePiece NimfSentencePiece;
typedef struct NimfLlama        NimfLlama;

/* ============================== runtime ============================== */

NIMF_API uint32_t    NIMF_CALL nimf_abi_version(void);
NIMF_API int32_t     NIMF_CALL nimf_cpu_isa(void);
NIMF_API const char* NIMF_CALL nimf_cpu_isa_name(void);
NIMF_API uint32_t    NIMF_CALL nimf_cpu_features(void);
NIMF_API int32_t     NIMF_CALL nimf_cpu_cache_sizes(int64_t* l1, int64_t* l2, int64_t* l3);
NIMF_API int32_t     NIMF_CALL nimf_cpu_block_sizes(int64_t* mc, int64_t* nc, int64_t* kc);
NIMF_API const char* NIMF_CALL nimf_version(void);
NIMF_API const char* NIMF_CALL nimf_last_error(void);
NIMF_API void        NIMF_CALL nimf_clear_error(void);
NIMF_API void        NIMF_CALL nimf_free_string(char* value);

NIMF_API NimfContext* NIMF_CALL nimf_context_create(int32_t backend_kind);
NIMF_API NimfContext* NIMF_CALL nimf_context_create_name(const char* backend_name);
NIMF_API int32_t      NIMF_CALL nimf_context_destroy(NimfContext* ctx);
NIMF_API int32_t      NIMF_CALL nimf_context_backend(NimfContext* ctx);
NIMF_API const char*  NIMF_CALL nimf_context_backend_name(NimfContext* ctx);
NIMF_API int32_t      NIMF_CALL nimf_context_sync(NimfContext* ctx);
NIMF_API int32_t      NIMF_CALL nimf_context_set_cpu_fallback(NimfContext* ctx, int32_t allow);

/* ============================== tensors ============================== */

NIMF_API NimfTensor* NIMF_CALL nimf_tensor_create(const float* data, int64_t n);
NIMF_API NimfTensor* NIMF_CALL nimf_tensor_create_shaped(const float* data, int64_t n,
                                                          const int64_t* shape, int64_t rank);
NIMF_API NimfTensor* NIMF_CALL nimf_tensor_zeros(const int64_t* shape, int64_t rank, float fill);
NIMF_API NimfTensor* NIMF_CALL nimf_tensor_randn(const int64_t* shape, int64_t rank, float scale);
NIMF_API NimfTensor* NIMF_CALL nimf_tensor_clone(NimfTensor* t);
NIMF_API int32_t     NIMF_CALL nimf_tensor_destroy(NimfTensor* t);
NIMF_API int64_t     NIMF_CALL nimf_tensor_rank(NimfTensor* t);
NIMF_API int64_t     NIMF_CALL nimf_tensor_numel(NimfTensor* t);
NIMF_API int64_t     NIMF_CALL nimf_tensor_shape(NimfTensor* t, int64_t* out_shape, int64_t cap);
NIMF_API float*      NIMF_CALL nimf_tensor_data(NimfTensor* t);
NIMF_API int64_t     NIMF_CALL nimf_tensor_len(NimfTensor* t);
NIMF_API int32_t     NIMF_CALL nimf_tensor_set_data(NimfTensor* t, const float* data, int64_t n);
NIMF_API int32_t     NIMF_CALL nimf_tensor_fill(NimfTensor* t, float value);
NIMF_API int32_t     NIMF_CALL nimf_tensor_reshape(NimfTensor* t, const int64_t* shape, int64_t rank);

/* ============================ elementwise ============================ */

NIMF_API int32_t NIMF_CALL nimf_add(NimfContext*, const float* a, const float* b, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_sub(NimfContext*, const float* a, const float* b, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_mul(NimfContext*, const float* a, const float* b, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_div(NimfContext*, const float* a, const float* b, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_relu(NimfContext*, const float* x, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_sqrt(NimfContext*, const float* x, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_exp(NimfContext*, const float* x, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_log(NimfContext*, const float* x, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_sigmoid(NimfContext*, const float* x, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_tanh(NimfContext*, const float* x, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_neg(NimfContext*, const float* x, float* out, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_pow(NimfContext*, const float* x, float* out, int64_t n, float exponent);
NIMF_API int32_t NIMF_CALL nimf_add_scalar(NimfContext*, const float* x, float* out, int64_t n, float scalar);
NIMF_API int32_t NIMF_CALL nimf_mul_scalar(NimfContext*, const float* x, float* out, int64_t n, float scalar);
NIMF_API int32_t NIMF_CALL nimf_apflu(NimfContext*, const float* x, float* out, int64_t n,
                                       float alpha, float beta);
NIMF_API int32_t NIMF_CALL nimf_apflu_backward(NimfContext*, const float* x, const float* dy,
                                                float* out, int64_t n, float alpha, float beta);

/* ============================== matmul =============================== */

NIMF_API int32_t NIMF_CALL nimf_matmul(NimfContext*, const float* a, const float* b, float* out,
                                        int64_t m, int64_t k, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_matmul2(NimfContext*,
                                         const float* a1, const float* b1, float* out1,
                                         int64_t m1, int64_t k1, int64_t n1,
                                         const float* a2, const float* b2, float* out2,
                                         int64_t m2, int64_t k2, int64_t n2);
NIMF_API int32_t NIMF_CALL nimf_matmul_nt(NimfContext*, const float* a, const float* b, float* out,
                                           int64_t m, int64_t k, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_matmul_q4(NimfContext*, const float* a, int64_t a_len,
                                           const uint8_t* packed_weight, int64_t packed_len,
                                           const float* scales, const float* zeros,
                                           int64_t scale_len, int64_t zero_len,
                                           float* out, int64_t m, int64_t k, int64_t n,
                                           int64_t group_size, int64_t groups_per_row);

/* ========================== resident weights ========================= */

NIMF_API NimfResidentQ4* NIMF_CALL nimf_resident_q4_upload(NimfContext*,
    const uint8_t* packed, int64_t packed_len, const float* scales, const float* zeros,
    int64_t scale_len, int64_t zero_len, int64_t k, int64_t n,
    int64_t group_size, int64_t groups_per_row);
NIMF_API int32_t NIMF_CALL nimf_resident_q4_linear(NimfContext*, NimfResidentQ4* weight,
    const float* x, int64_t x_len, int64_t m, float* out);
NIMF_API int32_t NIMF_CALL nimf_resident_q4_destroy(NimfContext*, NimfResidentQ4* weight);

NIMF_API NimfResidentF32* NIMF_CALL nimf_resident_f32_upload(NimfContext*,
    const float* weights, int64_t n);
NIMF_API int32_t NIMF_CALL nimf_resident_f32_linear(NimfContext*, NimfResidentF32* weight,
    const float* x, int64_t x_len, int64_t m, int64_t k, int64_t n, float* out);
NIMF_API int32_t NIMF_CALL nimf_resident_f32_destroy(NimfContext*, NimfResidentF32* weight);

/* ========================== neural kernels =========================== */

NIMF_API int32_t NIMF_CALL nimf_softmax(NimfContext*, const float* x, float* out,
                                         int64_t rows, int64_t cols);
NIMF_API int32_t NIMF_CALL nimf_layernorm(NimfContext*, const float* x,
    const float* gamma, const float* beta, float* out,
    int64_t rows, int64_t cols, float eps);
NIMF_API int32_t NIMF_CALL nimf_add_layernorm(NimfContext*,
    const float* a, const float* b, const float* gamma, const float* beta,
    float* summed, float* normed, int64_t rows, int64_t cols, float eps);
NIMF_API int32_t NIMF_CALL nimf_layernorm_backward(NimfContext*,
    const float* dy, const float* x, const float* gamma, const float* beta,
    int64_t rows, int64_t cols, float eps,
    float* dx, float* dgamma, float* dbeta);
NIMF_API int32_t NIMF_CALL nimf_attention(NimfContext*,
    const float* q, const float* k, const float* v, const float* mask, int64_t mask_len,
    int64_t batch, int64_t heads, int64_t seq_len, int64_t head_dim, float scale,
    float* out, float* scores);
NIMF_API int32_t NIMF_CALL nimf_attention_backward(NimfContext*,
    const float* q, const float* k, const float* v, const float* scores, const float* dy,
    int64_t batch, int64_t heads, int64_t seq_len, int64_t head_dim, float scale,
    float* dq, float* dk, float* dv);
NIMF_API int32_t NIMF_CALL nimf_embedding_lookup(NimfContext*, const float* table,
    int64_t table_len, const int32_t* indices, int64_t index_count, float* out,
    int64_t vocab, int64_t dim);
NIMF_API int32_t NIMF_CALL nimf_gather(NimfContext*, const float* src, int64_t src_len,
    const int32_t* indices, int64_t index_count, float* out);
NIMF_API int32_t NIMF_CALL nimf_compare(NimfContext*, const float* a, const float* b,
    int64_t n, int32_t op, uint8_t* out_mask);
NIMF_API int32_t NIMF_CALL nimf_dropout(NimfContext*, const float* x, float* out,
    int64_t n, float probability, int32_t training, uint32_t seed);
NIMF_API int32_t NIMF_CALL nimf_dropout_mask(NimfContext*, int64_t n,
    float probability, uint32_t seed, float* out);
NIMF_API int32_t NIMF_CALL nimf_argmax(NimfContext*, const float* x, int64_t x_len,
    int64_t outer, int64_t dim_size, int64_t inner,
    float* out_values, int64_t* out_indices);
NIMF_API int32_t NIMF_CALL nimf_masked_fill(const float* x, const uint8_t* mask,
    int64_t n, float value, float* out);
NIMF_API int32_t NIMF_CALL nimf_where(const uint8_t* cond, const float* a, const float* b,
    int64_t n, float* out);

/* ========================= reductions / scans ======================== */

NIMF_API int32_t NIMF_CALL nimf_sum(NimfContext*, const float* x, int64_t x_len,
    const int64_t* shape, int64_t rank, int64_t dim, float* out);
NIMF_API int32_t NIMF_CALL nimf_mean(NimfContext*, const float* x, int64_t x_len,
    const int64_t* shape, int64_t rank, int64_t dim, float* out);
NIMF_API int32_t NIMF_CALL nimf_max(NimfContext*, const float* x, int64_t x_len,
    const int64_t* shape, int64_t rank, int64_t dim, float* out);
NIMF_API int32_t NIMF_CALL nimf_cumsum(NimfContext*, const float* x, int64_t x_len,
    const int64_t* shape, int64_t rank, int64_t dim, float* out);
NIMF_API int32_t NIMF_CALL nimf_cumprod(NimfContext*, const float* x, int64_t x_len,
    const int64_t* shape, int64_t rank, int64_t dim, float* out);
NIMF_API int32_t NIMF_CALL nimf_affine_scan(NimfContext*, const float* a_coef,
    const float* b_coef, int64_t n, const int64_t* shape, int64_t rank,
    int64_t dim, float* out);

/* =========================== shape movement ========================== */

NIMF_API int32_t NIMF_CALL nimf_permute(NimfContext*, const float* src, int64_t n,
    const int32_t* new_strides, const int32_t* old_strides, const int32_t* dims_map,
    int64_t rank, float* out);
NIMF_API int32_t NIMF_CALL nimf_split_slice(NimfContext*, const float* src,
    int64_t dim_total, int64_t this_size, int64_t offset, int64_t inner,
    int64_t n, float* out);
NIMF_API int32_t NIMF_CALL nimf_concat(NimfContext*, const float* src, int64_t src_len,
    const int32_t* src_base, const int32_t* src_dim_size, const int32_t* src_dim_offset,
    int64_t count, int64_t total_dim, int64_t inner, int64_t out_len, float* out);

/* ============================ quantization =========================== */

NIMF_API NimfQuantTensor* NIMF_CALL nimf_quantize(const float* x, int64_t n,
    const int64_t* shape, int64_t rank, int32_t kind,
    int32_t exponent_bits, int32_t mantissa_bits);
NIMF_API int64_t NIMF_CALL nimf_quant_dequantize(NimfQuantTensor*, float* out, int64_t capacity);
NIMF_API int64_t NIMF_CALL nimf_quant_dequantize_transposed(NimfQuantTensor*, float* out,
                                                             int64_t capacity);
NIMF_API int32_t NIMF_CALL nimf_quant_kind(NimfQuantTensor*);
NIMF_API int64_t NIMF_CALL nimf_quant_rank(NimfQuantTensor*);
NIMF_API int64_t NIMF_CALL nimf_quant_shape(NimfQuantTensor*, int64_t* out_shape, int64_t cap);
NIMF_API int64_t NIMF_CALL nimf_quant_numel(NimfQuantTensor*);
NIMF_API int64_t NIMF_CALL nimf_quant_group_size(NimfQuantTensor*);
NIMF_API int32_t NIMF_CALL nimf_quant_cf_bits(NimfQuantTensor*, int32_t* exponent_bits,
                                                int32_t* mantissa_bits);
NIMF_API int64_t NIMF_CALL nimf_quant_data_len(NimfQuantTensor*);
NIMF_API const uint8_t* NIMF_CALL nimf_quant_data(NimfQuantTensor*);
NIMF_API int64_t NIMF_CALL nimf_quant_scale_len(NimfQuantTensor*);
NIMF_API const float* NIMF_CALL nimf_quant_scale(NimfQuantTensor*);
NIMF_API int64_t NIMF_CALL nimf_quant_zero_len(NimfQuantTensor*);
NIMF_API const float* NIMF_CALL nimf_quant_zero(NimfQuantTensor*);
NIMF_API int32_t NIMF_CALL nimf_quant_destroy(NimfQuantTensor*);

/* ============================ state dicts ============================ */

NIMF_API NimfStateDict* NIMF_CALL nimf_state_dict_load(const char* path, int32_t mapped);
NIMF_API NimfStateDict* NIMF_CALL nimf_state_dict_create(void);
NIMF_API int32_t NIMF_CALL nimf_state_dict_append(NimfStateDict*, const char* name,
                                                   NimfQuantTensor* tensor);
NIMF_API int32_t NIMF_CALL nimf_state_dict_set_arch(NimfStateDict*, const int64_t* arch);
NIMF_API int32_t NIMF_CALL nimf_state_dict_save(const char* path, NimfStateDict*,
                                                 const int64_t* arch);
NIMF_API int64_t NIMF_CALL nimf_state_dict_count(NimfStateDict*);
NIMF_API int32_t NIMF_CALL nimf_state_dict_arch(NimfStateDict*, int64_t* out_arch, int64_t cap);
NIMF_API char*   NIMF_CALL nimf_state_dict_name(NimfStateDict*, int64_t index);
NIMF_API NimfQuantTensor* NIMF_CALL nimf_state_dict_tensor(NimfStateDict*, int64_t index);
NIMF_API int32_t NIMF_CALL nimf_state_dict_destroy(NimfStateDict*);

/* ======================= custom floats / APF ========================= */

NIMF_API int32_t NIMF_CALL nimf_customfloat_encode(NimfContext*, const float* x, int64_t n,
    int32_t exponent_bits, int32_t mantissa_bits,
    uint8_t* out_bytes, int64_t capacity, int64_t* written);
NIMF_API int32_t NIMF_CALL nimf_customfloat_decode(NimfContext*, const uint8_t* bytes,
    int64_t byte_len, int64_t item_count, int32_t exponent_bits, int32_t mantissa_bits,
    float* out);
NIMF_API int32_t NIMF_CALL nimf_apf_build_dtype(const float* x, int64_t n,
    const float* grad, int64_t grad_len, float rel_error_tol, int32_t exp_margin,
    int32_t* exponent_bits, int32_t* mantissa_bits);
NIMF_API int32_t NIMF_CALL nimf_apf_cast_training(const float* x, int64_t n,
    const float* grad, int64_t grad_len, float rel_error_tol, int32_t exp_margin,
    uint8_t* out_bytes, int64_t capacity, int64_t* written,
    int32_t* exponent_bits, int32_t* mantissa_bits);
NIMF_API int64_t NIMF_CALL nimf_apf_decode_training(const uint8_t* bytes, int64_t byte_len,
    int32_t exponent_bits, int32_t mantissa_bits, float* out, int64_t capacity);

/* ============================== micro-VM ============================= */

NIMF_API NimfVmProgram* NIMF_CALL nimf_vm_create(void);
NIMF_API int32_t NIMF_CALL nimf_vm_destroy(NimfVmProgram*);
NIMF_API int64_t NIMF_CALL nimf_vm_len(NimfVmProgram*);
NIMF_API int64_t NIMF_CALL nimf_vm_words(NimfVmProgram*, uint32_t* out_words, int64_t cap);
NIMF_API int32_t NIMF_CALL nimf_vm_set_words(NimfVmProgram*, const uint32_t* words, int64_t count);
NIMF_API int32_t NIMF_CALL nimf_vm_emit_word(NimfVmProgram*, uint32_t word);
NIMF_API int32_t NIMF_CALL nimf_vm_emit(NimfVmProgram*, uint32_t opcode,
    uint32_t a, uint32_t b, uint32_t c, uint32_t value);
NIMF_API int32_t NIMF_CALL nimf_vm_save(NimfVmProgram*, const char* path);
NIMF_API NimfVmProgram* NIMF_CALL nimf_vm_load(const char* path);
NIMF_API NimfVmProgram* NIMF_CALL nimf_vm_merge(NimfVmProgram* const* programs, int64_t count);
NIMF_API char* NIMF_CALL nimf_vm_lower_msl(NimfVmProgram*, const char* kernel_name);
NIMF_API char* NIMF_CALL nimf_vm_lower_glsl(NimfVmProgram*, const char* kernel_name);
NIMF_API char* NIMF_CALL nimf_vm_key(NimfVmProgram*, const char* tag);
NIMF_API int32_t NIMF_CALL nimf_vm_run(NimfContext*, NimfVmProgram*, float* memory,
    int64_t memory_len, int64_t n_threads, int64_t tg_floats, int64_t tg_size);
NIMF_API int32_t NIMF_CALL nimf_micro_vm_dispatch(NimfContext*, const uint32_t* code,
    int64_t code_len, float* memory, int64_t memory_len,
    int64_t n_threads, int64_t tg_floats, int64_t tg_size);

/* ============================= tokenizers ============================ */

NIMF_API NimfTokenizer* NIMF_CALL nimf_tokenizer_load(const char* model_dir);
NIMF_API int32_t NIMF_CALL nimf_tokenizer_destroy(NimfTokenizer*);
NIMF_API int64_t NIMF_CALL nimf_tokenizer_vocab_size(NimfTokenizer*);
NIMF_API int64_t NIMF_CALL nimf_tokenizer_total_vocab_size(NimfTokenizer*);
NIMF_API int32_t NIMF_CALL nimf_tokenizer_special_ids(NimfTokenizer*,
    int64_t* bos, int64_t* eos, int64_t* pad);
NIMF_API int64_t NIMF_CALL nimf_tokenizer_encode(NimfTokenizer*, const char* text,
    int32_t add_special, int32_t* out_ids, int64_t capacity);
NIMF_API int64_t NIMF_CALL nimf_tokenizer_apply_chat(NimfTokenizer*,
    const char* user_text, const char* system_text, int32_t* out_ids, int64_t capacity);
NIMF_API char* NIMF_CALL nimf_tokenizer_decode(NimfTokenizer*, const int32_t* ids,
    int64_t count, int32_t skip_special);

NIMF_API NimfSentencePiece* NIMF_CALL nimf_sentencepiece_load(const char* path);
NIMF_API int32_t NIMF_CALL nimf_sentencepiece_destroy(NimfSentencePiece*);
NIMF_API int64_t NIMF_CALL nimf_sentencepiece_encode(NimfSentencePiece*, const char* text,
    int32_t* out_ids, int64_t capacity);
NIMF_API char* NIMF_CALL nimf_sentencepiece_decode(NimfSentencePiece*,
    const int32_t* ids, int64_t count);
NIMF_API int64_t NIMF_CALL nimf_sentencepiece_piece_id(NimfSentencePiece*, const char* piece);
NIMF_API char* NIMF_CALL nimf_sentencepiece_normalizer(NimfSentencePiece*);

/* =============================== Llama =============================== */

NIMF_API int32_t NIMF_CALL nimf_llama_config_load(const char* config_path,
    int64_t* out_ints, float* out_floats);
NIMF_API NimfLlama* NIMF_CALL nimf_llama_load(NimfContext*, const char* config_path,
    const char* weights_path, int32_t upload_resident);
NIMF_API int32_t NIMF_CALL nimf_llama_config(NimfLlama*, int64_t* out_ints, float* out_floats);
NIMF_API char* NIMF_CALL nimf_llama_generate(NimfContext*, NimfLlama*, NimfTokenizer*,
    const char* prompt, const char* system_text, int64_t max_new_tokens,
    float temperature, float top_p, int32_t use_chat_template);
NIMF_API int32_t NIMF_CALL nimf_llama_destroy(NimfLlama*);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* NIMFORMERS_H */
