#ifndef LLAMA_ENGINE_H
#define LLAMA_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque engine and tokenizer types
typedef void* llama_engine_t;
typedef void* llama_tokenizer_t;

// ============================================================================
// Model Engine API
// ============================================================================

// Initialize engine from binary model file path. Returns NULL on failure.
llama_engine_t llama_engine_create(const char* model_path);

// Free engine resources
void llama_engine_free(llama_engine_t engine);

// Reset KV cache and internal scratch state for a new session
void llama_engine_reset(llama_engine_t engine);

// Model hyperparameter getters
int32_t llama_engine_get_num_layers(llama_engine_t engine);
int32_t llama_engine_get_dim(llama_engine_t engine);
int32_t llama_engine_get_ffn_dim(llama_engine_t engine);
int32_t llama_engine_get_vocab_size(llama_engine_t engine);
int32_t llama_engine_get_num_heads(llama_engine_t engine);
int32_t llama_engine_get_num_kv_heads(llama_engine_t engine);
int32_t llama_engine_get_head_dim(llama_engine_t engine);
int32_t llama_engine_get_max_seq_len(llama_engine_t engine);

// Forward single token at sequence position pos.
// compute_logits: 1 to compute final logits, 0 to skip (prefill optimization)
int llama_engine_forward_token(llama_engine_t engine, int32_t token_id, size_t pos, int compute_logits);

// Batch prefill: forward an array of tokens sequentially from start_pos without computing logits
// except optionally for the very last token if compute_last_logits != 0.
int llama_engine_prefill(llama_engine_t engine, const int32_t* token_ids, size_t count, size_t start_pos, int compute_last_logits);

// Sample token from latest logits using temperature and top_p (greedy if temp <= 0.01)
int32_t llama_engine_sample(llama_engine_t engine, float temperature, float top_p);

// Direct read-only pointer to latest float logits buffer (size = vocab_size)
const float* llama_engine_get_logits(llama_engine_t engine);

// Dequantize embedding row for a given token id into out_x (size = dim floats)
void llama_engine_dequantize_embedding(llama_engine_t engine, size_t token_id, float* out_x);

// ============================================================================
// Built-in Tokenizer API
// ============================================================================

llama_tokenizer_t llama_tokenizer_create(const char* tokenizer_path);
void llama_tokenizer_free(llama_tokenizer_t tokenizer);
int llama_tokenizer_encode(llama_tokenizer_t tokenizer, const char* text, int32_t* out_tokens, int max_tokens);
const char* llama_tokenizer_decode(llama_tokenizer_t tokenizer, int32_t token_id);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_ENGINE_H
