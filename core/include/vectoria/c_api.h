#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vectoria_graph_t;
typedef void* vectoria_engine_t;

// --- Graph Construction ---
vectoria_graph_t vectoria_graph_create();
void vectoria_graph_destroy(vectoria_graph_t g);

// Returns node index or -1 on error
int vectoria_graph_add_input(vectoria_graph_t g, const char* name, const int64_t* shape, int rank, int dtype);
int vectoria_graph_add_parameter(vectoria_graph_t g, const char* name, const int64_t* shape, int rank, int dtype);
int vectoria_graph_add_op_matmul(vectoria_graph_t g, int input_a, int input_b);
int vectoria_graph_add_op_bias_add(vectoria_graph_t g, int input, int bias);
int vectoria_graph_add_op_relu(vectoria_graph_t g, int input);
int vectoria_graph_add_op_add(vectoria_graph_t g, int input_a, int input_b);
int vectoria_graph_add_op_mul(vectoria_graph_t g, int input_a, int input_b);
int vectoria_graph_add_op_reduce_sum(vectoria_graph_t g, int input);
int vectoria_graph_add_op_transpose(vectoria_graph_t g, int input, const int64_t* perm, int rank);
int vectoria_graph_add_op_reshape(vectoria_graph_t g, int input, const int64_t* new_shape, int rank);
int vectoria_graph_add_op_concat(vectoria_graph_t g, const int* inputs, int num_inputs, int64_t axis);
int vectoria_graph_add_softmax(vectoria_graph_t g, int input);
int vectoria_graph_add_softmax_stable(vectoria_graph_t g, int input);
int vectoria_graph_add_logsoftmax(vectoria_graph_t g, int input);
int vectoria_graph_add_crossentropy(vectoria_graph_t g, int logits, int target);
int vectoria_graph_add_attention(vectoria_graph_t g, int q, int k, int v);
int vectoria_graph_add_multi_head_attention(vectoria_graph_t g, int x, int wq, int wk, int wv, int wo, int num_heads);
int vectoria_graph_add_transformer_encoder(
    vectoria_graph_t g, int x,
    int wq, int wk, int wv, int wo, int num_heads,
    int gamma1, int beta1,
    int w1, int b1, int w2, int b2,
    int gamma2, int beta2
);
int vectoria_graph_add_layernorm(vectoria_graph_t g, int input, int gamma, int beta);

void vectoria_graph_set_output(vectoria_graph_t g, int node_id);

// --- Lowering ---
// Returns 0 on success, -1 on failure
int vectoria_export_coreml(vectoria_graph_t g, const char* output_path);

// --- Engine Execution ---
vectoria_engine_t vectoria_engine_create(vectoria_graph_t g);
vectoria_engine_t vectoria_engine_create_with_policy(vectoria_graph_t g, int policy);
void vectoria_engine_destroy(vectoria_engine_t e);

void vectoria_engine_compile(vectoria_engine_t e);
void vectoria_engine_execute(vectoria_engine_t e);

// Returns pointer to raw buffer, or NULL if invalid
void* vectoria_engine_get_buffer(vectoria_engine_t e, int node_id);

// --- Observability ---
size_t vectoria_engine_get_trace_size(vectoria_engine_t e);

// Populates struct fields. Caller must handle strings? 
// Simplification: Copy data to fixed struct or individual getters.
// Let's use individual getters for safety and simplicity across FFI boundaries.
void vectoria_engine_get_trace_event(
    vectoria_engine_t e, 
    size_t index, 
    int* type, 
    uint64_t* timestamp_ns, 
    int64_t* node_id, 
    char* details_buffer, 
    size_t buffer_len
);

// --- Capabilities ---
void vectoria_get_capabilities(
    int* arch, // 0=Unk, 1=x86, 2=ARM
    int* simd_compiled, 
    int* simd_supported,
    char* arch_name_buffer,
    size_t arch_name_len
);

#ifdef __cplusplus
}
#endif
