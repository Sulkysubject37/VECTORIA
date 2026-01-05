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

void vectoria_graph_set_output(vectoria_graph_t g, int node_id);

// --- Engine Execution ---
vectoria_engine_t vectoria_engine_create(vectoria_graph_t g);
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
    int64_t* node_id, // -1 if none
    char* details_buffer, 
    size_t buffer_len
);

#ifdef __cplusplus
}
#endif
