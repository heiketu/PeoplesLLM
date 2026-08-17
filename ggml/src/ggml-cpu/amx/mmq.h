#pragma once
#include "common.h"

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst);

size_t ggml_backend_amx_mmid_desired_wsize(int n_threads, const struct ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor);

void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);

void ggml_backend_amx_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

void ggml_backend_amx_mul_mat_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);
