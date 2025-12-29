#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace marionette {
namespace optimization {

// Forward declaration
class adam_optimizer;

/**
 * @brief tensor_options for configuring tensor creation
 */
class tensor_options {
 public:
  tensor_options();
  ~tensor_options();

  // Friend declaration
  friend class tensor_t;

  // Factory methods for dtype
  static tensor_options float32();
  static tensor_options float64();

  // Device selection
  tensor_options device_cpu() const;
  tensor_options device_cuda(int device_index = 0) const;

 private:
  explicit tensor_options(torch::TensorOptions options);
  const torch::TensorOptions& torch_options() const;
  torch::TensorOptions options_;
};

/**
 * @brief tensor_t class for automatic differentiation with backward propagation
 *
 * This class provides a clean interface for tensor operations with autograd support.
 */
class tensor_t {
 public:
  tensor_t();
  ~tensor_t();

  // Friend declarations for global operators and internal classes
  friend class adam_optimizer;
  friend tensor_t operator-(float scalar, const tensor_t& tensor);
  friend tensor_t operator-(double scalar, const tensor_t& tensor);
  friend tensor_t operator*(float scalar, const tensor_t& tensor);
  friend tensor_t operator*(double scalar, const tensor_t& tensor);
  friend tensor_t operator+(float scalar, const tensor_t& tensor);
  friend tensor_t operator+(double scalar, const tensor_t& tensor);

  // Copy and move operations
  tensor_t(const tensor_t& other) = default;
  tensor_t(tensor_t&& other) noexcept = default;
  tensor_t& operator=(const tensor_t& other) & = default;
  tensor_t& operator=(tensor_t&& other) & noexcept = default;
  tensor_t&& operator=(const tensor_t& other) &&;
  tensor_t&& operator=(tensor_t&& other) && noexcept;

  // Factory methods
  static tensor_t zeros(const std::vector<int64_t>& shape);
  static tensor_t zeros(const std::vector<int64_t>& shape, const tensor_options& options);
  static tensor_t ones(const std::vector<int64_t>& shape);
  static tensor_t ones(const std::vector<int64_t>& shape, const tensor_options& options);
  static tensor_t eye(int64_t n);
  static tensor_t eye(int64_t n, const tensor_options& options);
  static tensor_t zeros_like(const tensor_t& other);
  static tensor_t from_blob(const float* data, const std::vector<int64_t>& shape);
  static tensor_t from_blob(const double* data, const std::vector<int64_t>& shape);

  // Shape and size operations
  std::vector<int64_t> sizes() const;
  int64_t size(int64_t dim) const;
  int64_t dim() const;
  int64_t numel() const;

  // Data type operations
  tensor_t to_float32() const;
  tensor_t to_float64() const;

  // Options (for tensor creation)
  tensor_options options() const;

  // Autograd operations
  tensor_t requires_grad(bool requires_grad = true) const;
  tensor_t requires_grad(bool requires_grad = true);
  bool requires_grad() const;
  tensor_t detach() const;
  void backward() const;
  tensor_t grad() const;
  void zero_grad();

  // Validity check
  bool defined() const;

  // Clone and reshape operations
  tensor_t clone() const;
  tensor_t view(const std::vector<int64_t>& shape) const;
  tensor_t reshape(const std::vector<int64_t>& shape) const;
  tensor_t unsqueeze(int64_t dim) const;
  tensor_t squeeze(int64_t dim) const;
  tensor_t expand(const std::vector<int64_t>& shape) const;
  tensor_t transpose(int64_t dim0, int64_t dim1) const;
  tensor_t narrow(int64_t dim, int64_t start, int64_t length) const;

  // Indexing operations
  tensor_t select(int64_t dim, int64_t index) const;

  // Unary operators
  tensor_t operator-() const;

  // Element-wise arithmetic operations
  tensor_t operator+(const tensor_t& other) const;
  tensor_t operator-(const tensor_t& other) const;
  tensor_t operator*(const tensor_t& other) const;
  tensor_t operator/(const tensor_t& other) const;
  tensor_t operator+(float scalar) const;
  tensor_t operator-(float scalar) const;
  tensor_t operator*(float scalar) const;
  tensor_t operator/(float scalar) const;

  // In-place operations
  tensor_t& operator+=(const tensor_t& other);
  tensor_t& operator-=(const tensor_t& other);
  tensor_t& operator*=(const tensor_t& other);
  tensor_t& operator/=(const tensor_t& other);
  tensor_t& operator-=(float scalar);
  tensor_t& operator+=(float scalar);

  // Reduction operations
  tensor_t sum() const;
  tensor_t sum(int64_t dim, bool keepdim = false) const;
  tensor_t mean() const;
  tensor_t mean(int64_t dim, bool keepdim = false) const;
  tensor_t min(int64_t dim) const;
  tensor_t max(int64_t dim) const;

  // Mathematical operations
  tensor_t sqrt() const;
  tensor_t cos() const;
  tensor_t sin() const;
  tensor_t norm(int64_t p, int64_t dim) const;

  // Matrix operations
  static tensor_t matmul(const tensor_t& a, const tensor_t& b);
  static tensor_t bmm(const tensor_t& a, const tensor_t& b);
  static tensor_t det(const tensor_t& a);
  static tensor_t inverse(const tensor_t& a);
  tensor_t t() const;  // Transpose

  // Concatenation and stacking
  static tensor_t cat(const std::vector<tensor_t>& tensors, int64_t dim);
  static tensor_t cat(std::initializer_list<tensor_t> tensors, int64_t dim);
  static tensor_t stack(const std::vector<tensor_t>& tensors, int64_t dim);

  // Minimum/Maximum
  static tensor_t min(const tensor_t& a, const tensor_t& b);
  static tensor_t max(const tensor_t& a, const tensor_t& b);

  // Data access
  float item_float() const;
  double item_double() const;
  const float* data_ptr_float() const;
  const double* data_ptr_double() const;
  float* mutable_data_ptr_float();
  double* mutable_data_ptr_double();
  template <typename T>
  const T* data_ptr() const;

  // Debugging
  std::string to_string() const;

  // Indexing with [] operator (returns tensor_t for chaining)
  tensor_t operator[](int64_t index) const;

  // Assignment operators for scalar values (for dense[i][j] = value pattern)
  tensor_t& operator=(float scalar) &;
  tensor_t&& operator=(float scalar) &&;
  tensor_t& operator=(double scalar) &;
  tensor_t&& operator=(double scalar) &&;

 private:
  explicit tensor_t(torch::Tensor tensor);

  torch::Tensor tensor_;
};

/**
 * @brief adam_optimizer optimizer options
 */
struct adam_options {
  float lr = 0.001f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
};

/**
 * @brief adam_optimizer optimizer for gradient-based optimization
 */
class adam_optimizer {
 public:
  adam_optimizer(const std::vector<tensor_t>& parameters, const adam_options& options);
  adam_optimizer(std::initializer_list<tensor_t> parameters, const adam_options& options);
  ~adam_optimizer();

  void zero_grad();
  void step();
  void set_learning_rate(float lr);
  float get_learning_rate() const;

 private:
  std::shared_ptr<torch::optim::Adam> optimizer_;
  adam_options options_;
};

// Global operators for scalar operations
tensor_t operator-(float scalar, const tensor_t& tensor);
tensor_t operator-(double scalar, const tensor_t& tensor);
tensor_t operator*(float scalar, const tensor_t& tensor);
tensor_t operator*(double scalar, const tensor_t& tensor);
tensor_t operator+(float scalar, const tensor_t& tensor);
tensor_t operator+(double scalar, const tensor_t& tensor);

}  // namespace optimization
}  // namespace marionette
