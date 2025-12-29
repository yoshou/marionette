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
class Adam;

/**
 * @brief Tensor class wrapping torch::Tensor
 *
 * This class provides a clean interface for tensor operations,
 * wrapping PyTorch's torch::Tensor directly.
 */
class Tensor {
 public:
  Tensor();
  ~Tensor();

  // Friend declarations for global operators and internal classes
  friend class Adam;
  friend Tensor operator-(float scalar, const Tensor& tensor);
  friend Tensor operator-(double scalar, const Tensor& tensor);
  friend Tensor operator*(float scalar, const Tensor& tensor);
  friend Tensor operator*(double scalar, const Tensor& tensor);
  friend Tensor operator+(float scalar, const Tensor& tensor);
  friend Tensor operator+(double scalar, const Tensor& tensor);

  // Copy and move operations
  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) noexcept = default;
  Tensor& operator=(const Tensor& other) & = default;
  Tensor& operator=(Tensor&& other) & noexcept = default;
  Tensor&& operator=(const Tensor& other) &&;
  Tensor&& operator=(Tensor&& other) && noexcept;

  // Factory methods
  static Tensor zeros(const std::vector<int64_t>& shape);
  static Tensor zeros(const std::vector<int64_t>& shape, const torch::TensorOptions& options);
  static Tensor ones(const std::vector<int64_t>& shape);
  static Tensor ones(const std::vector<int64_t>& shape, const torch::TensorOptions& options);
  static Tensor eye(int64_t n);
  static Tensor eye(int64_t n, const torch::TensorOptions& options);
  static Tensor zeros_like(const Tensor& other);
  static Tensor from_blob(const float* data, const std::vector<int64_t>& shape);
  static Tensor from_blob(const double* data, const std::vector<int64_t>& shape);

  // Shape and size operations
  std::vector<int64_t> sizes() const;
  int64_t size(int64_t dim) const;
  int64_t dim() const;
  int64_t numel() const;

  // Data type operations
  Tensor to_float32() const;
  Tensor to_float64() const;

  // Options (for tensor creation)
  torch::TensorOptions options() const;

  // Autograd operations
  Tensor requires_grad(bool requires_grad = true) const;
  Tensor requires_grad(bool requires_grad = true);
  bool requires_grad() const;
  Tensor detach() const;
  void backward() const;
  Tensor grad() const;
  void zero_grad();

  // Validity check
  bool defined() const;

  // Clone and reshape operations
  Tensor clone() const;
  Tensor view(const std::vector<int64_t>& shape) const;
  Tensor reshape(const std::vector<int64_t>& shape) const;
  Tensor unsqueeze(int64_t dim) const;
  Tensor squeeze(int64_t dim) const;
  Tensor expand(const std::vector<int64_t>& shape) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor narrow(int64_t dim, int64_t start, int64_t length) const;

  // Indexing operations
  Tensor select(int64_t dim, int64_t index) const;

  // Unary operators
  Tensor operator-() const;

  // Element-wise arithmetic operations
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator/(const Tensor& other) const;
  Tensor operator+(float scalar) const;
  Tensor operator-(float scalar) const;
  Tensor operator*(float scalar) const;
  Tensor operator/(float scalar) const;

  // In-place operations
  Tensor& operator+=(const Tensor& other);
  Tensor& operator-=(const Tensor& other);
  Tensor& operator*=(const Tensor& other);
  Tensor& operator/=(const Tensor& other);
  Tensor& operator-=(float scalar);
  Tensor& operator+=(float scalar);

  // Reduction operations
  Tensor sum() const;
  Tensor sum(int64_t dim, bool keepdim = false) const;
  Tensor mean() const;
  Tensor mean(int64_t dim, bool keepdim = false) const;
  Tensor min(int64_t dim) const;
  Tensor max(int64_t dim) const;

  // Mathematical operations
  Tensor sqrt() const;
  Tensor cos() const;
  Tensor sin() const;
  Tensor norm(int64_t p, int64_t dim) const;

  // Matrix operations
  static Tensor matmul(const Tensor& a, const Tensor& b);
  static Tensor bmm(const Tensor& a, const Tensor& b);
  static Tensor det(const Tensor& a);
  static Tensor inverse(const Tensor& a);
  Tensor t() const;  // Transpose

  // Concatenation and stacking
  static Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);
  static Tensor cat(std::initializer_list<Tensor> tensors, int64_t dim);
  static Tensor stack(const std::vector<Tensor>& tensors, int64_t dim);

  // Minimum/Maximum
  static Tensor min(const Tensor& a, const Tensor& b);
  static Tensor max(const Tensor& a, const Tensor& b);

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

  // Indexing with [] operator (returns Tensor for chaining)
  Tensor operator[](int64_t index) const;

  // Assignment operators for scalar values (for dense[i][j] = value pattern)
  Tensor& operator=(float scalar) &;
  Tensor&& operator=(float scalar) &&;
  Tensor& operator=(double scalar) &;
  Tensor&& operator=(double scalar) &&;

 private:
  explicit Tensor(torch::Tensor tensor);

  torch::Tensor tensor_;
};

/**
 * @brief Adam optimizer options
 */
struct AdamOptions {
  float lr = 0.001f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.0f;
};

/**
 * @brief Adam optimizer wrapping torch::optim::Adam
 */
class Adam {
 public:
  Adam(const std::vector<Tensor>& parameters, const AdamOptions& options);
  Adam(std::initializer_list<Tensor> parameters, const AdamOptions& options);
  ~Adam();

  void zero_grad();
  void step();
  void set_learning_rate(float lr);
  float get_learning_rate() const;

  // Access to parameter groups for advanced usage
  std::vector<torch::optim::OptimizerParamGroup>& param_groups();

 private:
  std::shared_ptr<torch::optim::Adam> optimizer_;
  AdamOptions options_;
};

// Global operators for scalar operations
Tensor operator-(float scalar, const Tensor& tensor);
Tensor operator-(double scalar, const Tensor& tensor);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator*(double scalar, const Tensor& tensor);
Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator+(double scalar, const Tensor& tensor);

}  // namespace optimization
}  // namespace marionette
