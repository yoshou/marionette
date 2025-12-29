#include "automatic_differentiation_backward.hpp"

#include <sstream>
#include <stdexcept>

namespace marionette {
namespace optimization {

// ============================================================================
// TensorOptions implementation
// ============================================================================

TensorOptions::TensorOptions() : options_(torch::TensorOptions()) {}

TensorOptions::TensorOptions(torch::TensorOptions options) : options_(std::move(options)) {}

TensorOptions::~TensorOptions() = default;

TensorOptions TensorOptions::float32() {
  return TensorOptions(torch::TensorOptions().dtype(torch::kFloat32));
}

TensorOptions TensorOptions::float64() {
  return TensorOptions(torch::TensorOptions().dtype(torch::kFloat64));
}

TensorOptions TensorOptions::device_cpu() const {
  return TensorOptions(options_.device(torch::kCPU));
}

TensorOptions TensorOptions::device_cuda(int device_index) const {
  return TensorOptions(options_.device(torch::Device(torch::kCUDA, device_index)));
}

const torch::TensorOptions& TensorOptions::torch_options() const { return options_; }

// ============================================================================
// Tensor implementation
// ============================================================================

Tensor::Tensor() : tensor_() {}

Tensor::Tensor(torch::Tensor tensor) : tensor_(std::move(tensor)) {}

Tensor::~Tensor() = default;

// Factory methods
Tensor Tensor::zeros(const std::vector<int64_t>& shape) {
  return Tensor(torch::zeros(shape, torch::kFloat32));
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, const TensorOptions& options) {
  return Tensor(torch::zeros(shape, options.torch_options()));
}

Tensor Tensor::ones(const std::vector<int64_t>& shape) {
  return Tensor(torch::ones(shape, torch::kFloat32));
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, const TensorOptions& options) {
  return Tensor(torch::ones(shape, options.torch_options()));
}

Tensor Tensor::eye(int64_t n) { return Tensor(torch::eye(n, torch::kFloat32)); }

Tensor Tensor::eye(int64_t n, const TensorOptions& options) {
  return Tensor(torch::eye(n, options.torch_options()));
}

Tensor Tensor::zeros_like(const Tensor& other) { return Tensor(torch::zeros_like(other.tensor_)); }

Tensor Tensor::from_blob(const float* data, const std::vector<int64_t>& shape) {
  return Tensor(torch::from_blob(const_cast<float*>(data), shape, torch::kFloat32).clone());
}

Tensor Tensor::from_blob(const double* data, const std::vector<int64_t>& shape) {
  return Tensor(torch::from_blob(const_cast<double*>(data), shape, torch::kFloat64).clone());
}

// Shape and size operations
std::vector<int64_t> Tensor::sizes() const {
  auto sizes = tensor_.sizes();
  return std::vector<int64_t>(sizes.begin(), sizes.end());
}

int64_t Tensor::size(int64_t dim) const { return tensor_.size(dim); }

int64_t Tensor::dim() const { return tensor_.dim(); }

int64_t Tensor::numel() const { return tensor_.numel(); }

// Data type operations
Tensor Tensor::to_float32() const { return Tensor(tensor_.to(torch::kFloat32)); }

Tensor Tensor::to_float64() const { return Tensor(tensor_.to(torch::kFloat64)); }

// Autograd operations
Tensor Tensor::requires_grad(bool requires_grad) const {
  auto new_tensor = tensor_.clone();
  new_tensor.requires_grad_(requires_grad);
  return Tensor(new_tensor);
}

Tensor Tensor::requires_grad(bool requires_grad) {
  tensor_.requires_grad_(requires_grad);
  return *this;
}

bool Tensor::requires_grad() const { return tensor_.requires_grad(); }

Tensor Tensor::detach() const { return Tensor(tensor_.detach()); }

void Tensor::backward() const { tensor_.backward(); }

Tensor Tensor::grad() const {
  if (!tensor_.grad().defined()) {
    return Tensor();
  }
  return Tensor(tensor_.grad());
}

void Tensor::zero_grad() {
  if (tensor_.grad().defined()) {
    tensor_.grad().zero_();
  }
}

// Validity check
bool Tensor::defined() const { return tensor_.defined(); }

// Options
TensorOptions Tensor::options() const { return TensorOptions(tensor_.options()); }

// Clone and reshape operations
Tensor Tensor::clone() const { return Tensor(tensor_.clone()); }

Tensor Tensor::view(const std::vector<int64_t>& shape) const { return Tensor(tensor_.view(shape)); }

Tensor Tensor::reshape(const std::vector<int64_t>& shape) const {
  return Tensor(tensor_.reshape(shape));
}

Tensor Tensor::unsqueeze(int64_t dim) const { return Tensor(tensor_.unsqueeze(dim)); }

Tensor Tensor::squeeze(int64_t dim) const { return Tensor(tensor_.squeeze(dim)); }

Tensor Tensor::expand(const std::vector<int64_t>& shape) const {
  return Tensor(tensor_.expand(shape));
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  return Tensor(tensor_.transpose(dim0, dim1));
}

// Indexing operations
Tensor Tensor::select(int64_t dim, int64_t index) const {
  return Tensor(tensor_.select(dim, index));
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
  return Tensor(tensor_.narrow(dim, start, length));
}

// Unary operators
Tensor Tensor::operator-() const { return Tensor(-tensor_); }

// Element-wise arithmetic operations
Tensor Tensor::operator+(const Tensor& other) const { return Tensor(tensor_ + other.tensor_); }

Tensor Tensor::operator-(const Tensor& other) const { return Tensor(tensor_ - other.tensor_); }

Tensor Tensor::operator*(const Tensor& other) const { return Tensor(tensor_ * other.tensor_); }

Tensor Tensor::operator/(const Tensor& other) const { return Tensor(tensor_ / other.tensor_); }

Tensor Tensor::operator+(float scalar) const { return Tensor(tensor_ + scalar); }

Tensor Tensor::operator-(float scalar) const { return Tensor(tensor_ - scalar); }

Tensor Tensor::operator*(float scalar) const { return Tensor(tensor_ * scalar); }

Tensor Tensor::operator/(float scalar) const { return Tensor(tensor_ / scalar); }

// In-place operations
Tensor& Tensor::operator+=(const Tensor& other) {
  tensor_ += other.tensor_;
  return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  tensor_ -= other.tensor_;
  return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
  tensor_ *= other.tensor_;
  return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
  tensor_ /= other.tensor_;
  return *this;
}

Tensor& Tensor::operator-=(float scalar) {
  tensor_ -= scalar;
  return *this;
}

Tensor& Tensor::operator+=(float scalar) {
  tensor_ += scalar;
  return *this;
}

// Reduction operations
Tensor Tensor::sum() const { return Tensor(tensor_.sum()); }

Tensor Tensor::sum(int64_t dim, bool keepdim) const { return Tensor(tensor_.sum(dim, keepdim)); }

Tensor Tensor::mean() const { return Tensor(tensor_.mean()); }

Tensor Tensor::mean(int64_t dim, bool keepdim) const { return Tensor(tensor_.mean(dim, keepdim)); }

Tensor Tensor::min(int64_t dim) const { return Tensor(std::get<0>(tensor_.min(dim))); }

Tensor Tensor::max(int64_t dim) const { return Tensor(std::get<0>(tensor_.max(dim))); }

// Mathematical operations
Tensor Tensor::sqrt() const { return Tensor(torch::sqrt(tensor_)); }

Tensor Tensor::cos() const { return Tensor(torch::cos(tensor_)); }

Tensor Tensor::sin() const { return Tensor(torch::sin(tensor_)); }

Tensor Tensor::norm(int64_t p, int64_t dim) const { return Tensor(torch::norm(tensor_, p, dim)); }

// Matrix operations
Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
  return Tensor(torch::matmul(a.tensor_, b.tensor_));
}

Tensor Tensor::bmm(const Tensor& a, const Tensor& b) {
  return Tensor(torch::bmm(a.tensor_, b.tensor_));
}

Tensor Tensor::det(const Tensor& a) { return Tensor(torch::det(a.tensor_)); }

Tensor Tensor::inverse(const Tensor& a) { return Tensor(torch::inverse(a.tensor_)); }

Tensor Tensor::t() const { return Tensor(tensor_.t()); }

// Concatenation and stacking
Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
  std::vector<torch::Tensor> torch_tensors;
  for (const auto& t : tensors) {
    torch_tensors.push_back(t.tensor_);
  }
  return Tensor(torch::cat(torch_tensors, dim));
}

Tensor Tensor::cat(std::initializer_list<Tensor> tensors, int64_t dim) {
  return cat(std::vector<Tensor>(tensors), dim);
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int64_t dim) {
  std::vector<torch::Tensor> torch_tensors;
  for (const auto& t : tensors) {
    torch_tensors.push_back(t.tensor_);
  }
  return Tensor(torch::stack(torch_tensors, dim));
}

// Minimum/Maximum
Tensor Tensor::min(const Tensor& a, const Tensor& b) {
  return Tensor(torch::min(a.tensor_, b.tensor_));
}

Tensor Tensor::max(const Tensor& a, const Tensor& b) {
  return Tensor(torch::max(a.tensor_, b.tensor_));
}

// Data access
float Tensor::item_float() const { return tensor_.item<float>(); }

double Tensor::item_double() const { return tensor_.item<double>(); }

const float* Tensor::data_ptr_float() const { return tensor_.data_ptr<float>(); }

const double* Tensor::data_ptr_double() const { return tensor_.data_ptr<double>(); }

float* Tensor::mutable_data_ptr_float() { return tensor_.data_ptr<float>(); }

double* Tensor::mutable_data_ptr_double() { return tensor_.data_ptr<double>(); }

template <typename T>
const T* Tensor::data_ptr() const {
  return tensor_.data_ptr<T>();
}

// Explicit instantiations
template const float* Tensor::data_ptr<float>() const;
template const double* Tensor::data_ptr<double>() const;

// Debugging
std::string Tensor::to_string() const {
  if (!tensor_.defined()) {
    return "Tensor(undefined)";
  }
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  auto sizes = tensor_.sizes();
  for (size_t i = 0; i < sizes.size(); ++i) {
    oss << sizes[i];
    if (i < sizes.size() - 1) oss << ", ";
  }
  oss << "], dtype=" << tensor_.dtype() << ", device=" << tensor_.device() << ")";
  return oss.str();
}

Tensor Tensor::operator[](int64_t index) const { return Tensor(tensor_[index]); }

Tensor&& Tensor::operator=(const Tensor& other) && {
  tensor_.copy_(other.tensor_);
  return std::move(*this);
}

Tensor&& Tensor::operator=(Tensor&& other) && noexcept {
  tensor_.copy_(other.tensor_);
  return std::move(*this);
}

Tensor& Tensor::operator=(float scalar) & {
  tensor_.fill_(scalar);
  return *this;
}

Tensor&& Tensor::operator=(float scalar) && {
  tensor_.fill_(scalar);
  return std::move(*this);
}

Tensor& Tensor::operator=(double scalar) & {
  tensor_.fill_(scalar);
  return *this;
}

Tensor&& Tensor::operator=(double scalar) && {
  tensor_.fill_(scalar);
  return std::move(*this);
}

// ============================================================================
// Adam implementation
// ============================================================================

Adam::Adam(const std::vector<Tensor>& parameters, const AdamOptions& options) : options_(options) {
  std::vector<torch::Tensor> torch_params;
  for (const auto& param : parameters) {
    torch_params.push_back(param.tensor_);
  }

  torch::optim::AdamOptions torch_options(options.lr);
  torch_options.betas(std::make_tuple(options.beta1, options.beta2));
  torch_options.eps(options.eps);
  torch_options.weight_decay(options.weight_decay);

  optimizer_ = std::make_shared<torch::optim::Adam>(torch_params, torch_options);
}

Adam::Adam(std::initializer_list<Tensor> parameters, const AdamOptions& options)
    : Adam(std::vector<Tensor>(parameters), options) {}

Adam::~Adam() = default;

void Adam::zero_grad() { optimizer_->zero_grad(); }

void Adam::step() { optimizer_->step(); }

void Adam::set_learning_rate(float lr) {
  options_.lr = lr;
  for (auto& param_group : optimizer_->param_groups()) {
    static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(lr);
  }
}

float Adam::get_learning_rate() const { return options_.lr; }

// Global operators
Tensor operator-(float scalar, const Tensor& tensor) { return Tensor(scalar - tensor.tensor_); }

Tensor operator-(double scalar, const Tensor& tensor) { return Tensor(scalar - tensor.tensor_); }

Tensor operator*(float scalar, const Tensor& tensor) { return Tensor(scalar * tensor.tensor_); }

Tensor operator*(double scalar, const Tensor& tensor) { return Tensor(scalar * tensor.tensor_); }

Tensor operator+(float scalar, const Tensor& tensor) { return Tensor(scalar + tensor.tensor_); }

Tensor operator+(double scalar, const Tensor& tensor) { return Tensor(scalar + tensor.tensor_); }

}  // namespace optimization
}  // namespace marionette
