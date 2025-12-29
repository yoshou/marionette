#include "automatic_differentiation_backward.hpp"

#include <sstream>
#include <stdexcept>

namespace marionette {
namespace optimization {

// ============================================================================
// tensor_options implementation
// ============================================================================

tensor_options::tensor_options() : options_(torch::TensorOptions()) {}

tensor_options::tensor_options(torch::TensorOptions options) : options_(std::move(options)) {}

tensor_options::~tensor_options() = default;

tensor_options tensor_options::float32() {
  return tensor_options(torch::TensorOptions().dtype(torch::kFloat32));
}

tensor_options tensor_options::float64() {
  return tensor_options(torch::TensorOptions().dtype(torch::kFloat64));
}

tensor_options tensor_options::device_cpu() const {
  return tensor_options(options_.device(torch::kCPU));
}

tensor_options tensor_options::device_cuda(int device_index) const {
  return tensor_options(options_.device(torch::Device(torch::kCUDA, device_index)));
}

const torch::TensorOptions& tensor_options::torch_options() const { return options_; }

// ============================================================================
// tensor_t implementation
// ============================================================================

tensor_t::tensor_t() : tensor_() {}

tensor_t::tensor_t(torch::Tensor tensor) : tensor_(std::move(tensor)) {}

tensor_t::~tensor_t() = default;

// Factory methods
tensor_t tensor_t::zeros(const std::vector<int64_t>& shape) {
  return tensor_t(torch::zeros(shape, torch::kFloat32));
}

tensor_t tensor_t::zeros(const std::vector<int64_t>& shape, const tensor_options& options) {
  return tensor_t(torch::zeros(shape, options.torch_options()));
}

tensor_t tensor_t::ones(const std::vector<int64_t>& shape) {
  return tensor_t(torch::ones(shape, torch::kFloat32));
}

tensor_t tensor_t::ones(const std::vector<int64_t>& shape, const tensor_options& options) {
  return tensor_t(torch::ones(shape, options.torch_options()));
}

tensor_t tensor_t::eye(int64_t n) { return tensor_t(torch::eye(n, torch::kFloat32)); }

tensor_t tensor_t::eye(int64_t n, const tensor_options& options) {
  return tensor_t(torch::eye(n, options.torch_options()));
}

tensor_t tensor_t::zeros_like(const tensor_t& other) {
  return tensor_t(torch::zeros_like(other.tensor_));
}

tensor_t tensor_t::from_blob(const float* data, const std::vector<int64_t>& shape) {
  return tensor_t(torch::from_blob(const_cast<float*>(data), shape, torch::kFloat32).clone());
}

tensor_t tensor_t::from_blob(const double* data, const std::vector<int64_t>& shape) {
  return tensor_t(torch::from_blob(const_cast<double*>(data), shape, torch::kFloat64).clone());
}

// Shape and size operations
std::vector<int64_t> tensor_t::sizes() const {
  auto sizes = tensor_.sizes();
  return std::vector<int64_t>(sizes.begin(), sizes.end());
}

int64_t tensor_t::size(int64_t dim) const { return tensor_.size(dim); }

int64_t tensor_t::dim() const { return tensor_.dim(); }

int64_t tensor_t::numel() const { return tensor_.numel(); }

// Data type operations
tensor_t tensor_t::to_float32() const { return tensor_t(tensor_.to(torch::kFloat32)); }

tensor_t tensor_t::to_float64() const { return tensor_t(tensor_.to(torch::kFloat64)); }

// Autograd operations
tensor_t tensor_t::requires_grad(bool requires_grad) const {
  auto new_tensor = tensor_.clone();
  new_tensor.requires_grad_(requires_grad);
  return tensor_t(new_tensor);
}

tensor_t tensor_t::requires_grad(bool requires_grad) {
  tensor_.requires_grad_(requires_grad);
  return *this;
}

bool tensor_t::requires_grad() const { return tensor_.requires_grad(); }

tensor_t tensor_t::detach() const { return tensor_t(tensor_.detach()); }

void tensor_t::backward() const { tensor_.backward(); }

tensor_t tensor_t::grad() const {
  if (!tensor_.grad().defined()) {
    return tensor_t();
  }
  return tensor_t(tensor_.grad());
}

void tensor_t::zero_grad() {
  if (tensor_.grad().defined()) {
    tensor_.grad().zero_();
  }
}

// Validity check
bool tensor_t::defined() const { return tensor_.defined(); }

// Options
tensor_options tensor_t::options() const { return tensor_options(tensor_.options()); }

// Clone and reshape operations
tensor_t tensor_t::clone() const { return tensor_t(tensor_.clone()); }

tensor_t tensor_t::view(const std::vector<int64_t>& shape) const {
  return tensor_t(tensor_.view(shape));
}

tensor_t tensor_t::reshape(const std::vector<int64_t>& shape) const {
  return tensor_t(tensor_.reshape(shape));
}

tensor_t tensor_t::unsqueeze(int64_t dim) const { return tensor_t(tensor_.unsqueeze(dim)); }

tensor_t tensor_t::squeeze(int64_t dim) const { return tensor_t(tensor_.squeeze(dim)); }

tensor_t tensor_t::expand(const std::vector<int64_t>& shape) const {
  return tensor_t(tensor_.expand(shape));
}

tensor_t tensor_t::transpose(int64_t dim0, int64_t dim1) const {
  return tensor_t(tensor_.transpose(dim0, dim1));
}

// Indexing operations
tensor_t tensor_t::select(int64_t dim, int64_t index) const {
  return tensor_t(tensor_.select(dim, index));
}

tensor_t tensor_t::narrow(int64_t dim, int64_t start, int64_t length) const {
  return tensor_t(tensor_.narrow(dim, start, length));
}

// Unary operators
tensor_t tensor_t::operator-() const { return tensor_t(-tensor_); }

// Element-wise arithmetic operations
tensor_t tensor_t::operator+(const tensor_t& other) const {
  return tensor_t(tensor_ + other.tensor_);
}

tensor_t tensor_t::operator-(const tensor_t& other) const {
  return tensor_t(tensor_ - other.tensor_);
}

tensor_t tensor_t::operator*(const tensor_t& other) const {
  return tensor_t(tensor_ * other.tensor_);
}

tensor_t tensor_t::operator/(const tensor_t& other) const {
  return tensor_t(tensor_ / other.tensor_);
}

tensor_t tensor_t::operator+(float scalar) const { return tensor_t(tensor_ + scalar); }

tensor_t tensor_t::operator-(float scalar) const { return tensor_t(tensor_ - scalar); }

tensor_t tensor_t::operator*(float scalar) const { return tensor_t(tensor_ * scalar); }

tensor_t tensor_t::operator/(float scalar) const { return tensor_t(tensor_ / scalar); }

// In-place operations
tensor_t& tensor_t::operator+=(const tensor_t& other) {
  tensor_ += other.tensor_;
  return *this;
}

tensor_t& tensor_t::operator-=(const tensor_t& other) {
  tensor_ -= other.tensor_;
  return *this;
}

tensor_t& tensor_t::operator*=(const tensor_t& other) {
  tensor_ *= other.tensor_;
  return *this;
}

tensor_t& tensor_t::operator/=(const tensor_t& other) {
  tensor_ /= other.tensor_;
  return *this;
}

tensor_t& tensor_t::operator-=(float scalar) {
  tensor_ -= scalar;
  return *this;
}

tensor_t& tensor_t::operator+=(float scalar) {
  tensor_ += scalar;
  return *this;
}

// Reduction operations
tensor_t tensor_t::sum() const { return tensor_t(tensor_.sum()); }

tensor_t tensor_t::sum(int64_t dim, bool keepdim) const {
  return tensor_t(tensor_.sum(dim, keepdim));
}

tensor_t tensor_t::mean() const { return tensor_t(tensor_.mean()); }

tensor_t tensor_t::mean(int64_t dim, bool keepdim) const {
  return tensor_t(tensor_.mean(dim, keepdim));
}

tensor_t tensor_t::min(int64_t dim) const { return tensor_t(std::get<0>(tensor_.min(dim))); }

tensor_t tensor_t::max(int64_t dim) const { return tensor_t(std::get<0>(tensor_.max(dim))); }

// Mathematical operations
tensor_t tensor_t::sqrt() const { return tensor_t(torch::sqrt(tensor_)); }

tensor_t tensor_t::cos() const { return tensor_t(torch::cos(tensor_)); }

tensor_t tensor_t::sin() const { return tensor_t(torch::sin(tensor_)); }

tensor_t tensor_t::norm(int64_t p, int64_t dim) const {
  return tensor_t(torch::norm(tensor_, p, dim));
}

// Matrix operations
tensor_t tensor_t::matmul(const tensor_t& a, const tensor_t& b) {
  return tensor_t(torch::matmul(a.tensor_, b.tensor_));
}

tensor_t tensor_t::bmm(const tensor_t& a, const tensor_t& b) {
  return tensor_t(torch::bmm(a.tensor_, b.tensor_));
}

tensor_t tensor_t::det(const tensor_t& a) { return tensor_t(torch::det(a.tensor_)); }

tensor_t tensor_t::inverse(const tensor_t& a) { return tensor_t(torch::inverse(a.tensor_)); }

tensor_t tensor_t::t() const { return tensor_t(tensor_.t()); }

// Concatenation and stacking
tensor_t tensor_t::cat(const std::vector<tensor_t>& tensors, int64_t dim) {
  std::vector<torch::Tensor> torch_tensors;
  for (const auto& t : tensors) {
    torch_tensors.push_back(t.tensor_);
  }
  return tensor_t(torch::cat(torch_tensors, dim));
}

tensor_t tensor_t::cat(std::initializer_list<tensor_t> tensors, int64_t dim) {
  return cat(std::vector<tensor_t>(tensors), dim);
}

tensor_t tensor_t::stack(const std::vector<tensor_t>& tensors, int64_t dim) {
  std::vector<torch::Tensor> torch_tensors;
  for (const auto& t : tensors) {
    torch_tensors.push_back(t.tensor_);
  }
  return tensor_t(torch::stack(torch_tensors, dim));
}

// Minimum/Maximum
tensor_t tensor_t::min(const tensor_t& a, const tensor_t& b) {
  return tensor_t(torch::min(a.tensor_, b.tensor_));
}

tensor_t tensor_t::max(const tensor_t& a, const tensor_t& b) {
  return tensor_t(torch::max(a.tensor_, b.tensor_));
}

// Data access
float tensor_t::item_float() const { return tensor_.item<float>(); }

double tensor_t::item_double() const { return tensor_.item<double>(); }

const float* tensor_t::data_ptr_float() const { return tensor_.data_ptr<float>(); }

const double* tensor_t::data_ptr_double() const { return tensor_.data_ptr<double>(); }

float* tensor_t::mutable_data_ptr_float() { return tensor_.data_ptr<float>(); }

double* tensor_t::mutable_data_ptr_double() { return tensor_.data_ptr<double>(); }

template <typename T>
const T* tensor_t::data_ptr() const {
  return tensor_.data_ptr<T>();
}

// Explicit instantiations
template const float* tensor_t::data_ptr<float>() const;
template const double* tensor_t::data_ptr<double>() const;

// Debugging
std::string tensor_t::to_string() const {
  if (!tensor_.defined()) {
    return "tensor_t(undefined)";
  }
  std::ostringstream oss;
  oss << "tensor_t(shape=[";
  auto sizes = tensor_.sizes();
  for (size_t i = 0; i < sizes.size(); ++i) {
    oss << sizes[i];
    if (i < sizes.size() - 1) oss << ", ";
  }
  oss << "], dtype=" << tensor_.dtype() << ", device=" << tensor_.device() << ")";
  return oss.str();
}

tensor_t tensor_t::operator[](int64_t index) const { return tensor_t(tensor_[index]); }

tensor_t&& tensor_t::operator=(const tensor_t& other) && {
  tensor_.copy_(other.tensor_);
  return std::move(*this);
}

tensor_t&& tensor_t::operator=(tensor_t&& other) && noexcept {
  tensor_.copy_(other.tensor_);
  return std::move(*this);
}

tensor_t& tensor_t::operator=(float scalar) & {
  tensor_.fill_(scalar);
  return *this;
}

tensor_t&& tensor_t::operator=(float scalar) && {
  tensor_.fill_(scalar);
  return std::move(*this);
}

tensor_t& tensor_t::operator=(double scalar) & {
  tensor_.fill_(scalar);
  return *this;
}

tensor_t&& tensor_t::operator=(double scalar) && {
  tensor_.fill_(scalar);
  return std::move(*this);
}

// ============================================================================
// adam_optimizer implementation
// ============================================================================

adam_optimizer::adam_optimizer(const std::vector<tensor_t>& parameters, const adam_options& options)
    : options_(options) {
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

adam_optimizer::adam_optimizer(std::initializer_list<tensor_t> parameters,
                               const adam_options& options)
    : adam_optimizer(std::vector<tensor_t>(parameters), options) {}

adam_optimizer::~adam_optimizer() = default;

void adam_optimizer::zero_grad() { optimizer_->zero_grad(); }

void adam_optimizer::step() { optimizer_->step(); }

void adam_optimizer::set_learning_rate(float lr) {
  options_.lr = lr;
  for (auto& param_group : optimizer_->param_groups()) {
    static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(lr);
  }
}

float adam_optimizer::get_learning_rate() const { return options_.lr; }

// Global operators
tensor_t operator-(float scalar, const tensor_t& tensor) {
  return tensor_t(scalar - tensor.tensor_);
}

tensor_t operator-(double scalar, const tensor_t& tensor) {
  return tensor_t(scalar - tensor.tensor_);
}

tensor_t operator*(float scalar, const tensor_t& tensor) {
  return tensor_t(scalar * tensor.tensor_);
}

tensor_t operator*(double scalar, const tensor_t& tensor) {
  return tensor_t(scalar * tensor.tensor_);
}

tensor_t operator+(float scalar, const tensor_t& tensor) {
  return tensor_t(scalar + tensor.tensor_);
}

tensor_t operator+(double scalar, const tensor_t& tensor) {
  return tensor_t(scalar + tensor.tensor_);
}

}  // namespace optimization
}  // namespace marionette
