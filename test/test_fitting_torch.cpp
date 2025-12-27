#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

// Prevent glog from initializing by defining these symbols
extern "C" {
void __attribute__((weak)) InitGoogleLogging(const char*) {}
}

// Disable glog to avoid conflicts
#define GLOG_NO_ABBREVIATED_SEVERITIES
#undef LOG
#define LOG(severity) std::cout

template <typename T>
static std::string get_typename() {
  if constexpr (std::is_same_v<T, float>) {
    return "float32";
  } else if constexpr (std::is_same_v<T, double>) {
    return "float64";
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return "uint32";
  }
  throw std::runtime_error("Invalid type");
}

template <typename T>
static std::tuple<std::vector<T>, std::vector<uint32_t>> load_tensor(const std::string& file_name) {
  std::ifstream ifs;
  ifs.open(file_name, std::ios::in);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_name);
  }
  nlohmann::json j = nlohmann::json::parse(ifs);
  if (j["type"].get<std::string>() != get_typename<T>()) {
    throw std::runtime_error("Invalid type");
  }
  const auto data = j["data"].get<std::vector<T>>();
  const auto shape = j["shape"].get<std::vector<uint32_t>>();
  return std::forward_as_tuple(data, shape);
}

torch::Tensor load_tensor_as_torch(const std::string& file_name) {
  auto [data, shape] = load_tensor<float>(file_name);

  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.push_back(static_cast<int64_t>(s));
  }

  auto tensor = torch::from_blob(data.data(), torch_shape, torch::kFloat32).clone();
  return tensor;
}

torch::Tensor load_dense_matrix_as_torch(const std::string& file_name) {
  std::ifstream ifs;
  ifs.open(file_name, std::ios::in);
  nlohmann::json j = nlohmann::json::parse(ifs);

  const auto type_str = j["type"].get<std::string>();
  const auto data_vec = j["data"];
  const auto shape = j["shape"].get<std::vector<uint32_t>>();

  if (shape.size() != 2) {
    throw std::runtime_error("Invalid shape");
  }

  torch::Tensor tensor;
  if (type_str == "float64") {
    auto data = data_vec.get<std::vector<double>>();
    tensor = torch::from_blob(const_cast<double*>(data.data()),
                              {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])},
                              torch::kFloat64)
                 .clone();
  } else if (type_str == "float32") {
    auto data = data_vec.get<std::vector<float>>();
    tensor = torch::from_blob(const_cast<float*>(data.data()),
                              {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])},
                              torch::kFloat32)
                 .clone();
  } else {
    throw std::runtime_error("Unsupported type: " + type_str);
  }

  return tensor.to(torch::kFloat32);
}

torch::Tensor load_sparse_matrix_as_torch(const std::string& file_name) {
  std::ifstream ifs;
  ifs.open(file_name, std::ios::in);
  nlohmann::json j = nlohmann::json::parse(ifs);

  const auto type_str = j["type"].get<std::string>();
  const auto row = j["row"].get<std::vector<uint32_t>>();
  const auto col = j["col"].get<std::vector<uint32_t>>();
  const auto shape = j["shape"].get<std::vector<uint32_t>>();

  if (shape.size() != 2) {
    throw std::runtime_error("Invalid shape");
  }

  torch::Tensor dense = torch::zeros(
      {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])}, torch::kFloat32);

  if (type_str == "float64") {
    const auto data = j["data"].get<std::vector<double>>();
    for (size_t i = 0; i < data.size(); i++) {
      dense[row[i]][col[i]] = static_cast<float>(data[i]);
    }
  } else if (type_str == "float32") {
    const auto data = j["data"].get<std::vector<float>>();
    for (size_t i = 0; i < data.size(); i++) {
      dense[row[i]][col[i]] = data[i];
    }
  } else {
    throw std::runtime_error("Unsupported type: " + type_str);
  }

  return dense;
}

// Prior loss using GMM
class PriorLoss {
 public:
  torch::Tensor gmm_means;       // [8, 69]
  torch::Tensor gmm_precisions;  // [8, 69, 69] - inverse of covariance
  torch::Tensor nll_weights;     // [8]

  PriorLoss() {
    using path = std::filesystem::path;
    path param_dir("../data/opt/");

    // Load GMM parameters
    auto [means_data, means_shape] =
        load_tensor<double>((param_dir / "gmm_means.json").generic_string());
    gmm_means = torch::from_blob(
                    means_data.data(),
                    {static_cast<int64_t>(means_shape[0]), static_cast<int64_t>(means_shape[1])},
                    torch::kFloat64)
                    .clone()
                    .to(torch::kFloat32);

    auto [covs_data, covs_shape] =
        load_tensor<double>((param_dir / "gmm_covars.json").generic_string());
    auto covs =
        torch::from_blob(covs_data.data(),
                         {static_cast<int64_t>(covs_shape[0]), static_cast<int64_t>(covs_shape[1]),
                          static_cast<int64_t>(covs_shape[2])},
                         torch::kFloat64)
            .clone();

    auto [weights_data, weights_shape] =
        load_tensor<double>((param_dir / "gmm_weights.json").generic_string());
    auto gmm_weights = torch::from_blob(weights_data.data(),
                                        {static_cast<int64_t>(weights_shape[0])}, torch::kFloat64)
                           .clone();

    // Compute precisions (inverse of covariance) and determinants
    int num_gaussians = covs.size(0);
    std::vector<torch::Tensor> precisions_list;
    std::vector<double> sqrdets;

    for (int m = 0; m < num_gaussians; m++) {
      auto cov = covs[m].to(torch::kFloat64);
      auto det = torch::det(cov).item<double>();
      auto sqrdet = std::sqrt(det);
      sqrdets.push_back(sqrdet);

      auto precision = torch::inverse(cov);
      precisions_list.push_back(precision);
    }

    gmm_precisions = torch::stack(precisions_list, 0).to(torch::kFloat32);

    // Compute nll_weights
    const double PI = 3.141592653589793;
    const double c = std::pow(2.0 * PI, 69.0 / 2.0);
    double min_sqrdet = *std::min_element(sqrdets.begin(), sqrdets.end());

    std::vector<float> nll_weights_vec;
    for (int m = 0; m < num_gaussians; m++) {
      double w = gmm_weights[m].item<double>();
      double nll = -std::log(w / (c * (sqrdets[m] / min_sqrdet)));
      nll_weights_vec.push_back(static_cast<float>(nll));
    }

    nll_weights =
        torch::from_blob(nll_weights_vec.data(), {num_gaussians}, torch::kFloat32).clone();
  }

  torch::Tensor compute(const torch::Tensor& poses) {
    // poses: [num_frames, 69]
    int num_frames = poses.size(0);

    torch::Tensor total_loss = torch::zeros({}, poses.options());

    for (int b = 0; b < num_frames; b++) {
      auto pose = poses[b];  // [69]

      // Compute differences from all gaussian means: d = pose - means
      // pose: [69], gmm_means: [8, 69] -> d: [8, 69]
      auto d = pose.unsqueeze(0) - gmm_means;  // [8, 69]

      // Compute prec_d = d @ precision^T for each gaussian
      // d: [8, 69], gmm_precisions: [8, 69, 69]
      // For each gaussian m: prec_d[m] = sum_j(d[m,j] * precision[m,:,j])
      auto d_expanded = d.unsqueeze(1);                                 // [8, 1, 69]
      auto prec_d = torch::bmm(d_expanded, gmm_precisions).squeeze(1);  // [8, 69]

      // Compute prec_dd = d @ prec_d for each gaussian
      auto prec_dd = (d * prec_d).sum(1);  // [8]

      // Compute log-likelihood for each gaussian
      auto loglikelihood = 0.5f * prec_dd + nll_weights;  // [8]

      // Take minimum log-likelihood
      auto min_likelihood = loglikelihood.min();
      total_loss += min_likelihood;
    }

    // Average over frames
    total_loss /= static_cast<float>(num_frames);

    return total_loss;
  }
};

// Forward declaration of SMPL model
class SMPLModel {
 public:
  torch::Tensor v_template;
  torch::Tensor weights;
  torch::Tensor j_regressor;
  torch::Tensor j_regressor_body25;
  torch::Tensor shapedirs;
  torch::Tensor posedirs;
  std::vector<int32_t> parents;

 private:
  // Convert axis-angle to rotation matrix using Rodrigues formula
  torch::Tensor batch_rodrigues(const torch::Tensor& pose_vec) {
    // pose_vec: [N, 3] or [3] - axis-angle representation
    auto pose_reshaped = pose_vec.dim() == 1 ? pose_vec.unsqueeze(0) : pose_vec;

    auto theta2 = (pose_reshaped * pose_reshaped).sum(1, true);  // [N, 1]
    auto theta = torch::sqrt(theta2 + 1e-8);
    auto w = pose_reshaped / (theta + 1e-8);  // [N, 3]

    auto wx = w.select(1, 0).unsqueeze(1);
    auto wy = w.select(1, 1).unsqueeze(1);
    auto wz = w.select(1, 2).unsqueeze(1);

    auto cos_theta = torch::cos(theta);
    auto sin_theta = torch::sin(theta);
    auto one_minus_cos = 1.0 - cos_theta;

    auto r00 = cos_theta + wx * wx * one_minus_cos;
    auto r01 = wx * wy * one_minus_cos - wz * sin_theta;
    auto r02 = wy * sin_theta + wx * wz * one_minus_cos;
    auto r10 = wz * sin_theta + wx * wy * one_minus_cos;
    auto r11 = cos_theta + wy * wy * one_minus_cos;
    auto r12 = -wx * sin_theta + wy * wz * one_minus_cos;
    auto r20 = -wy * sin_theta + wx * wz * one_minus_cos;
    auto r21 = wx * sin_theta + wy * wz * one_minus_cos;
    auto r22 = cos_theta + wz * wz * one_minus_cos;

    auto row0 = torch::cat({r00, r01, r02}, 1).unsqueeze(1);
    auto row1 = torch::cat({r10, r11, r12}, 1).unsqueeze(1);
    auto row2 = torch::cat({r20, r21, r22}, 1).unsqueeze(1);
    return torch::cat({row0, row1, row2}, 1);  // [N, 3, 3]
  }

  // Apply shape blending: v_template + shapedirs * betas
  torch::Tensor apply_shape_blend(const torch::Tensor& betas) {
    auto v_shaped = v_template.unsqueeze(0);  // [1, 6890, 3]

    if (betas.defined() && betas.numel() > 0) {
      // shapedirs: [6890, 3, 10], betas: [1, 10] -> [1, 6890, 3]
      auto shape_disps = torch::einsum("mkl,bl->bmk", {shapedirs, betas});
      v_shaped = v_shaped + shape_disps;
    }

    return v_shaped;
  }

  // Apply pose-dependent vertex deformations
  torch::Tensor apply_pose_blend(const torch::Tensor& v_shaped, const torch::Tensor& poses) {
    if (!poses.defined() || poses.numel() != 69) {
      return v_shaped;
    }

    auto poses_reshaped = poses.view({23, 3});

    // Convert all joint poses to rotation matrices (batch operation)
    auto rot_mats_3x3 = batch_rodrigues(poses_reshaped);  // [23, 3, 3]
    auto rot_mats = rot_mats_3x3.view({23, 9});           // [23, 9]

    // Pose feature: rot_mats - Identity (modify in-place)
    rot_mats.select(1, 0) -= 1.0f;  // R[0,0] -= 1
    rot_mats.select(1, 4) -= 1.0f;  // R[1,1] -= 1
    rot_mats.select(1, 8) -= 1.0f;  // R[2,2] -= 1

    // Compute pose-dependent vertex displacements
    auto pose_feature_flat = rot_mats.view({-1});                                  // [207]
    auto pose_offset = torch::einsum("vdp,p->vd", {posedirs, pose_feature_flat});  // [6890, 3]

    return v_shaped + pose_offset.unsqueeze(0);
  }

  // Compute Linear Blend Skinning
  torch::Tensor compute_lbs(const torch::Tensor& v_posed, const torch::Tensor& poses) {
    if (!poses.defined() || poses.numel() != 69) {
      return v_posed;
    }

    // 1. Compute 24 joints from v_posed
    auto joints_24 = torch::einsum("bik,ji->bjk", {v_posed, j_regressor});  // [1, 24, 3]
    auto joints_squeezed = joints_24.squeeze(0);                            // [24, 3]

    // 2. Build rotation matrices for each joint (batch operation)
    auto poses_reshaped = poses.view({23, 3});
    auto rot_mats_3x3 = batch_rodrigues(poses_reshaped);  // [23, 3, 3]

    // 3. Build 4x4 transformation matrices for kinematic chain
    std::vector<torch::Tensor> transform_mats_list;
    for (int j = 0; j < 24; j++) {
      auto transform_mat = torch::eye(4, poses.options());

      auto joint = joints_squeezed[j];
      torch::Tensor parent_joint =
          (j == 0) ? torch::zeros({3}, poses.options()) : joints_squeezed[parents[j]];
      auto rel_joint = joint - parent_joint;

      // Set rotation part (only for non-root joints)
      if (j > 0) {
        transform_mat.narrow(0, 0, 3).narrow(1, 0, 3) = rot_mats_3x3[j - 1];
      }

      // Set translation part
      transform_mat.narrow(0, 0, 3).select(1, 3) = rel_joint;

      // Multiply with parent transformation
      if (j > 0 && parents[j] >= 0) {
        transform_mat = torch::matmul(transform_mats_list[parents[j]], transform_mat);
      }

      transform_mats_list.push_back(transform_mat);
    }

    // 4. Convert to relative transformations
    std::vector<torch::Tensor> rel_transform_mats_list;
    for (int j = 0; j < 24; j++) {
      auto& transform_mat = transform_mats_list[j];
      auto joint = joints_squeezed[j];

      auto rot_part = transform_mat.narrow(0, 0, 3).narrow(1, 0, 3);
      auto trans_part = transform_mat.narrow(0, 0, 3).select(1, 3);
      auto new_trans = trans_part - torch::matmul(rot_part, joint);

      auto rel_transform = torch::eye(4, poses.options());
      rel_transform.narrow(0, 0, 3).narrow(1, 0, 3) = rot_part;
      rel_transform.narrow(0, 0, 3).select(1, 3) = new_trans;

      rel_transform_mats_list.push_back(rel_transform);
    }

    auto rel_transform_mats = torch::stack(rel_transform_mats_list, 0);  // [24, 4, 4]

    // 5. Blend transformations using skinning weights
    auto blended_mats =
        torch::einsum("vj,jmn->vmn", {weights, rel_transform_mats});  // [6890, 4, 4]

    // 6. Apply blended transformations to vertices
    auto v_posed_homo = torch::cat({v_posed, torch::ones({1, 6890, 1}, poses.options())}, 2);
    v_posed_homo = v_posed_homo.squeeze(0);  // [6890, 4]

    auto verts_homo = torch::einsum("vmn,vn->vm", {blended_mats, v_posed_homo});  // [6890, 4]
    return verts_homo.narrow(1, 0, 3).unsqueeze(0);                               // [1, 6890, 3]
  }

  // Apply global rotation (rh) and translation (th)
  torch::Tensor apply_global_transform(torch::Tensor joints, const torch::Tensor& rh,
                                       const torch::Tensor& th) {
    // Apply Rodrigues rotation
    if (rh.defined() && rh.numel() > 0) {
      auto rh_flat = rh.view({-1, 3});                    // [1, 3]
      auto R = batch_rodrigues(rh_flat);                  // [1, 3, 3]
      joints = torch::matmul(joints, R.transpose(1, 2));  // [1, 25, 3]
    }

    // Add translation
    if (th.defined() && th.numel() > 0) {
      torch::Tensor th_reshaped;
      if (th.dim() == 2 && th.size(0) > 1) {
        // Multi-frame: average over frames
        th_reshaped = th.mean(0).view({1, 1, 3}).expand({joints.size(0), joints.size(1), 3});
      } else {
        // Single frame
        th_reshaped = th.view({1, 1, 3}).expand({joints.size(0), joints.size(1), 3});
      }
      joints = joints + th_reshaped;
    }

    return joints;
  }

 public:
  SMPLModel() {
    using path = std::filesystem::path;
    path param_dir("../data/opt/");

    v_template =
        load_dense_matrix_as_torch((param_dir / "SMPL_NEUTRAL_v_template.json").generic_string());
    weights =
        load_dense_matrix_as_torch((param_dir / "SMPL_NEUTRAL_weights.json").generic_string());
    j_regressor =
        load_sparse_matrix_as_torch((param_dir / "SMPL_NEUTRAL_J_regressor.json").generic_string());
    j_regressor_body25 = load_dense_matrix_as_torch(
        (param_dir / "SMPL_NEUTRAL_J_regressor_body25.json").generic_string());

    auto [posedirs_data, posedirs_shape] =
        load_tensor<double>((param_dir / "SMPL_NEUTRAL_posedirs.json").generic_string());
    std::vector<int64_t> posedirs_torch_shape;
    for (auto s : posedirs_shape) posedirs_torch_shape.push_back(static_cast<int64_t>(s));
    posedirs = torch::from_blob(posedirs_data.data(), posedirs_torch_shape, torch::kFloat64)
                   .clone()
                   .to(torch::kFloat32);

    auto [shapedirs_data, shapedirs_shape] =
        load_tensor<double>((param_dir / "SMPL_NEUTRAL_shapedirs.json").generic_string());
    std::vector<int64_t> shapedirs_torch_shape;
    for (auto s : shapedirs_shape) shapedirs_torch_shape.push_back(static_cast<int64_t>(s));
    shapedirs = torch::from_blob(shapedirs_data.data(), shapedirs_torch_shape, torch::kFloat64)
                    .clone()
                    .to(torch::kFloat32);

    auto [kintree_table_data, kintree_table_shape] =
        load_tensor<uint32_t>((param_dir / "SMPL_NEUTRAL_kintree_table.json").generic_string());
    std::copy_n(kintree_table_data.begin(), kintree_table_shape.back(),
                std::back_inserter(parents));
  }

  torch::Tensor forward(torch::Tensor betas, torch::Tensor poses, torch::Tensor rh,
                        torch::Tensor th) {
    // Ensure input tensors are on correct device and dtype
    if (betas.defined()) betas = betas.to(v_template.device()).to(v_template.dtype());
    if (poses.defined()) poses = poses.to(v_template.device()).to(v_template.dtype());
    if (rh.defined()) rh = rh.to(v_template.device()).to(v_template.dtype());
    if (th.defined()) th = th.to(v_template.device()).to(v_template.dtype());

    try {
      // 1. Shape blending: v_template + shapedirs * betas
      auto v_shaped = apply_shape_blend(betas);

      // 2. Pose blending: add pose-dependent deformations
      v_shaped = apply_pose_blend(v_shaped, poses);

      // 3. Linear Blend Skinning: apply joint transformations
      auto verts = compute_lbs(v_shaped, poses);

      // 4. Extract body25 keypoints from transformed vertices
      auto joints = torch::einsum("bik,ji->bjk", {verts, j_regressor_body25});  // [1, 25, 3]

      // 5. Apply global rotation and translation
      joints = apply_global_transform(joints, rh, th);

      return joints;
    } catch (const std::exception& e) {
      std::cout << "Error in forward: " << e.what() << std::endl;
      throw;
    }
  }
};

// Limb length loss (matches original implementation)
torch::Tensor compute_limb_length_loss(const torch::Tensor& pred_keypoints,
                                       const std::vector<float>& target_keypoints) {
  // Kinematic tree matching original implementation
  std::vector<std::pair<int, int>> kintree = {
      {8, 1}, {2, 5}, {2, 3}, {5, 6}, {3, 4}, {6, 7},  {2, 3},  {5, 6},   {3, 4},   {6, 7},
      {2, 3}, {5, 6}, {3, 4}, {6, 7}, {1, 0}, {9, 12}, {9, 10}, {10, 11}, {12, 13}, {13, 14}};

  int batch_size = pred_keypoints.size(0);
  int num_edges = kintree.size();

  // Compute predicted limb lengths
  torch::Tensor pred_lengths = torch::zeros({batch_size, num_edges});
  for (size_t i = 0; i < kintree.size(); i++) {
    auto v1 = pred_keypoints.index({torch::indexing::Slice(), kintree[i].first});
    auto v2 = pred_keypoints.index({torch::indexing::Slice(), kintree[i].second});
    auto diff = v2 - v1;
    auto length = torch::norm(diff, 2, -1);  // L2 norm along xyz dimension
    pred_lengths.index_put_({torch::indexing::Slice(), static_cast<int64_t>(i)}, length);
  }

  // Compute target limb lengths from keypoints3d data
  torch::Tensor target_lengths = torch::zeros({batch_size, num_edges});
  torch::Tensor confidence = torch::zeros({batch_size, num_edges});

  int num_keypoints = target_keypoints.size() / (batch_size * 4);
  for (int b = 0; b < batch_size; b++) {
    for (size_t i = 0; i < kintree.size(); i++) {
      int idx1 = kintree[i].first;
      int idx2 = kintree[i].second;

      // Extract v1 and v2 from target_keypoints (format: [x,y,z,conf])
      float v1[4] = {target_keypoints[b * num_keypoints * 4 + idx1 * 4],
                     target_keypoints[b * num_keypoints * 4 + idx1 * 4 + 1],
                     target_keypoints[b * num_keypoints * 4 + idx1 * 4 + 2],
                     target_keypoints[b * num_keypoints * 4 + idx1 * 4 + 3]};
      float v2[4] = {target_keypoints[b * num_keypoints * 4 + idx2 * 4],
                     target_keypoints[b * num_keypoints * 4 + idx2 * 4 + 1],
                     target_keypoints[b * num_keypoints * 4 + idx2 * 4 + 2],
                     target_keypoints[b * num_keypoints * 4 + idx2 * 4 + 3]};

      // Target limb length (including confidence in calculation - matches original BUG)
      float diff[4] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2], v2[3] - v1[3]};
      float target_length =
          sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + diff[3] * diff[3]);
      target_lengths[b][i] = target_length;

      // Confidence is minimum of both joint confidences
      confidence[b][i] = std::min(v1[3], v2[3]);
    }
  }

  // Compute loss: num += (pred - target)^2 * conf, denom += conf
  auto diff = pred_lengths - target_lengths;
  auto squared_diff = diff * diff;
  auto weighted_error = squared_diff * confidence;
  auto num = weighted_error.sum();
  auto denom = confidence.sum();
  auto loss = num / (denom + 1e-5f);

  return loss;
}

// Smooth loss for temporal smoothness
torch::Tensor compute_smooth_loss(const torch::Tensor& values,
                                  const std::vector<float>& window_heights = {0.5f, 0.3f, 0.1f,
                                                                              0.1f},
                                  bool order2 = true) {
  // values: [frames, num_items, dims]
  // Matches original implementation exactly
  int num_frames = values.size(0);
  int num_items = values.size(1);

  torch::Tensor total_loss = torch::zeros({}, values.options());

  for (size_t k = 0; k < window_heights.size(); k++) {
    torch::Tensor sq_sum = torch::zeros({}, values.options());

    if (order2) {
      if (num_frames < static_cast<int>(k + 3)) continue;

      // Second order differences: (values[b+k+2] - values[b+1]) - (values[b+k+1] - values[b])
      // Original: d1 = values[b+k+1] - values[b], d2 = values[b+k+2] - values[b+1], d = d2 - d1
      for (int b = 0; b < num_frames - static_cast<int>(k + 2); b++) {
        auto v0 = values[b];
        auto v1 = values[b + k + 1];
        auto v2 = values[b + k + 2];
        auto v_next = values[b + 1];

        auto d1 = v1 - v0;
        auto d2 = v2 - v_next;
        auto d = d2 - d1;
        sq_sum += (d * d).sum();  // Just sum, don't divide yet
      }
      // Original: sq_sum /= T((values_shape[0] - (k + 2)) * values_shape[1]);
      sq_sum /= ((num_frames - static_cast<int>(k + 2)) * num_items);
    } else {
      if (num_frames < static_cast<int>(k + 2)) continue;

      // First order differences
      for (int b = 0; b < num_frames - static_cast<int>(k + 1); b++) {
        auto d = values[b + k + 1] - values[b];
        sq_sum += (d * d).sum();  // Just sum, don't divide yet
      }
      // Original: sq_sum /= T((values_shape[0] - (k + 1)) * values_shape[1]);
      sq_sum /= ((num_frames - static_cast<int>(k + 1)) * num_items);
    }

    total_loss += sq_sum * window_heights[k];
  }

  return total_loss;
}

torch::Tensor compute_loss(const torch::Tensor& pred_keypoints,
                           const torch::Tensor& target_keypoints,
                           const std::vector<int>& indices = {},
                           const std::string& phase = "keypoints3d") {
  // pred_keypoints: [1, 25, 3] - single pose prediction
  // target_keypoints: [10, 25, 4] - multiple frames with confidence

  // Extract confidence and xyz from target
  auto confidence = target_keypoints.index({"...", 3});                             // [10, 25]
  auto target_xyz = target_keypoints.index({"...", torch::indexing::Slice(0, 3)});  // [10, 25, 3]

  // pred_keypoints is [1, 25, 3], need to squeeze batch dim
  auto pred_xyz = pred_keypoints.squeeze(0);  // [25, 3]

  // If indices provided, only use those keypoints
  std::vector<int> use_indices = indices.empty() ? std::vector<int>() : indices;
  if (use_indices.empty()) {
    use_indices.resize(pred_xyz.size(0));
    std::iota(use_indices.begin(), use_indices.end(), 0);  // [0, 1, 2, ..., 24]
  }

  // Original implementation loops over frames: for (b = 0; b < keypoints3d_shape[0]; b++)
  // Computing: e = target - pred for each frame
  torch::Tensor num = torch::zeros({}, torch::kFloat32);
  torch::Tensor denom = torch::zeros({}, torch::kFloat32);

  int num_frames = target_xyz.size(0);
  for (int b = 0; b < num_frames; b++) {
    for (int idx : use_indices) {
      auto pred_pt = pred_xyz[idx];         // [3]
      auto target_pt = target_xyz[b][idx];  // [3]
      auto conf = confidence[b][idx];       // scalar

      // e = target - pred
      auto diff = target_pt - pred_pt;     // [3]
      auto sq_dist = (diff * diff).sum();  // scalar - sum over xyz

      // num += sq_dist * conf, denom += conf
      num += sq_dist * conf;
      denom += conf;
    }
  }

  // Compute loss = num / (1e-5 + denom)
  auto loss = num / (denom + 1e-5f);

  return loss;
}

int main() {
  std::cout << "Starting optimization with LibTorch..." << std::endl;
  auto total_start = std::chrono::high_resolution_clock::now();

  // Load data
  auto keypoints3d = load_tensor_as_torch("../data/opt/observations_keypoints3d.json");
  std::cout << "keypoints3d loaded shape: " << keypoints3d.sizes() << std::endl;
  auto poses = load_tensor_as_torch("../data/opt/params_poses.json");
  auto shapes = load_tensor_as_torch("../data/opt/params_shapes.json");
  auto rh = load_tensor_as_torch("../data/opt/params_Rh.json");
  auto th = load_tensor_as_torch("../data/opt/params_Th.json");

  // Initialize model
  SMPLModel model;

  std::cout << "\n=== Phase 1: Fitting shape ===" << std::endl;
  auto phase_start = std::chrono::high_resolution_clock::now();
  {
    auto shapes_opt = shapes.clone().detach().requires_grad_(true);
    torch::optim::Adam optimizer({shapes_opt}, torch::optim::AdamOptions(0.01));

    // Convert keypoints3d tensor to vector for limb_length_loss
    auto keypoints_vec = std::vector<float>(keypoints3d.data_ptr<float>(),
                                            keypoints3d.data_ptr<float>() + keypoints3d.numel());

    float prev_loss = std::numeric_limits<float>::max();
    for (int i = 0; i < 1000; i++) {  // Match original implementation
      optimizer.zero_grad();
      auto pred_keypoints = model.forward(shapes_opt, poses, rh, th);

      // Use limb_length_loss + regression_loss (matching original Phase 1)
      auto limb_loss = compute_limb_length_loss(pred_keypoints, keypoints_vec);
      auto reg_loss = (shapes_opt * shapes_opt).sum() / shapes_opt.size(0);  // L2 regularization

      auto loss = limb_loss * 100.0f + reg_loss * 0.1f;  // Exact weights from original

      loss.backward();
      optimizer.step();

      float current_loss = loss.item<float>();
      if (i == 0 || i == 999) {
        std::cout << "Iteration " << i << ": loss = " << current_loss << std::endl;
      }

      // Check for convergence like the original
      if (std::abs(prev_loss - current_loss) < 1e-7) {
        std::cout << "Converged at iteration " << i << ": loss = " << current_loss << std::endl;
        break;
      }
      prev_loss = current_loss;
    }
    shapes = shapes_opt.detach();
  }
  auto phase_end = std::chrono::high_resolution_clock::now();
  auto phase_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);
  std::cout << "Phase 1 completed in " << phase_duration.count() << " ms" << std::endl;

  std::cout << "\n=== Phase 2: Initializing RT ===" << std::endl;
  phase_start = std::chrono::high_resolution_clock::now();
  {
    // Phase 2 optimizes rh and th for all frames simultaneously
    // rh: [10, 3], th: [10, 3]
    auto rh_opt = rh.clone().requires_grad_(true);
    auto th_opt = th.clone().requires_grad_(true);
    // Use Adam optimizer
    torch::optim::Adam optimizer({rh_opt, th_opt}, torch::optim::AdamOptions(0.01));

    // Phase 2 uses only 4 keypoints: indices {2, 5, 9, 12}
    std::vector<int> phase2_indices = {2, 5, 9, 12};

    float prev_loss = std::numeric_limits<float>::max();
    for (int iteration = 0; iteration < 1000; iteration++) {
      optimizer.zero_grad();

      // Compute keypoints for all frames
      int num_frames = keypoints3d.size(0);
      std::vector<torch::Tensor> pred_list;

      for (int frame = 0; frame < num_frames; frame++) {
        auto frame_rh = rh_opt.select(0, frame).unsqueeze(0);  // [1, 3]
        auto frame_th = th_opt.select(0, frame).unsqueeze(0);  // [1, 3]

        auto pred_keypoints = model.forward(shapes, poses, frame_rh, frame_th);
        pred_list.push_back(pred_keypoints.squeeze(0));  // [25, 3]
      }

      // Stack all predictions: [10, 25, 3]
      auto all_pred_keypoints = torch::stack(pred_list, 0);

      // Compute keypoints3d loss
      torch::Tensor keypoints3d_loss = torch::zeros({}, torch::kFloat32);
      for (int frame = 0; frame < num_frames; frame++) {
        auto pred = all_pred_keypoints.select(0, frame).unsqueeze(0);  // [1, 25, 3]
        auto target = keypoints3d.select(0, frame).unsqueeze(0);       // [1, 25, 4]
        keypoints3d_loss += compute_loss(pred, target, phase2_indices);
      }
      keypoints3d_loss /= num_frames;

      // Compute smooth losses
      auto smooth_keypoints_loss =
          compute_smooth_loss(all_pred_keypoints, {0.5f, 0.3f, 0.1f, 0.1f});
      auto th_reshaped = th_opt.unsqueeze(1);  // [10, 1, 3]
      auto smooth_th_loss = compute_smooth_loss(th_reshaped, {0.5f, 0.3f, 0.1f, 0.1f});

      // Total loss
      auto loss = keypoints3d_loss * 100.0f +
                  (smooth_keypoints_loss * 10.0f + smooth_th_loss * 100.0f) * 1.0f;

      loss.backward();
      optimizer.step();

      float current_loss = loss.item<float>();
      if (iteration == 0 || iteration == 999) {
        std::cout << "Iteration " << iteration << ": loss = " << current_loss << std::endl;
      }

      // Check for convergence
      if (std::abs(prev_loss - current_loss) < 1e-7) {
        std::cout << "Converged at iteration " << iteration << ": loss = " << current_loss
                  << std::endl;
        break;
      }
      prev_loss = current_loss;
    }
    rh = rh_opt.detach();
    th = th_opt.detach();
  }
  phase_end = std::chrono::high_resolution_clock::now();
  phase_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);
  std::cout << "Phase 2 completed in " << phase_duration.count() << " ms" << std::endl;

  std::cout << "\n=== Phase 3: Refining pose ===" << std::endl;
  phase_start = std::chrono::high_resolution_clock::now();
  {
    // Initialize prior loss
    PriorLoss prior_loss;

    auto poses_opt = poses.clone().requires_grad_(true);  // [10, 69]
    auto rh_opt = rh.clone().requires_grad_(true);
    auto th_opt = th.clone().requires_grad_(true);
    torch::optim::Adam optimizer({poses_opt, rh_opt, th_opt}, torch::optim::AdamOptions(0.01));

    float prev_loss = std::numeric_limits<float>::max();
    for (int i = 0; i < 1000; i++) {  // Match original implementation
      optimizer.zero_grad();

      // Compute keypoints for all frames
      int num_frames = keypoints3d.size(0);
      std::vector<torch::Tensor> pred_list;

      for (int frame = 0; frame < num_frames; frame++) {
        auto frame_poses = poses_opt.select(0, frame).unsqueeze(0);  // [1, 69]
        auto frame_rh = rh_opt.select(0, frame).unsqueeze(0);        // [1, 3]
        auto frame_th = th_opt.select(0, frame).unsqueeze(0);        // [1, 3]

        auto pred_keypoints = model.forward(shapes, frame_poses, frame_rh, frame_th);
        pred_list.push_back(pred_keypoints.squeeze(0));  // [25, 3]
      }

      // Stack all predictions: [10, 25, 3]
      auto all_pred_keypoints = torch::stack(pred_list, 0);

      // Compute keypoints3d loss (all 25 keypoints in Phase 3)
      // Match original: sum over all frames, then divide by total confidence
      torch::Tensor num = torch::zeros({}, torch::kFloat32);
      torch::Tensor denom = torch::zeros({}, torch::kFloat32);
      for (int frame = 0; frame < num_frames; frame++) {
        auto pred_xyz = all_pred_keypoints.select(0, frame);      // [25, 3]
        auto target_xyzc = keypoints3d.select(0, frame);          // [25, 4]
        auto target_xyz = target_xyzc.slice(1, 0, 3);             // [25, 3]
        auto confidence = target_xyzc.slice(1, 3, 4).squeeze(1);  // [25]

        auto diff = pred_xyz - target_xyz;     // [25, 3]
        auto sq_dist = (diff * diff).sum(1);   // [25] - sum over xyz
        auto weighted = sq_dist * confidence;  // [25]
        num += weighted.sum();
        denom += confidence.sum();
      }
      auto keypoints3d_loss = num / (denom + 1e-5f);

      // Compute smooth losses
      // smooth_poses: poses_opt is [10, 69], reshape to [10, 1, 69]
      auto poses_reshaped = poses_opt.unsqueeze(1);  // [10, 1, 69]
      auto smooth_poses_loss = compute_smooth_loss(poses_reshaped, {0.5f, 0.3f, 0.1f, 0.1f});

      // smooth_keypoints: [10, 25, 3]
      auto smooth_keypoints_loss =
          compute_smooth_loss(all_pred_keypoints, {0.5f, 0.3f, 0.1f, 0.1f});

      // smooth_th: [10, 1, 3]
      auto th_reshaped = th_opt.unsqueeze(1);
      auto smooth_th_loss = compute_smooth_loss(th_reshaped, {0.5f, 0.3f, 0.1f, 0.1f});

      // Compute prior loss
      auto prior_loss_value = prior_loss.compute(poses_opt);  // Use original poses

      // Total loss matching original implementation
      // keypoints3d_loss * 1000 + smooth_loss * 1 + prior_loss * 0.1
      auto smooth_loss =
          smooth_poses_loss * 100.0f + smooth_keypoints_loss * 10.0f + smooth_th_loss * 10.0f;
      auto loss = keypoints3d_loss * 1000.0f + smooth_loss * 1.0f + prior_loss_value * 0.1f;

      loss.backward();
      optimizer.step();

      float current_loss = loss.item<float>();
      if (i == 0 || i == 999) {
        std::cout << "Iteration " << i << ": loss = " << current_loss << std::endl;
      }

      // Check for convergence like the original
      // Use relative change instead of absolute for large loss values
      float relative_change = std::abs(prev_loss - current_loss) / (prev_loss + 1e-8);
      if (relative_change < 1e-6) {  // Very strict convergence for Phase 3
        std::cout << "Converged at iteration " << i << ": loss = " << current_loss << std::endl;
        break;
      }
      prev_loss = current_loss;
    }
    poses = poses_opt.detach();
    rh = rh_opt.detach();
    th = th_opt.detach();
  }
  phase_end = std::chrono::high_resolution_clock::now();
  phase_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);
  std::cout << "Phase 3 completed in " << phase_duration.count() << " ms" << std::endl;

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
  std::cout << "\n=== Total optimization time: " << total_duration.count()
            << " ms ===" << std::endl;

  return 0;
}
