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
    // Batch process all frames at once

    // Compute differences from all gaussian means for all frames
    // poses: [B, 69], gmm_means: [8, 69]
    // Expand: poses: [B, 1, 69], gmm_means: [1, 8, 69]
    auto poses_expanded = poses.unsqueeze(1);      // [B, 1, 69]
    auto means_expanded = gmm_means.unsqueeze(0);  // [1, 8, 69]
    auto d = poses_expanded - means_expanded;      // [B, 8, 69]

    // Compute prec_d = d @ precision^T for each gaussian and frame
    // d: [B, 8, 69], gmm_precisions: [8, 69, 69]
    // Expand d to [B, 8, 1, 69] and precisions to [1, 8, 69, 69]
    auto d_expanded = d.unsqueeze(2);                                   // [B, 8, 1, 69]
    auto prec_expanded = gmm_precisions.unsqueeze(0);                   // [1, 8, 69, 69]
    auto prec_d = torch::matmul(d_expanded, prec_expanded).squeeze(2);  // [B, 8, 69]

    // Compute prec_dd = d @ prec_d for each gaussian and frame
    auto prec_dd = (d * prec_d).sum(2);  // [B, 8]

    // Compute log-likelihood for each gaussian and frame
    auto nll_expanded = nll_weights.unsqueeze(0);        // [1, 8]
    auto loglikelihood = 0.5f * prec_dd + nll_expanded;  // [B, 8]

    // Take minimum log-likelihood across gaussians for each frame
    auto min_likelihood = std::get<0>(loglikelihood.min(1));  // [B]

    // Average over frames
    auto total_loss = min_likelihood.mean();

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
  // betas: [1, 10] -> output: [1, 6890, 3]
  torch::Tensor apply_shape_blend(const torch::Tensor& betas) {
    auto v_shaped = v_template.clone();  // [6890, 3]

    if (betas.defined() && betas.numel() > 0) {
      // shapedirs: [6890, 3, 10], betas: [1, 10] -> [6890, 3]
      auto shape_disps = torch::matmul(shapedirs, betas.squeeze(0));  // [6890, 3]
      v_shaped = v_shaped + shape_disps;
    }

    return v_shaped.unsqueeze(0);  // [1, 6890, 3]
  }

  // Apply pose-dependent vertex deformations
  // v_shaped: [1, 6890, 3], poses: [B, 69]
  torch::Tensor apply_pose_blend(const torch::Tensor& v_shaped, const torch::Tensor& poses) {
    if (!poses.defined() || poses.numel() == 0) {
      return v_shaped;
    }

    int batch_size = poses.size(0);
    auto poses_reshaped = poses.view({batch_size, 23, 3});  // [B, 23, 3]

    // Convert all joint poses to rotation matrices (batch operation)
    auto rot_mats_3x3 = batch_rodrigues(poses_reshaped.view({-1, 3}));  // [B*23, 3, 3]
    auto rot_mats = rot_mats_3x3.view({batch_size, 23, 9});             // [B, 23, 9]

    // Pose feature: rot_mats - Identity
    auto rot_mats_feat = rot_mats.clone();
    rot_mats_feat.select(2, 0) -= 1.0f;  // R[0,0] -= 1
    rot_mats_feat.select(2, 4) -= 1.0f;  // R[1,1] -= 1
    rot_mats_feat.select(2, 8) -= 1.0f;  // R[2,2] -= 1

    // Compute pose-dependent vertex displacements
    auto pose_feature_flat = rot_mats_feat.view({batch_size, -1});  // [B, 207]
    // posedirs: [6890, 3, 207], pose_feature: [B, 207] -> [B, 6890, 3]
    auto posedirs_2d = posedirs.view({-1, 207});  // [6890*3, 207]
    auto pose_offset = torch::matmul(posedirs_2d, pose_feature_flat.t())
                           .t()
                           .view({batch_size, 6890, 3});  // [B, 6890, 3]

    // Expand v_shaped to batch size and add pose offset
    auto v_shaped_expanded = v_shaped.expand({batch_size, -1, -1});  // [B, 6890, 3]
    return v_shaped_expanded + pose_offset;
  }

  // Compute Linear Blend Skinning
  // v_posed: [B, 6890, 3], poses: [B, 69]
  torch::Tensor compute_lbs(const torch::Tensor& v_posed, const torch::Tensor& poses) {
    if (!poses.defined() || poses.numel() == 0) {
      return v_posed;
    }

    int batch_size = poses.size(0);

    // 1. Compute 24 joints from v_posed (batched)
    // j_regressor: [24, 6890], v_posed: [B, 6890, 3]
    auto j_regressor_expanded =
        j_regressor.unsqueeze(0).expand({batch_size, -1, -1});   // [B, 24, 6890]
    auto joints_24 = torch::bmm(j_regressor_expanded, v_posed);  // [B, 24, 3]

    // 2. Build rotation matrices for each joint (batched)
    auto poses_reshaped = poses.view({batch_size, 23, 3});              // [B, 23, 3]
    auto rot_mats_3x3 = batch_rodrigues(poses_reshaped.view({-1, 3}));  // [B*23, 3, 3]
    rot_mats_3x3 = rot_mats_3x3.view({batch_size, 23, 3, 3});           // [B, 23, 3, 3]

    // 3. Build 4x4 transformation matrices for kinematic chain (batched)
    std::vector<torch::Tensor> transform_mats_list;
    for (int j = 0; j < 24; j++) {
      auto transform_mat = torch::eye(4, poses.options())
                               .unsqueeze(0)
                               .expand({batch_size, 4, 4})
                               .clone();  // [B, 4, 4]

      auto joint = joints_24.select(1, j);  // [B, 3]
      torch::Tensor parent_joint = (j == 0) ? torch::zeros({batch_size, 3}, poses.options())
                                            : joints_24.select(1, parents[j]);  // [B, 3]
      auto rel_joint = joint - parent_joint;                                    // [B, 3]

      // Set rotation part (only for non-root joints)
      if (j > 0) {
        transform_mat.narrow(1, 0, 3).narrow(2, 0, 3) = rot_mats_3x3.select(1, j - 1);  // [B, 3, 3]
      }

      // Set translation part
      transform_mat.narrow(1, 0, 3).select(2, 3) = rel_joint;  // [B, 3]

      // Multiply with parent transformation
      if (j > 0 && parents[j] >= 0) {
        transform_mat = torch::bmm(transform_mats_list[parents[j]], transform_mat);  // [B, 4, 4]
      }

      transform_mats_list.push_back(transform_mat);
    }

    auto transform_mats = torch::stack(transform_mats_list, 1);  // [B, 24, 4, 4]

    // 4. Convert to relative transformations (batched)
    auto joints_24_expanded = joints_24.unsqueeze(2).unsqueeze(3);    // [B, 24, 1, 1, 3]
    auto rot_parts = transform_mats.narrow(2, 0, 3).narrow(3, 0, 3);  // [B, 24, 3, 3]
    auto trans_parts = transform_mats.narrow(2, 0, 3).select(3, 3);   // [B, 24, 3]

    auto new_trans =
        trans_parts - torch::matmul(rot_parts, joints_24.unsqueeze(3)).squeeze(3);  // [B, 24, 3]

    auto rel_transform_mats = torch::eye(4, poses.options())
                                  .unsqueeze(0)
                                  .unsqueeze(0)
                                  .expand({batch_size, 24, 4, 4})
                                  .clone();
    rel_transform_mats.narrow(2, 0, 3).narrow(3, 0, 3) = rot_parts;
    rel_transform_mats.narrow(2, 0, 3).select(3, 3) = new_trans;

    // 5. Blend transformations using skinning weights (batched)
    // weights: [6890, 24], rel_transform_mats: [B, 24, 4, 4]
    auto weights_expanded = weights.unsqueeze(0).unsqueeze(3).unsqueeze(4);  // [1, 6890, 24, 1, 1]
    auto rel_mats_expanded = rel_transform_mats.unsqueeze(1);                // [B, 1, 24, 4, 4]
    auto blended_mats = (weights_expanded * rel_mats_expanded).sum(2);       // [B, 6890, 4, 4]

    // 6. Apply blended transformations to vertices (batched)
    auto v_posed_homo = torch::cat({v_posed, torch::ones({batch_size, 6890, 1}, poses.options())},
                                   2);  // [B, 6890, 4]
    auto verts_homo = torch::bmm(blended_mats.view({-1, 4, 4}), v_posed_homo.view({-1, 4, 1}))
                          .view({batch_size, 6890, 4});  // [B, 6890, 4]
    auto verts = verts_homo.narrow(2, 0, 3);             // [B, 6890, 3]

    return verts;
  }

  // Apply global rotation (rh) and translation (th)
  // joints: [B, 25, 3], rh: [B, 3], th: [B, 3]
  torch::Tensor apply_global_transform(torch::Tensor joints, const torch::Tensor& rh,
                                       const torch::Tensor& th) {
    int batch_size = joints.size(0);

    // Apply Rodrigues rotation
    if (rh.defined() && rh.numel() > 0) {
      auto R = batch_rodrigues(rh);                    // [B, 3, 3]
      joints = torch::bmm(joints, R.transpose(1, 2));  // [B, 25, 3]
    }

    // Add translation
    if (th.defined() && th.numel() > 0) {
      auto th_reshaped = th.unsqueeze(1).expand({batch_size, joints.size(1), 3});  // [B, 25, 3]
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

    // Determine batch size from poses
    int batch_size = poses.size(0);

    // 1. Shape blending: v_template + shapedirs * betas (shared across frames)
    auto v_shaped = apply_shape_blend(betas);  // [1, 6890, 3]

    // 2. Pose blending: add pose-dependent deformations
    v_shaped = apply_pose_blend(v_shaped, poses);  // [B, 6890, 3]

    // 3. Linear Blend Skinning: apply joint transformations
    auto verts = compute_lbs(v_shaped, poses);

    // 4. Extract body25 keypoints from transformed vertices
    // verts: [B, 6890, 3], j_regressor_body25: [25, 6890]
    // Expand j_regressor_body25 to [B, 25, 6890] and use bmm
    auto j_reg_expanded =
        j_regressor_body25.unsqueeze(0).expand({batch_size, -1, -1});  // [B, 25, 6890]
    auto joints = torch::bmm(j_reg_expanded, verts);                   // [B, 25, 3]

    // 5. Apply global rotation and translation
    joints = apply_global_transform(joints, rh, th);

    return joints;
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
  int num_keypoints = target_keypoints.size() / (batch_size * 4);

  // Convert target_keypoints to tensor [B, num_keypoints, 4]
  auto target_tensor = torch::from_blob(const_cast<float*>(target_keypoints.data()),
                                        {batch_size, num_keypoints, 4}, torch::kFloat32)
                           .clone();

  // Compute predicted limb lengths (batched)
  std::vector<torch::Tensor> pred_lengths_list;
  std::vector<torch::Tensor> target_lengths_list;
  std::vector<torch::Tensor> confidence_list;

  for (size_t i = 0; i < kintree.size(); i++) {
    int idx1 = kintree[i].first;
    int idx2 = kintree[i].second;

    // Predicted limb length
    auto v1_pred = pred_keypoints.index({torch::indexing::Slice(), idx1});  // [B, 3]
    auto v2_pred = pred_keypoints.index({torch::indexing::Slice(), idx2});  // [B, 3]
    auto diff_pred = v2_pred - v1_pred;
    auto length_pred = torch::norm(diff_pred, 2, -1);  // [B]
    pred_lengths_list.push_back(length_pred);

    // Target limb length (including confidence in calculation - matches original BUG)
    auto v1_target = target_tensor.index({torch::indexing::Slice(), idx1});  // [B, 4]
    auto v2_target = target_tensor.index({torch::indexing::Slice(), idx2});  // [B, 4]
    auto diff_target = v2_target - v1_target;                                // [B, 4]
    auto length_target = torch::norm(diff_target, 2, -1);                    // [B]
    target_lengths_list.push_back(length_target);

    // Confidence is minimum of both joint confidences
    auto conf1 = v1_target.select(1, 3);   // [B]
    auto conf2 = v2_target.select(1, 3);   // [B]
    auto conf = torch::min(conf1, conf2);  // [B]
    confidence_list.push_back(conf);
  }

  auto pred_lengths = torch::stack(pred_lengths_list, 1);      // [B, num_edges]
  auto target_lengths = torch::stack(target_lengths_list, 1);  // [B, num_edges]
  auto confidence = torch::stack(confidence_list, 1);          // [B, num_edges]

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

// Compute keypoints loss (batched version)
// pred_keypoints: [B, 25, 3] - batch of predictions
// target_keypoints: [B, 25, 4] - batch of targets with confidence
// indices: optional list of keypoint indices to use
torch::Tensor compute_keypoints_loss(const torch::Tensor& pred_keypoints,
                                     const torch::Tensor& target_keypoints,
                                     const std::vector<int>& indices = {}) {
  // Extract confidence and xyz from target
  auto confidence =
      target_keypoints.index({torch::indexing::Slice(), torch::indexing::Slice(), 3});  // [B, 25]
  auto target_xyz = target_keypoints.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                            torch::indexing::Slice(0, 3)});  // [B, 25, 3]

  torch::Tensor num;
  torch::Tensor denom;

  if (indices.empty()) {
    // Use all keypoints
    auto diff = pred_keypoints - target_xyz;  // [B, 25, 3]
    auto sq_dist = (diff * diff).sum(2);      // [B, 25] - sum over xyz
    auto weighted = sq_dist * confidence;     // [B, 25]
    num = weighted.sum();
    denom = confidence.sum();
  } else {
    // Use only specified keypoints
    num = torch::zeros({}, torch::kFloat32);
    denom = torch::zeros({}, torch::kFloat32);

    for (int idx : indices) {
      auto pred_pt = pred_keypoints.index({torch::indexing::Slice(), idx});  // [B, 3]
      auto target_pt = target_xyz.index({torch::indexing::Slice(), idx});    // [B, 3]
      auto conf = confidence.index({torch::indexing::Slice(), idx});         // [B]

      auto diff = target_pt - pred_pt;      // [B, 3]
      auto sq_dist = (diff * diff).sum(1);  // [B]
      auto weighted = sq_dist * conf;       // [B]
      num += weighted.sum();
      denom += conf.sum();
    }
  }

  return num / (denom + 1e-5f);
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
    torch::optim::Adam optimizer({shapes_opt}, torch::optim::AdamOptions(0.05));

    // Convert keypoints3d tensor to vector for limb_length_loss
    auto keypoints_vec = std::vector<float>(keypoints3d.data_ptr<float>(),
                                            keypoints3d.data_ptr<float>() + keypoints3d.numel());

    float prev_loss = std::numeric_limits<float>::max();
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    float current_lr = 0.05f;

    for (int i = 0; i < 1000; i++) {
      optimizer.zero_grad();

      // Compute predictions for all frames in one batch
      // Note: shapes is [1, 10] and applies to all frames
      auto all_pred_keypoints = model.forward(shapes_opt, poses, rh, th);  // [10, 25, 3]

      // Use limb_length_loss + regression_loss (matching original Phase 1)
      auto limb_loss = compute_limb_length_loss(all_pred_keypoints, keypoints_vec);
      auto reg_loss = (shapes_opt * shapes_opt).sum() / shapes_opt.size(0);  // L2 regularization

      auto loss = limb_loss * 100.0f + reg_loss * 0.1f;  // Exact weights from original

      loss.backward();
      optimizer.step();

      float current_loss = loss.item<float>();
      if (i == 0 || i == 999) {
        std::cout << "Iteration " << i << ": loss = " << current_loss << ", lr = " << current_lr
                  << std::endl;
      }

      // Dynamic learning rate reduction
      if (current_loss < best_loss - 1e-6f) {
        best_loss = current_loss;
        patience_counter = 0;
      } else {
        patience_counter++;
        if (patience_counter >= 20 && current_lr > 1e-5f) {
          current_lr *= 0.5f;
          for (auto& param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(current_lr);
          }
          patience_counter = 0;
          std::cout << "  -> Reducing lr to " << current_lr << std::endl;
        }
      }

      // Relaxed convergence check
      if (std::abs(prev_loss - current_loss) < 1e-5f) {
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
    // Use Adam optimizer with higher learning rate
    torch::optim::Adam optimizer({rh_opt, th_opt}, torch::optim::AdamOptions(0.05));

    // Phase 2 uses only 4 keypoints: indices {2, 5, 9, 12}
    std::vector<int> phase2_indices = {2, 5, 9, 12};

    float prev_loss = std::numeric_limits<float>::max();
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    float current_lr = 0.05f;

    for (int iteration = 0; iteration < 1000; iteration++) {
      optimizer.zero_grad();

      // Compute keypoints for all frames in one batch
      // Note: shapes is [1, 10] and applies to all frames
      auto all_pred_keypoints = model.forward(shapes, poses, rh_opt, th_opt);  // [10, 25, 3]

      // Compute keypoints3d loss (batched)
      auto keypoints3d_loss =
          compute_keypoints_loss(all_pred_keypoints, keypoints3d, phase2_indices);

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
        std::cout << "Iteration " << iteration << ": loss = " << current_loss
                  << ", lr = " << current_lr << std::endl;
      }

      // Dynamic learning rate reduction
      if (current_loss < best_loss - 1e-6f) {
        best_loss = current_loss;
        patience_counter = 0;
      } else {
        patience_counter++;
        if (patience_counter >= 20 && current_lr > 1e-5f) {
          current_lr *= 0.5f;
          for (auto& param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(current_lr);
          }
          patience_counter = 0;
          std::cout << "  -> Reducing lr to " << current_lr << std::endl;
        }
      }

      // Relaxed convergence check
      if (std::abs(prev_loss - current_loss) < 1e-5f) {
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
    torch::optim::Adam optimizer({poses_opt, rh_opt, th_opt}, torch::optim::AdamOptions(0.02));

    float prev_loss = std::numeric_limits<float>::max();
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    float current_lr = 0.02f;

    for (int i = 0; i < 1000; i++) {
      optimizer.zero_grad();

      // Compute keypoints for all frames in one batch
      // Note: shapes is [1, 10] and applies to all frames
      auto all_pred_keypoints = model.forward(shapes, poses_opt, rh_opt, th_opt);  // [10, 25, 3]

      // Compute keypoints3d loss (all 25 keypoints in Phase 3, batched)
      auto keypoints3d_loss = compute_keypoints_loss(all_pred_keypoints, keypoints3d);

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
      auto prior_loss_value = prior_loss.compute(poses_opt);

      // Total loss matching original implementation
      // keypoints3d_loss * 1000 + smooth_loss * 1 + prior_loss * 0.1
      auto smooth_loss =
          smooth_poses_loss * 100.0f + smooth_keypoints_loss * 10.0f + smooth_th_loss * 10.0f;
      auto loss = keypoints3d_loss * 1000.0f + smooth_loss * 1.0f + prior_loss_value * 0.1f;

      loss.backward();
      optimizer.step();

      float current_loss = loss.item<float>();
      if (i == 0 || i == 999) {
        std::cout << "Iteration " << i << ": loss = " << current_loss << ", lr = " << current_lr
                  << std::endl;
      }

      // Dynamic learning rate reduction
      if (current_loss < best_loss - 1e-4f) {
        best_loss = current_loss;
        patience_counter = 0;
      } else {
        patience_counter++;
        if (patience_counter >= 30 && current_lr > 1e-5f) {
          current_lr *= 0.5f;
          for (auto& param_group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(param_group.options()).lr(current_lr);
          }
          patience_counter = 0;
          std::cout << "  -> Reducing lr to " << current_lr << std::endl;
        }
      }

      // Relaxed convergence check using relative change
      float relative_change = std::abs(prev_loss - current_loss) / (prev_loss + 1e-8f);
      if (relative_change < 1e-5f) {
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
