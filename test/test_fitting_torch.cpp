#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

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
    const auto data = data_vec.get<std::vector<double>>();
    tensor = torch::from_blob(const_cast<double*>(data.data()),
                              {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])},
                              torch::kFloat64)
                 .clone();
  } else if (type_str == "float32") {
    const auto data = data_vec.get<std::vector<float>>();
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

class PriorLoss {
 public:
  torch::Tensor gmm_means;
  torch::Tensor gmm_precisions;
  torch::Tensor nll_weights;

  PriorLoss() {
    using path = std::filesystem::path;
    path param_dir("../data/opt/");

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
    const auto covs =
        torch::from_blob(covs_data.data(),
                         {static_cast<int64_t>(covs_shape[0]), static_cast<int64_t>(covs_shape[1]),
                          static_cast<int64_t>(covs_shape[2])},
                         torch::kFloat64)
            .clone();

    auto [weights_data, weights_shape] =
        load_tensor<double>((param_dir / "gmm_weights.json").generic_string());
    const auto gmm_weights =
        torch::from_blob(weights_data.data(), {static_cast<int64_t>(weights_shape[0])},
                         torch::kFloat64)
            .clone();

    const auto num_gaussians = covs.size(0);
    std::vector<torch::Tensor> precisions_list;
    std::vector<double> sqrdets;

    for (int m = 0; m < num_gaussians; m++) {
      const auto cov = covs[m].to(torch::kFloat64);
      const auto det = torch::det(cov).item<double>();
      const auto sqrdet = std::sqrt(det);
      sqrdets.push_back(sqrdet);

      const auto precision = torch::inverse(cov);
      precisions_list.push_back(precision);
    }

    gmm_precisions = torch::stack(precisions_list, 0).to(torch::kFloat32);

    const double PI = 3.141592653589793;
    const double c = std::pow(2.0 * PI, 69.0 / 2.0);
    const auto min_sqrdet = *std::min_element(sqrdets.begin(), sqrdets.end());

    std::vector<float> nll_weights_vec;
    for (int m = 0; m < num_gaussians; m++) {
      const auto w = gmm_weights[m].item<double>();
      const auto nll = -std::log(w / (c * (sqrdets[m] / min_sqrdet)));
      nll_weights_vec.push_back(static_cast<float>(nll));
    }

    nll_weights =
        torch::from_blob(nll_weights_vec.data(), {num_gaussians}, torch::kFloat32).clone();
  }

  torch::Tensor compute(const torch::Tensor& poses) {
    const auto poses_expanded = poses.unsqueeze(1);
    const auto means_expanded = gmm_means.unsqueeze(0);
    const auto d = poses_expanded - means_expanded;

    const auto d_expanded = d.unsqueeze(2);
    const auto prec_expanded = gmm_precisions.unsqueeze(0);
    const auto prec_d = torch::matmul(d_expanded, prec_expanded).squeeze(2);

    const auto prec_dd = (d * prec_d).sum(2);

    const auto nll_expanded = nll_weights.unsqueeze(0);
    const auto loglikelihood = 0.5f * prec_dd + nll_expanded;

    const auto min_likelihood = std::get<0>(loglikelihood.min(1));
    return min_likelihood.mean();
  }
};

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
  torch::Tensor batch_rodrigues(const torch::Tensor& pose_vec) {
    const auto pose_reshaped = pose_vec.dim() == 1 ? pose_vec.unsqueeze(0) : pose_vec;

    const auto theta2 = (pose_reshaped * pose_reshaped).sum(1, true);
    const auto theta = torch::sqrt(theta2 + 1e-8);
    const auto w = pose_reshaped / (theta + 1e-8);

    const auto wx = w.select(1, 0).unsqueeze(1);
    const auto wy = w.select(1, 1).unsqueeze(1);
    const auto wz = w.select(1, 2).unsqueeze(1);

    const auto cos_theta = torch::cos(theta);
    const auto sin_theta = torch::sin(theta);
    const auto one_minus_cos = 1.0 - cos_theta;

    const auto r00 = cos_theta + wx * wx * one_minus_cos;
    const auto r01 = wx * wy * one_minus_cos - wz * sin_theta;
    const auto r02 = wy * sin_theta + wx * wz * one_minus_cos;
    const auto r10 = wz * sin_theta + wx * wy * one_minus_cos;
    const auto r11 = cos_theta + wy * wy * one_minus_cos;
    const auto r12 = -wx * sin_theta + wy * wz * one_minus_cos;
    const auto r20 = -wy * sin_theta + wx * wz * one_minus_cos;
    const auto r21 = wx * sin_theta + wy * wz * one_minus_cos;
    const auto r22 = cos_theta + wz * wz * one_minus_cos;

    const auto row0 = torch::cat({r00, r01, r02}, 1).unsqueeze(1);
    const auto row1 = torch::cat({r10, r11, r12}, 1).unsqueeze(1);
    const auto row2 = torch::cat({r20, r21, r22}, 1).unsqueeze(1);
    return torch::cat({row0, row1, row2}, 1);
  }

  torch::Tensor apply_shape_blend(const torch::Tensor& betas) {
    auto v_shaped = v_template.clone();

    if (betas.defined() && betas.numel() > 0) {
      const auto shape_disps = torch::matmul(shapedirs, betas.squeeze(0));
      v_shaped = v_shaped + shape_disps;
    }

    return v_shaped.unsqueeze(0);
  }

  torch::Tensor apply_pose_blend(const torch::Tensor& v_shaped, const torch::Tensor& poses) {
    if (!poses.defined() || poses.numel() == 0) {
      return v_shaped;
    }

    const auto batch_size = poses.size(0);
    const auto poses_reshaped = poses.view({batch_size, 23, 3});

    const auto rot_mats_3x3 = batch_rodrigues(poses_reshaped.view({-1, 3}));
    const auto rot_mats = rot_mats_3x3.view({batch_size, 23, 9});

    // Pose feature: rotation matrices relative to identity
    auto rot_mats_feat = rot_mats.clone();
    rot_mats_feat.select(2, 0) -= 1.0f;
    rot_mats_feat.select(2, 4) -= 1.0f;
    rot_mats_feat.select(2, 8) -= 1.0f;

    const auto pose_feature_flat = rot_mats_feat.view({batch_size, -1});
    const auto posedirs_2d = posedirs.view({-1, 207});
    const auto pose_offset =
        torch::matmul(posedirs_2d, pose_feature_flat.t()).t().view({batch_size, 6890, 3});

    const auto v_shaped_expanded = v_shaped.expand({batch_size, -1, -1});
    return v_shaped_expanded + pose_offset;
  }

  torch::Tensor compute_lbs(const torch::Tensor& v_posed, const torch::Tensor& poses) {
    if (!poses.defined() || poses.numel() == 0) {
      return v_posed;
    }

    const auto batch_size = poses.size(0);

    const auto j_regressor_expanded = j_regressor.unsqueeze(0).expand({batch_size, -1, -1});
    const auto joints_24 = torch::bmm(j_regressor_expanded, v_posed);

    const auto poses_reshaped = poses.view({batch_size, 23, 3});
    auto rot_mats_3x3 = batch_rodrigues(poses_reshaped.view({-1, 3}));
    rot_mats_3x3 = rot_mats_3x3.view({batch_size, 23, 3, 3});

    // Build 4x4 transformation matrices for kinematic chain
    std::vector<torch::Tensor> transform_mats_list;
    for (int j = 0; j < 24; j++) {
      auto transform_mat =
          torch::eye(4, poses.options()).unsqueeze(0).expand({batch_size, 4, 4}).clone();

      const auto joint = joints_24.select(1, j);
      const torch::Tensor parent_joint = (j == 0) ? torch::zeros({batch_size, 3}, poses.options())
                                                  : joints_24.select(1, parents[j]);
      const auto rel_joint = joint - parent_joint;

      if (j > 0) {
        transform_mat.narrow(1, 0, 3).narrow(2, 0, 3) = rot_mats_3x3.select(1, j - 1);
      }

      transform_mat.narrow(1, 0, 3).select(2, 3) = rel_joint;

      if (j > 0 && parents[j] >= 0) {
        transform_mat = torch::bmm(transform_mats_list[parents[j]], transform_mat);
      }

      transform_mats_list.push_back(transform_mat);
    }

    const auto transform_mats = torch::stack(transform_mats_list, 1);

    const auto joints_24_expanded = joints_24.unsqueeze(2).unsqueeze(3);
    const auto rot_parts = transform_mats.narrow(2, 0, 3).narrow(3, 0, 3);
    const auto trans_parts = transform_mats.narrow(2, 0, 3).select(3, 3);

    const auto new_trans =
        trans_parts - torch::matmul(rot_parts, joints_24.unsqueeze(3)).squeeze(3);

    auto rel_transform_mats = torch::eye(4, poses.options())
                                  .unsqueeze(0)
                                  .unsqueeze(0)
                                  .expand({batch_size, 24, 4, 4})
                                  .clone();
    rel_transform_mats.narrow(2, 0, 3).narrow(3, 0, 3) = rot_parts;
    rel_transform_mats.narrow(2, 0, 3).select(3, 3) = new_trans;

    const auto weights_expanded = weights.unsqueeze(0).unsqueeze(3).unsqueeze(4);
    const auto rel_mats_expanded = rel_transform_mats.unsqueeze(1);
    const auto blended_mats = (weights_expanded * rel_mats_expanded).sum(2);

    const auto v_posed_homo =
        torch::cat({v_posed, torch::ones({batch_size, 6890, 1}, poses.options())}, 2);
    const auto verts_homo = torch::bmm(blended_mats.view({-1, 4, 4}), v_posed_homo.view({-1, 4, 1}))
                                .view({batch_size, 6890, 4});
    const auto verts = verts_homo.narrow(2, 0, 3);

    return verts;
  }

  torch::Tensor apply_global_transform(torch::Tensor joints, const torch::Tensor& rh,
                                       const torch::Tensor& th) {
    const auto batch_size = joints.size(0);

    if (rh.defined() && rh.numel() > 0) {
      const auto R = batch_rodrigues(rh);
      joints = torch::bmm(joints, R.transpose(1, 2));
    }

    if (th.defined() && th.numel() > 0) {
      const auto th_reshaped = th.unsqueeze(1).expand({batch_size, joints.size(1), 3});
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
    for (const auto s : posedirs_shape) posedirs_torch_shape.push_back(static_cast<int64_t>(s));
    posedirs = torch::from_blob(posedirs_data.data(), posedirs_torch_shape, torch::kFloat64)
                   .clone()
                   .to(torch::kFloat32);

    auto [shapedirs_data, shapedirs_shape] =
        load_tensor<double>((param_dir / "SMPL_NEUTRAL_shapedirs.json").generic_string());
    std::vector<int64_t> shapedirs_torch_shape;
    for (const auto s : shapedirs_shape) shapedirs_torch_shape.push_back(static_cast<int64_t>(s));
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
    if (betas.defined()) betas = betas.to(v_template.device()).to(v_template.dtype());
    if (poses.defined()) poses = poses.to(v_template.device()).to(v_template.dtype());
    if (rh.defined()) rh = rh.to(v_template.device()).to(v_template.dtype());
    if (th.defined()) th = th.to(v_template.device()).to(v_template.dtype());

    const auto batch_size = poses.size(0);

    auto v_shaped = apply_shape_blend(betas);
    v_shaped = apply_pose_blend(v_shaped, poses);
    auto verts = compute_lbs(v_shaped, poses);

    const auto j_reg_expanded = j_regressor_body25.unsqueeze(0).expand({batch_size, -1, -1});
    auto joints = torch::bmm(j_reg_expanded, verts);

    joints = apply_global_transform(joints, rh, th);

    return joints;
  }
};

torch::Tensor compute_limb_length_loss(const torch::Tensor& pred_keypoints,
                                       const std::vector<float>& target_keypoints) {
  const std::vector<std::pair<int, int>> kintree = {
      {8, 1}, {2, 5}, {2, 3}, {5, 6}, {3, 4}, {6, 7},  {2, 3},  {5, 6},   {3, 4},   {6, 7},
      {2, 3}, {5, 6}, {3, 4}, {6, 7}, {1, 0}, {9, 12}, {9, 10}, {10, 11}, {12, 13}, {13, 14}};

  const auto batch_size = pred_keypoints.size(0);
  const auto num_keypoints = static_cast<int64_t>(target_keypoints.size() / (batch_size * 4));

  const auto target_tensor = torch::from_blob(const_cast<float*>(target_keypoints.data()),
                                              {batch_size, num_keypoints, 4}, torch::kFloat32)
                                 .clone();

  std::vector<torch::Tensor> pred_lengths_list;
  std::vector<torch::Tensor> target_lengths_list;
  std::vector<torch::Tensor> confidence_list;

  for (size_t i = 0; i < kintree.size(); i++) {
    const auto idx1 = kintree[i].first;
    const auto idx2 = kintree[i].second;

    const auto v1_pred = pred_keypoints.select(1, idx1);
    const auto v2_pred = pred_keypoints.select(1, idx2);
    const auto diff_pred = v2_pred - v1_pred;
    const auto length_pred = torch::norm(diff_pred, 2, -1);
    pred_lengths_list.push_back(length_pred);

    const auto v1_target = target_tensor.select(1, idx1);
    const auto v2_target = target_tensor.select(1, idx2);
    const auto diff_target = v2_target - v1_target;
    const auto length_target = torch::norm(diff_target, 2, -1);
    target_lengths_list.push_back(length_target);

    const auto conf1 = v1_target.select(1, 3);
    const auto conf2 = v2_target.select(1, 3);
    const auto conf = torch::min(conf1, conf2);
    confidence_list.push_back(conf);
  }

  const auto pred_lengths = torch::stack(pred_lengths_list, 1);
  const auto target_lengths = torch::stack(target_lengths_list, 1);
  const auto confidence = torch::stack(confidence_list, 1);

  const auto diff = pred_lengths - target_lengths;
  const auto squared_diff = diff * diff;
  const auto weighted_error = squared_diff * confidence;
  const auto num = weighted_error.sum();
  const auto denom = confidence.sum();
  const auto loss = num / (denom + 1e-5f);

  return loss;
}

torch::Tensor compute_smooth_loss(const torch::Tensor& values,
                                  const std::vector<float>& window_heights = {0.5f, 0.3f, 0.1f,
                                                                              0.1f},
                                  bool order2 = true) {
  const auto num_frames = values.size(0);
  const auto num_items = values.size(1);

  torch::Tensor total_loss = torch::zeros({}, values.options());

  for (size_t k = 0; k < window_heights.size(); k++) {
    torch::Tensor sq_sum = torch::zeros({}, values.options());

    if (order2) {
      if (num_frames < static_cast<int>(k + 3)) continue;

      for (int b = 0; b < num_frames - static_cast<int>(k + 2); b++) {
        const auto v0 = values[b];
        const auto v1 = values[b + k + 1];
        const auto v2 = values[b + k + 2];
        const auto v_next = values[b + 1];

        const auto d1 = v1 - v0;
        const auto d2 = v2 - v_next;
        const auto d = d2 - d1;
        sq_sum += (d * d).sum();
      }
      sq_sum /= ((num_frames - static_cast<int>(k + 2)) * num_items);
    } else {
      if (num_frames < static_cast<int>(k + 2)) continue;

      for (int b = 0; b < num_frames - static_cast<int>(k + 1); b++) {
        const auto d = values[b + k + 1] - values[b];
        sq_sum += (d * d).sum();
      }
      sq_sum /= ((num_frames - static_cast<int>(k + 1)) * num_items);
    }

    total_loss += sq_sum * window_heights[k];
  }

  return total_loss;
}

torch::Tensor compute_keypoints_loss(const torch::Tensor& pred_keypoints,
                                     const torch::Tensor& target_keypoints,
                                     const std::vector<int>& indices = {}) {
  const auto confidence = target_keypoints.select(2, 3);
  const auto target_xyz = target_keypoints.narrow(2, 0, 3);

  torch::Tensor num;
  torch::Tensor denom;

  if (indices.empty()) {
    const auto diff = pred_keypoints - target_xyz;
    const auto sq_dist = (diff * diff).sum(2);
    const auto weighted = sq_dist * confidence;
    num = weighted.sum();
    denom = confidence.sum();
  } else {
    num = torch::zeros({}, torch::kFloat32);
    denom = torch::zeros({}, torch::kFloat32);

    for (const auto idx : indices) {
      const auto pred_pt = pred_keypoints.select(1, idx);
      const auto target_pt = target_xyz.select(1, idx);
      const auto conf = confidence.select(1, idx);

      const auto diff = target_pt - pred_pt;
      const auto sq_dist = (diff * diff).sum(1);
      const auto weighted = sq_dist * conf;
      num += weighted.sum();
      denom += conf.sum();
    }
  }

  return num / (denom + 1e-5f);
}

int main() {
  std::cout << "Starting optimization with LibTorch..." << std::endl;
  auto total_start = std::chrono::high_resolution_clock::now();

  auto keypoints3d = load_tensor_as_torch("../data/opt/observations_keypoints3d.json");
  std::cout << "keypoints3d loaded shape: " << keypoints3d.sizes() << std::endl;
  auto poses = load_tensor_as_torch("../data/opt/params_poses.json");
  auto shapes = load_tensor_as_torch("../data/opt/params_shapes.json");
  auto rh = load_tensor_as_torch("../data/opt/params_Rh.json");
  auto th = load_tensor_as_torch("../data/opt/params_Th.json");

  SMPLModel model;

  std::cout << "\n=== Phase 1: Fitting shape ===" << std::endl;
  auto phase_start = std::chrono::high_resolution_clock::now();
  {
    auto shapes_opt = shapes.clone().detach().requires_grad_(true);
    torch::optim::Adam optimizer({shapes_opt}, torch::optim::AdamOptions(0.05));

    auto keypoints_vec = std::vector<float>(keypoints3d.data_ptr<float>(),
                                            keypoints3d.data_ptr<float>() + keypoints3d.numel());

    float prev_loss = std::numeric_limits<float>::max();
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    float current_lr = 0.05f;

    for (int i = 0; i < 1000; i++) {
      optimizer.zero_grad();

      auto all_pred_keypoints = model.forward(shapes_opt, poses, rh, th);

      auto limb_loss = compute_limb_length_loss(all_pred_keypoints, keypoints_vec);
      auto reg_loss = (shapes_opt * shapes_opt).sum() / shapes_opt.size(0);

      auto loss = limb_loss * 100.0f + reg_loss * 0.1f;

      loss.backward();
      optimizer.step();

      float current_loss = loss.item<float>();
      if (i == 0 || i == 999) {
        std::cout << "Iteration " << i << ": loss = " << current_loss << ", lr = " << current_lr
                  << std::endl;
      }

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
    auto rh_opt = rh.clone().requires_grad_(true);
    auto th_opt = th.clone().requires_grad_(true);
    torch::optim::Adam optimizer({rh_opt, th_opt}, torch::optim::AdamOptions(0.05));

    const std::vector<int> phase2_indices = {2, 5, 9, 12};

    float prev_loss = std::numeric_limits<float>::max();
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    float current_lr = 0.05f;

    for (int iteration = 0; iteration < 1000; iteration++) {
      optimizer.zero_grad();

      auto all_pred_keypoints = model.forward(shapes, poses, rh_opt, th_opt);

      auto keypoints3d_loss =
          compute_keypoints_loss(all_pred_keypoints, keypoints3d, phase2_indices);

      auto smooth_keypoints_loss =
          compute_smooth_loss(all_pred_keypoints, {0.5f, 0.3f, 0.1f, 0.1f});
      auto th_reshaped = th_opt.unsqueeze(1);
      auto smooth_th_loss = compute_smooth_loss(th_reshaped, {0.5f, 0.3f, 0.1f, 0.1f});

      auto loss = keypoints3d_loss * 100.0f +
                  (smooth_keypoints_loss * 10.0f + smooth_th_loss * 100.0f) * 1.0f;

      loss.backward();
      optimizer.step();

      float current_loss = loss.item<float>();
      if (iteration == 0 || iteration == 999) {
        std::cout << "Iteration " << iteration << ": loss = " << current_loss
                  << ", lr = " << current_lr << std::endl;
      }

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
    PriorLoss prior_loss;

    auto poses_opt = poses.clone().requires_grad_(true);
    auto rh_opt = rh.clone().requires_grad_(true);
    auto th_opt = th.clone().requires_grad_(true);
    torch::optim::Adam optimizer({poses_opt, rh_opt, th_opt}, torch::optim::AdamOptions(0.02));

    float prev_loss = std::numeric_limits<float>::max();
    float best_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    float current_lr = 0.02f;

    for (int i = 0; i < 1000; i++) {
      optimizer.zero_grad();

      auto all_pred_keypoints = model.forward(shapes, poses_opt, rh_opt, th_opt);

      auto keypoints3d_loss = compute_keypoints_loss(all_pred_keypoints, keypoints3d);

      auto poses_reshaped = poses_opt.unsqueeze(1);
      auto smooth_poses_loss = compute_smooth_loss(poses_reshaped, {0.5f, 0.3f, 0.1f, 0.1f});

      auto smooth_keypoints_loss =
          compute_smooth_loss(all_pred_keypoints, {0.5f, 0.3f, 0.1f, 0.1f});

      auto th_reshaped = th_opt.unsqueeze(1);
      auto smooth_th_loss = compute_smooth_loss(th_reshaped, {0.5f, 0.3f, 0.1f, 0.1f});

      auto prior_loss_value = prior_loss.compute(poses_opt);

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
