#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../src/optimization/tensor_wrapper.hpp"

using namespace marionette::optimization;

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

Tensor load_tensor_as_tensor(const std::string& file_name) {
  auto [data, shape] = load_tensor<float>(file_name);

  std::vector<int64_t> shape_i64;
  for (auto s : shape) {
    shape_i64.push_back(static_cast<int64_t>(s));
  }

  return Tensor::from_blob(data.data(), shape_i64);
}

Tensor load_dense_matrix_as_tensor(const std::string& file_name) {
  std::ifstream ifs;
  ifs.open(file_name, std::ios::in);
  nlohmann::json j = nlohmann::json::parse(ifs);

  const auto type_str = j["type"].get<std::string>();
  const auto data_vec = j["data"];
  const auto shape = j["shape"].get<std::vector<uint32_t>>();

  if (shape.size() != 2) {
    throw std::runtime_error("Invalid shape");
  }

  Tensor tensor;
  if (type_str == "float64") {
    const auto data = data_vec.get<std::vector<double>>();
    tensor = Tensor::from_blob(data.data(),
                               {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])});
  } else if (type_str == "float32") {
    const auto data = data_vec.get<std::vector<float>>();
    tensor = Tensor::from_blob(data.data(),
                               {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])});
  } else {
    throw std::runtime_error("Unsupported type: " + type_str);
  }

  return tensor.to_float32();
}

Tensor load_sparse_matrix_as_tensor(const std::string& file_name) {
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

  Tensor dense = Tensor::zeros({static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1])});

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
  Tensor gmm_means;
  Tensor gmm_precisions;
  Tensor nll_weights;

  PriorLoss() {
    using path = std::filesystem::path;
    path param_dir("../data/opt/");

    auto [means_data, means_shape] =
        load_tensor<double>((param_dir / "gmm_means.json").generic_string());
    gmm_means = Tensor::from_blob(means_data.data(), {static_cast<int64_t>(means_shape[0]),
                                                      static_cast<int64_t>(means_shape[1])})
                    .to_float32();

    auto [covs_data, covs_shape] =
        load_tensor<double>((param_dir / "gmm_covars.json").generic_string());
    const auto covs = Tensor::from_blob(
        covs_data.data(), {static_cast<int64_t>(covs_shape[0]), static_cast<int64_t>(covs_shape[1]),
                           static_cast<int64_t>(covs_shape[2])});

    auto [weights_data, weights_shape] =
        load_tensor<double>((param_dir / "gmm_weights.json").generic_string());
    const auto gmm_weights =
        Tensor::from_blob(weights_data.data(), {static_cast<int64_t>(weights_shape[0])});

    const auto num_gaussians = covs.size(0);
    std::vector<Tensor> precisions_list;
    std::vector<double> sqrdets;

    for (int m = 0; m < num_gaussians; m++) {
      const auto cov = covs.select(0, m).to_float64();
      const auto det = Tensor::det(cov).item_double();
      const auto sqrdet = std::sqrt(det);
      sqrdets.push_back(sqrdet);

      const auto precision = Tensor::inverse(cov);
      precisions_list.push_back(precision);
    }

    gmm_precisions = Tensor::stack(precisions_list, 0).to_float32();

    constexpr double pi = 3.141592653589793;
    constexpr int pose_dim = 69;  // 23 joints * 3 dimensions
    const double c = std::pow(2.0 * pi, static_cast<double>(pose_dim) / 2.0);
    const auto min_sqrdet = *std::min_element(sqrdets.begin(), sqrdets.end());

    std::vector<float> nll_weights_vec;
    for (int m = 0; m < num_gaussians; m++) {
      const auto w = gmm_weights.select(0, m).item_double();
      const auto nll = -std::log(w / (c * (sqrdets[m] / min_sqrdet)));
      nll_weights_vec.push_back(static_cast<float>(nll));
    }

    nll_weights = Tensor::from_blob(nll_weights_vec.data(), {num_gaussians});
  }

  Tensor compute(const Tensor& poses) {
    const auto poses_expanded = poses.unsqueeze(1);
    const auto means_expanded = gmm_means.unsqueeze(0);
    const auto d = poses_expanded - means_expanded;

    const auto d_expanded = d.unsqueeze(2);
    const auto prec_expanded = gmm_precisions.unsqueeze(0);
    const auto prec_d = Tensor::matmul(d_expanded, prec_expanded).squeeze(2);

    const auto prec_dd = (d * prec_d).sum(2);

    const auto nll_expanded = nll_weights.unsqueeze(0);
    constexpr float mahalanobis_scale = 0.5f;
    const auto loglikelihood = mahalanobis_scale * prec_dd + nll_expanded;

    const auto min_likelihood = loglikelihood.min(1);
    return min_likelihood.mean();
  }
};

class SMPLModel {
 public:
  Tensor v_template;
  Tensor weights;
  Tensor j_regressor;
  Tensor j_regressor_body25;
  Tensor shapedirs;
  Tensor posedirs;
  std::vector<int32_t> parents;

 private:
  Tensor batch_rodrigues(const Tensor& pose_vec) {
    const auto pose_reshaped = pose_vec.dim() == 1 ? pose_vec.unsqueeze(0) : pose_vec;

    const auto theta2 = (pose_reshaped * pose_reshaped).sum(1, true);
    constexpr float epsilon = 1e-8f;
    const auto theta = (theta2 + epsilon).sqrt();
    const auto w = pose_reshaped / (theta + epsilon);

    const auto wx = w.select(1, 0).unsqueeze(1);
    const auto wy = w.select(1, 1).unsqueeze(1);
    const auto wz = w.select(1, 2).unsqueeze(1);

    const auto cos_theta = theta.cos();
    const auto sin_theta = theta.sin();
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

    std::vector<Tensor> row0_vec;
    row0_vec.push_back(r00);
    row0_vec.push_back(r01);
    row0_vec.push_back(r02);
    const auto row0 = Tensor::cat(row0_vec, 1).unsqueeze(1);

    std::vector<Tensor> row1_vec;
    row1_vec.push_back(r10);
    row1_vec.push_back(r11);
    row1_vec.push_back(r12);
    const auto row1 = Tensor::cat(row1_vec, 1).unsqueeze(1);

    std::vector<Tensor> row2_vec;
    row2_vec.push_back(r20);
    row2_vec.push_back(r21);
    row2_vec.push_back(r22);
    const auto row2 = Tensor::cat(row2_vec, 1).unsqueeze(1);

    std::vector<Tensor> rows_vec;
    rows_vec.push_back(row0);
    rows_vec.push_back(row1);
    rows_vec.push_back(row2);
    return Tensor::cat(rows_vec, 1);
  }

  Tensor apply_shape_blend(const Tensor& betas) {
    auto v_shaped = v_template.clone();

    if (betas.defined() && betas.numel() > 0) {
      const auto shape_disps = Tensor::matmul(shapedirs, betas.squeeze(0));
      v_shaped = v_shaped + shape_disps;
    }

    return v_shaped.unsqueeze(0);
  }

  Tensor apply_pose_blend(const Tensor& v_shaped, const Tensor& poses) {
    if (!poses.defined() || poses.numel() == 0) {
      return v_shaped;
    }

    const auto batch_size = poses.size(0);
    constexpr int num_body_joints = 23;  // Number of body joints (excluding root)
    constexpr int axis_angle_dim = 3;    // Axis-angle representation dimension
    constexpr int rot_mat_dim = 9;       // Flattened 3x3 rotation matrix dimension
    const auto poses_reshaped = poses.view({batch_size, num_body_joints, axis_angle_dim});

    const auto rot_mats_3x3 = batch_rodrigues(poses_reshaped.view({-1, axis_angle_dim}));
    const auto rot_mats = rot_mats_3x3.view({batch_size, num_body_joints, rot_mat_dim});

    // Pose feature: rotation matrices relative to identity
    auto rot_mats_feat = rot_mats.clone();
    rot_mats_feat.select(2, 0) -= 1.0f;
    rot_mats_feat.select(2, 4) -= 1.0f;
    rot_mats_feat.select(2, 8) -= 1.0f;

    const auto pose_feature_flat = rot_mats_feat.view({batch_size, -1});
    constexpr int pose_blend_dim = num_body_joints * rot_mat_dim;  // 23 * 9 = 207
    const auto posedirs_2d = posedirs.view({-1, pose_blend_dim});
    const auto num_vertices = v_shaped.size(1);  // 6890 vertices in SMPL model
    const auto pose_offset = Tensor::matmul(posedirs_2d, pose_feature_flat.t())
                                 .t()
                                 .view({batch_size, num_vertices, axis_angle_dim});

    const auto v_shaped_expanded = v_shaped.expand({batch_size, -1, -1});
    return v_shaped_expanded + pose_offset;
  }

  Tensor compute_lbs(const Tensor& v_posed, const Tensor& poses) {
    if (!poses.defined() || poses.numel() == 0) {
      return v_posed;
    }

    const auto batch_size = poses.size(0);

    constexpr int num_body_joints = 23;    // Number of body joints (excluding root)
    constexpr int num_total_joints = 24;   // Total number of joints (including root)
    constexpr int xyz_dim = 3;             // 3D coordinate dimension
    constexpr int transform_mat_size = 4;  // 4x4 transformation matrix

    const auto j_regressor_expanded = j_regressor.unsqueeze(0).expand({batch_size, -1, -1});
    const auto joints_24 = Tensor::bmm(j_regressor_expanded, v_posed);

    const auto poses_reshaped = poses.view({batch_size, num_body_joints, xyz_dim});
    auto rot_mats_3x3 = batch_rodrigues(poses_reshaped.view({-1, xyz_dim}));
    rot_mats_3x3 = rot_mats_3x3.view({batch_size, num_body_joints, xyz_dim, xyz_dim});

    // Build 4x4 transformation matrices for kinematic chain
    std::vector<Tensor> transform_mats_list;
    for (int j = 0; j < num_total_joints; j++) {
      auto transform_mat = Tensor::eye(transform_mat_size, poses.options())
                               .unsqueeze(0)
                               .expand({batch_size, transform_mat_size, transform_mat_size})
                               .clone();

      const auto joint = joints_24.select(1, j);
      const Tensor parent_joint = (j == 0) ? Tensor::zeros({batch_size, xyz_dim}, poses.options())
                                           : joints_24.select(1, parents[j]);
      const auto rel_joint = joint - parent_joint;

      if (j > 0) {
        transform_mat.narrow(1, 0, xyz_dim).narrow(2, 0, xyz_dim) = rot_mats_3x3.select(1, j - 1);
      }

      transform_mat.narrow(1, 0, xyz_dim).select(2, xyz_dim) = rel_joint;

      if (j > 0 && parents[j] >= 0) {
        transform_mat = Tensor::bmm(transform_mats_list[parents[j]], transform_mat);
      }

      transform_mats_list.push_back(transform_mat);
    }

    const auto transform_mats = Tensor::stack(transform_mats_list, 1);

    const auto joints_24_expanded = joints_24.unsqueeze(2).unsqueeze(3);
    const auto rot_parts = transform_mats.narrow(2, 0, xyz_dim).narrow(3, 0, xyz_dim);
    const auto trans_parts = transform_mats.narrow(2, 0, xyz_dim).select(3, xyz_dim);

    const auto new_trans =
        trans_parts - Tensor::matmul(rot_parts, joints_24.unsqueeze(3)).squeeze(3);

    auto rel_transform_mats =
        Tensor::eye(transform_mat_size, poses.options())
            .unsqueeze(0)
            .unsqueeze(0)
            .expand({batch_size, num_total_joints, transform_mat_size, transform_mat_size})
            .clone();
    rel_transform_mats.narrow(2, 0, xyz_dim).narrow(3, 0, xyz_dim) = rot_parts;
    rel_transform_mats.narrow(2, 0, xyz_dim).select(3, xyz_dim) = new_trans;

    const auto weights_expanded = weights.unsqueeze(0).unsqueeze(3).unsqueeze(4);
    const auto rel_mats_expanded = rel_transform_mats.unsqueeze(1);
    const auto blended_mats = (weights_expanded * rel_mats_expanded).sum(2);

    const auto num_vertices = v_posed.size(1);  // 6890 vertices in SMPL model
    std::vector<Tensor> cat_list;
    cat_list.push_back(v_posed);
    cat_list.push_back(Tensor::ones({batch_size, num_vertices, 1}, poses.options()));
    const auto v_posed_homo = Tensor::cat(cat_list, 2);
    const auto verts_homo =
        Tensor::bmm(blended_mats.view({-1, transform_mat_size, transform_mat_size}),
                    v_posed_homo.view({-1, transform_mat_size, 1}))
            .view({batch_size, num_vertices, transform_mat_size});
    const auto verts = verts_homo.narrow(2, 0, xyz_dim);

    return verts;
  }

  Tensor apply_global_transform(Tensor joints, const Tensor& rh, const Tensor& th) {
    const auto batch_size = joints.size(0);

    if (rh.defined() && rh.numel() > 0) {
      const auto R = batch_rodrigues(rh);
      joints = Tensor::bmm(joints, R.transpose(1, 2));
    }

    if (th.defined() && th.numel() > 0) {
      constexpr int xyz_dim = 3;  // 3D coordinate dimension
      const auto th_reshaped = th.unsqueeze(1).expand({batch_size, joints.size(1), xyz_dim});
      joints = joints + th_reshaped;
    }

    return joints;
  }

 public:
  SMPLModel() {
    using path = std::filesystem::path;
    path param_dir("../data/opt/");

    v_template =
        load_dense_matrix_as_tensor((param_dir / "SMPL_NEUTRAL_v_template.json").generic_string());
    weights =
        load_dense_matrix_as_tensor((param_dir / "SMPL_NEUTRAL_weights.json").generic_string());
    j_regressor = load_sparse_matrix_as_tensor(
        (param_dir / "SMPL_NEUTRAL_J_regressor.json").generic_string());
    j_regressor_body25 = load_dense_matrix_as_tensor(
        (param_dir / "SMPL_NEUTRAL_J_regressor_body25.json").generic_string());

    auto [posedirs_data, posedirs_shape] =
        load_tensor<double>((param_dir / "SMPL_NEUTRAL_posedirs.json").generic_string());
    std::vector<int64_t> posedirs_shape_i64;
    for (const auto s : posedirs_shape) posedirs_shape_i64.push_back(static_cast<int64_t>(s));
    posedirs = Tensor::from_blob(posedirs_data.data(), posedirs_shape_i64).to_float32();

    auto [shapedirs_data, shapedirs_shape] =
        load_tensor<double>((param_dir / "SMPL_NEUTRAL_shapedirs.json").generic_string());
    std::vector<int64_t> shapedirs_shape_i64;
    for (const auto s : shapedirs_shape) shapedirs_shape_i64.push_back(static_cast<int64_t>(s));
    shapedirs = Tensor::from_blob(shapedirs_data.data(), shapedirs_shape_i64).to_float32();

    auto [kintree_table_data, kintree_table_shape] =
        load_tensor<uint32_t>((param_dir / "SMPL_NEUTRAL_kintree_table.json").generic_string());
    std::copy_n(kintree_table_data.begin(), kintree_table_shape.back(),
                std::back_inserter(parents));
  }

  Tensor forward(Tensor betas, Tensor poses, Tensor rh, Tensor th) {
    // Type conversions are handled by the current backend
    const auto batch_size = poses.size(0);

    auto v_shaped = apply_shape_blend(betas);
    v_shaped = apply_pose_blend(v_shaped, poses);
    auto verts = compute_lbs(v_shaped, poses);

    const auto j_reg_expanded = j_regressor_body25.unsqueeze(0).expand({batch_size, -1, -1});
    auto joints = Tensor::bmm(j_reg_expanded, verts);

    joints = apply_global_transform(joints, rh, th);

    return joints;
  }
};

Tensor compute_limb_length_loss(const Tensor& pred_keypoints,
                                const std::vector<float>& target_keypoints) {
  const std::vector<std::pair<int, int>> kintree = {
      {8, 1}, {2, 5}, {2, 3}, {5, 6}, {3, 4}, {6, 7},  {2, 3},  {5, 6},   {3, 4},   {6, 7},
      {2, 3}, {5, 6}, {3, 4}, {6, 7}, {1, 0}, {9, 12}, {9, 10}, {10, 11}, {12, 13}, {13, 14}};

  constexpr int xyzc_dim = 4;  // 3D coordinates + confidence
  const auto batch_size = pred_keypoints.size(0);
  const auto num_keypoints =
      static_cast<int64_t>(target_keypoints.size() / (batch_size * xyzc_dim));

  const auto target_tensor =
      Tensor::from_blob(target_keypoints.data(), {batch_size, num_keypoints, xyzc_dim});

  std::vector<Tensor> pred_lengths_list;
  std::vector<Tensor> target_lengths_list;
  std::vector<Tensor> confidence_list;

  for (size_t i = 0; i < kintree.size(); i++) {
    const auto idx1 = kintree[i].first;
    const auto idx2 = kintree[i].second;

    const auto v1_pred = pred_keypoints.select(1, idx1);
    const auto v2_pred = pred_keypoints.select(1, idx2);
    const auto diff_pred = v2_pred - v1_pred;
    const auto length_pred = diff_pred.norm(2, -1);
    pred_lengths_list.push_back(length_pred);

    const auto v1_target = target_tensor.select(1, idx1);
    const auto v2_target = target_tensor.select(1, idx2);
    const auto diff_target = v2_target - v1_target;
    const auto length_target = diff_target.norm(2, -1);
    target_lengths_list.push_back(length_target);

    constexpr int confidence_idx = 3;  // Index of confidence value in xyzc
    const auto conf1 = v1_target.select(1, confidence_idx);
    const auto conf2 = v2_target.select(1, confidence_idx);
    const auto conf = Tensor::min(conf1, conf2);
    confidence_list.push_back(conf);
  }

  const auto pred_lengths = Tensor::stack(pred_lengths_list, 1);
  const auto target_lengths = Tensor::stack(target_lengths_list, 1);
  const auto confidence = Tensor::stack(confidence_list, 1);

  const auto diff = pred_lengths - target_lengths;
  const auto squared_diff = diff * diff;
  const auto weighted_error = squared_diff * confidence;
  const auto num = weighted_error.sum();
  const auto denom = confidence.sum();
  constexpr float small_threshold = 1e-5f;
  const auto loss = num / (denom + small_threshold);

  return loss;
}

Tensor compute_smooth_loss(const Tensor& values,
                           const std::vector<float>& window_heights = {0.5f, 0.3f, 0.1f, 0.1f},
                           bool order2 = true) {
  const auto num_frames = values.size(0);
  const auto num_items = values.size(1);

  Tensor total_loss = Tensor::zeros({}, values.options());

  for (size_t k = 0; k < window_heights.size(); k++) {
    Tensor sq_sum = Tensor::zeros({}, values.options());

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
      sq_sum = sq_sum / ((num_frames - static_cast<int>(k + 2)) * num_items);
    } else {
      if (num_frames < static_cast<int>(k + 2)) continue;

      for (int b = 0; b < num_frames - static_cast<int>(k + 1); b++) {
        const auto d = values[b + k + 1] - values[b];
        sq_sum += (d * d).sum();
      }
      sq_sum = sq_sum / ((num_frames - static_cast<int>(k + 1)) * num_items);
    }

    total_loss += sq_sum * window_heights[k];
  }

  return total_loss;
}

Tensor compute_keypoints_loss(const Tensor& pred_keypoints, const Tensor& target_keypoints,
                              const std::vector<int>& indices = {}) {
  constexpr int xyz_dim = 3;         // 3D coordinate dimension
  constexpr int confidence_idx = 3;  // Index of confidence value
  const auto confidence = target_keypoints.select(2, confidence_idx);
  const auto target_xyz = target_keypoints.narrow(2, 0, xyz_dim);

  Tensor num;
  Tensor denom;

  if (indices.empty()) {
    const auto diff = pred_keypoints - target_xyz;
    const auto sq_dist = (diff * diff).sum(2);
    const auto weighted = sq_dist * confidence;
    num = weighted.sum();
    denom = confidence.sum();
  } else {
    num = Tensor::zeros({});
    denom = Tensor::zeros({});

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

  constexpr float small_threshold = 1e-5f;
  return num / (denom + small_threshold);
}

template <typename OptimizerFunc>
void optimize_loop(Adam& optimizer, OptimizerFunc compute_loss, float initial_lr,
                   int max_iterations, int patience, float convergence_threshold,
                   bool use_relative_change = false) {
  float prev_loss = std::numeric_limits<float>::max();
  float best_loss = std::numeric_limits<float>::max();
  int patience_counter = 0;
  float current_lr = initial_lr;

  for (int i = 0; i < max_iterations; i++) {
    optimizer.zero_grad();

    const auto loss = compute_loss();

    loss.backward();
    optimizer.step();

    const auto current_loss = loss.item_float();
    if (i == 0 || i == max_iterations - 1) {
      std::cout << "Iteration " << i << ": loss = " << current_loss << ", lr = " << current_lr
                << std::endl;
    }

    constexpr float loss_improvement_threshold = 1e-6f;
    if (current_loss < best_loss - loss_improvement_threshold) {
      best_loss = current_loss;
      patience_counter = 0;
    } else {
      patience_counter++;
      constexpr float min_learning_rate = 1e-5f;
      if (patience_counter >= patience && current_lr > min_learning_rate) {
        constexpr float learning_rate_decay = 0.5f;
        current_lr *= learning_rate_decay;
        optimizer.set_learning_rate(current_lr);
        patience_counter = 0;
        std::cout << "  -> Reducing lr to " << current_lr << std::endl;
      }
    }

    constexpr float epsilon = 1e-8f;
    const auto change = use_relative_change
                            ? std::abs(prev_loss - current_loss) / (prev_loss + epsilon)
                            : std::abs(prev_loss - current_loss);
    if (change < convergence_threshold) {
      std::cout << "Converged at iteration " << i << ": loss = " << current_loss << std::endl;
      break;
    }
    prev_loss = current_loss;
  }
}

int main() {
  std::cout << "Starting optimization..." << std::endl;
  auto total_start = std::chrono::high_resolution_clock::now();

  Tensor keypoints3d = load_tensor_as_tensor("../data/opt/observations_keypoints3d.json");
  std::cout << "keypoints3d loaded shape: " << keypoints3d.sizes() << std::endl;
  Tensor poses = load_tensor_as_tensor("../data/opt/params_poses.json");
  Tensor shapes = load_tensor_as_tensor("../data/opt/params_shapes.json");
  Tensor rh = load_tensor_as_tensor("../data/opt/params_Rh.json");
  Tensor th = load_tensor_as_tensor("../data/opt/params_Th.json");

  SMPLModel model;

  std::cout << "\n=== Phase 1: Fitting shape ===" << std::endl;
  auto phase_start = std::chrono::high_resolution_clock::now();
  {
    constexpr float learning_rate = 0.05f;
    constexpr int max_iterations = 1000;
    constexpr int patience = 20;
    constexpr float convergence_threshold = 1e-5f;

    Tensor shapes_opt = shapes.clone().detach().requires_grad_(true);
    AdamOptions adam_options;
    adam_options.lr = learning_rate;
    std::vector<Tensor> params1;
    params1.push_back(shapes_opt);
    Adam optimizer(params1, adam_options);

    const auto keypoints_vec = std::vector<float>(
        keypoints3d.data_ptr<float>(), keypoints3d.data_ptr<float>() + keypoints3d.numel());

    optimize_loop(
        optimizer,
        [&]() {
          const auto all_pred_keypoints = model.forward(shapes_opt, poses, rh, th);
          const auto limb_loss = compute_limb_length_loss(all_pred_keypoints, keypoints_vec);
          const auto reg_loss = (shapes_opt * shapes_opt).sum() / shapes_opt.size(0);
          constexpr float limb_length_weight = 100.0f;
          constexpr float shape_reg_weight = 0.1f;
          return limb_loss * limb_length_weight + reg_loss * shape_reg_weight;
        },
        learning_rate, max_iterations, patience, convergence_threshold);

    shapes = shapes_opt.detach();
  }
  auto phase_end = std::chrono::high_resolution_clock::now();
  auto phase_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);
  std::cout << "Phase 1 completed in " << phase_duration.count() << " ms" << std::endl;

  std::cout << "\n=== Phase 2: Initializing RT ===" << std::endl;
  phase_start = std::chrono::high_resolution_clock::now();
  {
    constexpr float learning_rate = 0.05f;
    constexpr int max_iterations = 1000;
    constexpr int patience = 20;
    constexpr float convergence_threshold = 1e-5f;

    Tensor rh_opt = rh.clone().requires_grad_(true);
    Tensor th_opt = th.clone().requires_grad_(true);
    AdamOptions adam_options;
    adam_options.lr = learning_rate;
    std::vector<Tensor> params2;
    params2.push_back(rh_opt);
    params2.push_back(th_opt);
    Adam optimizer(params2, adam_options);

    const std::vector<int> phase2_indices = {2, 5, 9, 12};  // Shoulder and hip keypoints

    optimize_loop(
        optimizer,
        [&]() {
          const auto all_pred_keypoints = model.forward(shapes, poses, rh_opt, th_opt);
          const auto keypoints3d_loss =
              compute_keypoints_loss(all_pred_keypoints, keypoints3d, phase2_indices);
          const auto smooth_keypoints_loss = compute_smooth_loss(all_pred_keypoints);
          const Tensor th_reshaped = th_opt.unsqueeze(1);
          const auto smooth_th_loss = compute_smooth_loss(th_reshaped);
          constexpr float keypoints3d_weight = 100.0f;
          constexpr float smooth_keypoints_weight = 10.0f;
          constexpr float smooth_th_weight = 100.0f;
          constexpr float smooth_loss_weight = 1.0f;
          return keypoints3d_loss * keypoints3d_weight +
                 (smooth_keypoints_loss * smooth_keypoints_weight +
                  smooth_th_loss * smooth_th_weight) *
                     smooth_loss_weight;
        },
        learning_rate, max_iterations, patience, convergence_threshold);

    rh = rh_opt.detach();
    th = th_opt.detach();
  }
  phase_end = std::chrono::high_resolution_clock::now();
  phase_duration = std::chrono::duration_cast<std::chrono::milliseconds>(phase_end - phase_start);
  std::cout << "Phase 2 completed in " << phase_duration.count() << " ms" << std::endl;

  std::cout << "\n=== Phase 3: Refining pose ===" << std::endl;
  phase_start = std::chrono::high_resolution_clock::now();
  {
    constexpr float learning_rate = 0.02f;
    constexpr int max_iterations = 1000;
    constexpr int patience = 30;
    constexpr float convergence_threshold = 1e-5f;
    constexpr bool use_relative_change = true;

    PriorLoss prior_loss;

    Tensor poses_opt = poses.clone().requires_grad_(true);
    Tensor rh_opt = rh.clone().requires_grad_(true);
    Tensor th_opt = th.clone().requires_grad_(true);
    AdamOptions adam_options;
    adam_options.lr = learning_rate;
    std::vector<Tensor> params3;
    params3.push_back(poses_opt);
    params3.push_back(rh_opt);
    params3.push_back(th_opt);
    Adam optimizer(params3, adam_options);

    optimize_loop(
        optimizer,
        [&]() {
          const auto all_pred_keypoints = model.forward(shapes, poses_opt, rh_opt, th_opt);
          const auto keypoints3d_loss = compute_keypoints_loss(all_pred_keypoints, keypoints3d);

          const Tensor poses_reshaped = poses_opt.unsqueeze(1);
          const auto smooth_poses_loss = compute_smooth_loss(poses_reshaped);
          const auto smooth_keypoints_loss = compute_smooth_loss(all_pred_keypoints);
          const Tensor th_reshaped = th_opt.unsqueeze(1);
          const auto smooth_th_loss = compute_smooth_loss(th_reshaped);
          const auto prior_loss_value = prior_loss.compute(poses_opt);

          constexpr float keypoints3d_weight = 1000.0f;
          constexpr float smooth_poses_weight = 100.0f;
          constexpr float smooth_keypoints_weight = 10.0f;
          constexpr float smooth_th_weight = 10.0f;
          constexpr float smooth_loss_weight = 1.0f;
          constexpr float prior_weight = 0.1f;

          const auto smooth_loss = smooth_poses_loss * smooth_poses_weight +
                                   smooth_keypoints_loss * smooth_keypoints_weight +
                                   smooth_th_loss * smooth_th_weight;
          return keypoints3d_loss * keypoints3d_weight + smooth_loss * smooth_loss_weight +
                 prior_loss_value * prior_weight;
        },
        learning_rate, max_iterations, patience, convergence_threshold, use_relative_change);

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
