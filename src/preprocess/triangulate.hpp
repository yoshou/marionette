#pragma once

#include <Eigen/Dense>
#include <vector>

namespace marionette {
namespace preprocess {

class triangulator_t {
 public:
  static std::vector<Eigen::Vector4d> triangulate_keypoints(
      const std::vector<std::vector<Eigen::Vector3d>>& keypoints2d,
      const std::vector<Eigen::Matrix<double, 3, 4>>& projection_matrices, int min_views = 2) {
    if (keypoints2d.empty() || projection_matrices.empty()) {
      return {};
    }

    size_t n_views = keypoints2d.size();
    size_t n_joints = keypoints2d[0].size();

    std::vector<Eigen::Vector4d> keypoints3d(n_joints, Eigen::Vector4d::Zero());

    for (size_t joint_idx = 0; joint_idx < n_joints; ++joint_idx) {
      std::vector<size_t> valid_views;
      double total_conf = 0.0;

      for (size_t view_idx = 0; view_idx < n_views; ++view_idx) {
        if (keypoints2d[view_idx][joint_idx](2) > 0.0) {
          valid_views.push_back(view_idx);
          total_conf += keypoints2d[view_idx][joint_idx](2);
        }
      }

      if (valid_views.size() < static_cast<size_t>(min_views)) {
        continue;
      }

      Eigen::MatrixXd A(2 * valid_views.size(), 4);
      Eigen::VectorXd weights(2 * valid_views.size());

      for (size_t i = 0; i < valid_views.size(); ++i) {
        size_t view_idx = valid_views[i];
        const auto& kpt = keypoints2d[view_idx][joint_idx];
        const auto& P = projection_matrices[view_idx];

        double x = kpt(0);
        double y = kpt(1);
        double conf = kpt(2);

        A.row(2 * i + 0) = x * P.row(2) - P.row(0);
        weights(2 * i + 0) = conf;

        A.row(2 * i + 1) = y * P.row(2) - P.row(1);
        weights(2 * i + 1) = conf;
      }

      for (int i = 0; i < A.rows(); ++i) {
        A.row(i) *= weights(i);
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
      Eigen::Vector4d X = svd.matrixV().col(3);
      X /= X(3);

      keypoints3d[joint_idx].head<3>() = X.head<3>();
      keypoints3d[joint_idx](3) = total_conf / valid_views.size();
    }

    return keypoints3d;
  }

  static Eigen::Vector3d project_point(const Eigen::Matrix<double, 3, 4>& P,
                                       const Eigen::Vector3d& X) {
    Eigen::Vector4d Xh;
    Xh.head<3>() = X;
    Xh(3) = 1.0;
    Eigen::Vector3d proj = P * Xh;
    proj /= proj(2);
    return proj;
  }

  static bool triangulate_point_dlt(const std::vector<Eigen::Vector3d>& kpts2d,
                                    const std::vector<Eigen::Matrix<double, 3, 4>>& Pall,
                                    const std::vector<int>& view_indices, Eigen::Vector4d& out_X) {
    if (view_indices.size() < 2) {
      return false;
    }

    Eigen::MatrixXd A(2 * static_cast<int>(view_indices.size()), 4);
    Eigen::VectorXd weights(2 * static_cast<int>(view_indices.size()));

    for (size_t i = 0; i < view_indices.size(); ++i) {
      const int v = view_indices[i];
      const auto& P = Pall[v];
      const auto& kpt = kpts2d[v];
      const double x = kpt(0);
      const double y = kpt(1);
      const double conf = kpt(2);

      A.row(2 * i + 0) = x * P.row(2) - P.row(0);
      A.row(2 * i + 1) = y * P.row(2) - P.row(1);
      weights(2 * i + 0) = conf;
      weights(2 * i + 1) = conf;
    }

    for (int r = 0; r < A.rows(); ++r) {
      A.row(r) *= weights(r);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);
    if (std::abs(X(3)) < 1e-12) {
      return false;
    }
    X /= X(3);
    out_X = X;
    return true;
  }

  static void combinations_k_of_n(int n, int k, std::vector<std::vector<int>>& out) {
    out.clear();
    if (k <= 0 || n < k) return;
    std::vector<int> comb(k);
    for (int i = 0; i < k; ++i) comb[i] = i;
    while (true) {
      out.push_back(comb);
      int i = k - 1;
      for (; i >= 0; --i) {
        if (comb[i] != i + n - k) break;
      }
      if (i < 0) break;
      ++comb[i];
      for (int j = i + 1; j < k; ++j) {
        comb[j] = comb[j - 1] + 1;
      }
    }
  }

  static std::vector<int> robust_select_views_for_joint(
      const std::vector<Eigen::Vector3d>& kpts2d_by_view,
      const std::vector<Eigen::Matrix<double, 3, 4>>& Pall, const std::vector<int>& valid_views,
      double dist_max, int min_views) {
    if (static_cast<int>(valid_views.size()) < min_views) {
      return {};
    }

    const int nV = static_cast<int>(valid_views.size());
    const int k = std::min(min_views, nV);

    std::vector<std::vector<int>> combs_local;
    combinations_k_of_n(nV, k, combs_local);
    if (combs_local.empty()) {
      return {};
    }

    int best_idx = -1;
    double best_score = -1e18;
    std::vector<std::vector<int>> combs_views;
    combs_views.reserve(combs_local.size());
    for (const auto& c : combs_local) {
      std::vector<int> views;
      views.reserve(c.size());
      for (int id : c) views.push_back(valid_views[id]);
      combs_views.push_back(std::move(views));
    }

    for (size_t ci = 0; ci < combs_views.size(); ++ci) {
      const auto& comb_views = combs_views[ci];
      Eigen::Vector4d X;
      if (!triangulate_point_dlt(kpts2d_by_view, Pall, comb_views, X)) {
        continue;
      }
      if (X(3) <= 0.0) {
        continue;
      }
      const Eigen::Vector3d X3 = X.head<3>();

      double score = 0.0;
      double self_bonus = 0.0;
      for (int v : valid_views) {
        const auto proj = project_point(Pall[v], X3);
        const double conf = kpts2d_by_view[v](2);
        if (conf <= 0.0) continue;
        const double err = (proj.head<2>() - kpts2d_by_view[v].head<2>()).norm();
        double valid = 1.0 - err / dist_max;
        if (valid < 0.0) valid = 0.0;
        score += conf * valid;
      }
      for (int v : comb_views) {
        const auto proj = project_point(Pall[v], X3);
        const double conf = kpts2d_by_view[v](2);
        if (conf <= 0.0) continue;
        const double err = (proj.head<2>() - kpts2d_by_view[v].head<2>()).norm();
        if (err <= dist_max) {
          self_bonus += 100.0;
        }
      }
      score += self_bonus - static_cast<double>(min_views) * 100.0;
      if (score > best_score) {
        best_score = score;
        best_idx = static_cast<int>(ci);
      }
    }

    if (best_idx < 0 || best_score < 0.0) {
      return {};
    }

    std::vector<int> selected = combs_views[best_idx];
    {
      Eigen::Vector4d X;
      if (!triangulate_point_dlt(kpts2d_by_view, Pall, selected, X)) {
        return {};
      }
      if (X(3) <= 0.0) {
        return {};
      }
      const Eigen::Vector3d X3 = X.head<3>();
      for (int v : selected) {
        const auto proj = project_point(Pall[v], X3);
        const double err = (proj.head<2>() - kpts2d_by_view[v].head<2>()).norm();
        if (err > dist_max) {
          return {};
        }
      }
    }

    std::vector<int> candidates;
    candidates.reserve(valid_views.size());
    {
      Eigen::Vector4d X;
      triangulate_point_dlt(kpts2d_by_view, Pall, selected, X);
      const Eigen::Vector3d X3 = X.head<3>();
      for (int v : valid_views) {
        if (std::find(selected.begin(), selected.end(), v) != selected.end()) continue;
        const double conf = kpts2d_by_view[v](2);
        if (conf <= 0.0) continue;
        const auto proj = project_point(Pall[v], X3);
        const double err = (proj.head<2>() - kpts2d_by_view[v].head<2>()).norm();
        if (err <= dist_max) {
          candidates.push_back(v);
        }
      }
      std::sort(candidates.begin(), candidates.end(),
                [&](int a, int b) { return kpts2d_by_view[a](2) > kpts2d_by_view[b](2); });
    }

    for (int v : candidates) {
      std::vector<int> trial = selected;
      trial.push_back(v);
      Eigen::Vector4d X;
      if (!triangulate_point_dlt(kpts2d_by_view, Pall, trial, X)) {
        continue;
      }
      if (X(3) <= 0.0) {
        continue;
      }
      const Eigen::Vector3d X3 = X.head<3>();
      bool ok = true;
      for (int vv : trial) {
        const auto proj = project_point(Pall[vv], X3);
        const double err = (proj.head<2>() - kpts2d_by_view[vv].head<2>()).norm();
        if (err > dist_max) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        break;
      }
      selected = std::move(trial);
    }

    return selected;
  }

  static std::vector<Eigen::Vector4d> iterative_triangulate(
      const std::vector<std::vector<Eigen::Vector3d>>& keypoints2d,
      const std::vector<Eigen::Matrix<double, 3, 4>>& projection_matrices, double dist_max = 25.0,
      int min_views = 3, double min_conf = 0.1, double thres_outlier_view = 0.4,
      double thres_outlier_joint = 0.4, int max_iterations = 30, int min_joints = 3) {
    if (keypoints2d.empty() || projection_matrices.empty()) {
      return {};
    }

    const size_t n_views = keypoints2d.size();
    const size_t n_joints = keypoints2d[0].size();
    if (projection_matrices.size() != n_views) {
      return {};
    }

    auto kpts2d = keypoints2d;
    for (size_t v = 0; v < n_views; ++v) {
      for (size_t j = 0; j < n_joints; ++j) {
        if (kpts2d[v][j](2) < min_conf) {
          kpts2d[v][j](2) = 0.0;
        }
      }
    }

    std::vector<Eigen::Vector4d> kpts3d(n_joints, Eigen::Vector4d::Zero());

    for (int iter = 0; iter < max_iterations; ++iter) {
      kpts3d = triangulate_keypoints(kpts2d, projection_matrices, min_views);

      std::vector<std::vector<double>> dist(n_views, std::vector<double>(n_joints, 0.0));
      std::vector<std::vector<int>> valid(n_views, std::vector<int>(n_joints, 0));
      std::vector<std::vector<int>> outlier(n_views, std::vector<int>(n_joints, 0));

      bool any_outlier = false;
      std::vector<double> dist_sum_view(n_views, 0.0);
      std::vector<int> valid_cnt_view(n_views, 0);
      std::vector<int> out_cnt_view(n_views, 0);

      std::vector<double> dist_sum_joint(n_joints, 0.0);
      std::vector<int> valid_cnt_joint(n_joints, 0);
      std::vector<int> out_cnt_joint(n_joints, 0);

      for (size_t v = 0; v < n_views; ++v) {
        const auto& P = projection_matrices[v];
        for (size_t j = 0; j < n_joints; ++j) {
          if (kpts3d[j](3) <= 0.0) continue;
          if (kpts2d[v][j](2) <= 0.0) continue;

          Eigen::Vector4d X;
          X.head<3>() = kpts3d[j].head<3>();
          X(3) = 1.0;
          Eigen::Vector3d proj = P * X;
          proj /= proj(2);

          double d = (proj.head<2>() - kpts2d[v][j].head<2>()).norm();
          dist[v][j] = d;
          valid[v][j] = 1;
          valid_cnt_view[v] += 1;
          valid_cnt_joint[j] += 1;
          dist_sum_view[v] += d;
          dist_sum_joint[j] += d;

          if (d > dist_max) {
            outlier[v][j] = 1;
            out_cnt_view[v] += 1;
            out_cnt_joint[j] += 1;
            any_outlier = true;
          }
        }
      }

      if (!any_outlier) {
        break;
      }

      int remove_view = -1;
      double best_score = -1.0;
      for (size_t v = 0; v < n_views; ++v) {
        if (valid_cnt_view[v] <= 0) continue;
        double ratio =
            static_cast<double>(out_cnt_view[v]) / (1e-5 + static_cast<double>(valid_cnt_view[v]));
        if (ratio <= thres_outlier_view) continue;
        double mean_dist = dist_sum_view[v] / (1e-5 + static_cast<double>(valid_cnt_view[v]));
        if (mean_dist > best_score) {
          best_score = mean_dist;
          remove_view = static_cast<int>(v);
        }
      }
      if (remove_view >= 0) {
        for (size_t j = 0; j < n_joints; ++j) {
          kpts2d[remove_view][j](2) = 0.0;
        }
        continue;
      }

      int removed_any_joint = 0;
      for (size_t j = 0; j < n_joints; ++j) {
        if (valid_cnt_joint[j] <= 0) continue;
        double ratio = static_cast<double>(out_cnt_joint[j]) /
                       (1e-5 + static_cast<double>(valid_cnt_joint[j]));
        if (ratio <= thres_outlier_joint) continue;

        std::vector<int> valid_views;
        valid_views.reserve(n_views);
        for (size_t v = 0; v < n_views; ++v) {
          if (kpts2d[v][j](2) > 0.0) {
            valid_views.push_back(static_cast<int>(v));
          }
        }
        if (static_cast<int>(valid_views.size()) < min_views) {
          for (size_t v = 0; v < n_views; ++v) {
            kpts2d[v][j](2) = 0.0;
          }
          removed_any_joint = 1;
          continue;
        }

        std::vector<Eigen::Vector3d> kpts_by_view(n_views, Eigen::Vector3d::Zero());
        for (size_t v = 0; v < n_views; ++v) {
          kpts_by_view[v] = kpts2d[v][j];
        }

        const auto selected_views = robust_select_views_for_joint(kpts_by_view, projection_matrices,
                                                                  valid_views, dist_max, min_views);

        std::vector<char> keep(n_views, 0);
        for (int v : selected_views) {
          if (v >= 0 && static_cast<size_t>(v) < n_views) {
            keep[static_cast<size_t>(v)] = 1;
          }
        }
        for (size_t v = 0; v < n_views; ++v) {
          if (!keep[v]) {
            kpts2d[v][j](2) = 0.0;
          }
        }
        removed_any_joint = 1;
      }
      if (removed_any_joint) {
        continue;
      }

      for (size_t v = 0; v < n_views; ++v) {
        for (size_t j = 0; j < n_joints; ++j) {
          if (outlier[v][j]) {
            kpts2d[v][j](2) = 0.0;
          }
        }
      }
    }

    int n_valid = 0;
    for (size_t j = 0; j < n_joints; ++j) {
      if (kpts3d[j](3) > 0.0) n_valid++;
    }
    if (n_valid < min_joints) {
      for (auto& k : kpts3d) k(3) = 0.0;
    }
    return kpts3d;
  }
};

}  // namespace preprocess
}  // namespace marionette
