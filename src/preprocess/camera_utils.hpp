#pragma once

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace marionette {
namespace preprocess {

struct camera_params_t {
  std::string name;
  Eigen::Matrix3d k;              // Intrinsics
  Eigen::VectorXd dist;           // Distortion [k1, k2, p1, p2, k3, ...]
  Eigen::Matrix3d r;              // Rotation
  Eigen::Vector3d t;              // Translation
  Eigen::Matrix<double, 3, 4> p;  // Projection matrix: P = K [R|T]

  // Compute P = K [R|T]
  void compute_projection_matrix() {
    Eigen::Matrix<double, 3, 4> rt;
    rt.leftCols<3>() = r;
    rt.rightCols<1>() = t;
    p = k * rt;
  }

  // Undistort a single 2D point (pixel coordinates)
  Eigen::Vector2d undistort_point(const Eigen::Vector2d& pt) const {
    if (dist.size() < 4) {
      return pt;
    }

    cv::Mat k_cv(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        k_cv.at<double>(row, col) = k(row, col);
      }
    }

    // OpenCV distortion order: (k1, k2, p1, p2, k3, ...)
    cv::Mat distcv(1, static_cast<int>(dist.size()), CV_64F);
    for (int i = 0; i < distcv.cols; ++i) {
      distcv.at<double>(0, i) = dist(i);
    }

    std::vector<cv::Point2f> src(1);
    src[0] = cv::Point2f(static_cast<float>(pt.x()), static_cast<float>(pt.y()));
    std::vector<cv::Point2f> dst;

    // Use P=K so outputs stay in pixel coordinates
    cv::undistortPoints(src, dst, k_cv, distcv, cv::noArray(), k_cv);

    Eigen::Vector2d undist;
    undist.x() = static_cast<double>(dst[0].x);
    undist.y() = static_cast<double>(dst[0].y);
    return undist;
  }
};

class camera_loader_t {
 public:
  static std::vector<camera_params_t> load_cameras(const std::string& intri_path,
                                                   const std::string& extri_path) {
    std::vector<camera_params_t> cameras;

    // Load YAML files
    YAML::Node intri = YAML::LoadFile(intri_path);
    YAML::Node extri = YAML::LoadFile(extri_path);

    auto names = intri["names"].as<std::vector<std::string>>();

    for (const auto& name : names) {
      camera_params_t cam;
      cam.name = name;

      // Intrinsics
      std::string k_key = "K_" + name;
      std::string dist_key = "dist_" + name;

      auto k_node = intri[k_key];
      if (!k_node.IsDefined()) {
        std::cerr << "Warning: K not found for camera " << name << std::endl;
        continue;
      }

      auto k_data = k_node["data"].as<std::vector<double>>();
      cam.k = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(k_data.data());

      if (intri[dist_key].IsDefined()) {
        auto dist_data = intri[dist_key]["data"].as<std::vector<double>>();
        cam.dist = Eigen::Map<Eigen::VectorXd>(dist_data.data(), dist_data.size());
      } else {
        cam.dist = Eigen::VectorXd::Zero(5);
      }

      // Extrinsics
      std::string rvec_key = "R_" + name;
      std::string rot_key = "Rot_" + name;
      std::string t_key = "T_" + name;

      auto rvec_node = extri[rvec_key];
      auto rot_node = extri[rot_key];
      auto t_node = extri[t_key];

      if ((!rvec_node.IsDefined() && !rot_node.IsDefined()) || !t_node.IsDefined()) {
        std::cerr << "Warning: R/Rot or T not found for camera " << name << std::endl;
        continue;
      }

      auto t_data = t_node["data"].as<std::vector<double>>();

      // Prefer Rodrigues vector R_XX if present
      if (rvec_node.IsDefined()) {
        auto rvec_data = rvec_node["data"].as<std::vector<double>>();
        if (rvec_data.size() != 3) {
          std::cerr << "Warning: Rvec size != 3 for camera " << name << std::endl;
          continue;
        }
        cv::Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = rvec_data[0];
        rvec.at<double>(1, 0) = rvec_data[1];
        rvec.at<double>(2, 0) = rvec_data[2];
        cv::Mat r_cv;
        cv::Rodrigues(rvec, r_cv);
        Eigen::Matrix3d r_mat;
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            r_mat(row, col) = r_cv.at<double>(row, col);
          }
        }
        cam.r = r_mat;
      } else {
        auto rot_data = rot_node["data"].as<std::vector<double>>();
        cam.r = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rot_data.data());
      }

      cam.t = Eigen::Map<Eigen::Vector3d>(t_data.data());

      // Compute projection matrix
      cam.compute_projection_matrix();

      cameras.push_back(cam);
    }

    return cameras;
  }

  // Filter by camera names
  static std::vector<camera_params_t> filter_cameras(
      const std::vector<camera_params_t>& all_cameras,
      const std::vector<std::string>& selected_names) {
    std::vector<camera_params_t> filtered;
    for (const auto& name : selected_names) {
      for (const auto& cam : all_cameras) {
        if (cam.name == name) {
          filtered.push_back(cam);
          break;
        }
      }
    }
    return filtered;
  }
};

}  // namespace preprocess
}  // namespace marionette
