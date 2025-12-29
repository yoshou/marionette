#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <yaml-cpp/yaml.h>

namespace marionette {
namespace preprocess {

struct CameraParams {
  std::string name;
  Eigen::Matrix3d K;              // Intrinsics
  Eigen::VectorXd dist;           // Distortion [k1, k2, p1, p2, k3, ...]
  Eigen::Matrix3d R;              // Rotation
  Eigen::Vector3d T;              // Translation
  Eigen::Matrix<double, 3, 4> P;  // Projection matrix: P = K [R|T]

  // Compute P = K [R|T]
  void compute_projection_matrix() {
    Eigen::Matrix<double, 3, 4> RT;
    RT.leftCols<3>() = R;
    RT.rightCols<1>() = T;
    P = K * RT;
  }

  // Undistort a single 2D point (pixel coordinates)
  Eigen::Vector2d undistort_point(const Eigen::Vector2d& pt) const {
    if (dist.size() < 4) {
      return pt;
    }

    cv::Mat Kcv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        Kcv.at<double>(r, c) = K(r, c);
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
    cv::undistortPoints(src, dst, Kcv, distcv, cv::noArray(), Kcv);

    Eigen::Vector2d undist;
    undist.x() = static_cast<double>(dst[0].x);
    undist.y() = static_cast<double>(dst[0].y);
    return undist;
  }
};

class CameraLoader {
 public:
  static std::vector<CameraParams> load_cameras(const std::string& intri_path,
                                                 const std::string& extri_path) {
    std::vector<CameraParams> cameras;

    // Load YAML files
    YAML::Node intri = YAML::LoadFile(intri_path);
    YAML::Node extri = YAML::LoadFile(extri_path);

    auto names = intri["names"].as<std::vector<std::string>>();

    for (const auto& name : names) {
      CameraParams cam;
      cam.name = name;

      // Intrinsics
      std::string k_key = "K_" + name;
      std::string dist_key = "dist_" + name;

      auto K_node = intri[k_key];
      if (!K_node.IsDefined()) {
        std::cerr << "Warning: K not found for camera " << name << std::endl;
        continue;
      }

      auto K_data = K_node["data"].as<std::vector<double>>();
      cam.K = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K_data.data());

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

      auto Rvec_node = extri[rvec_key];
      auto Rot_node = extri[rot_key];
      auto T_node = extri[t_key];

      if ((!Rvec_node.IsDefined() && !Rot_node.IsDefined()) || !T_node.IsDefined()) {
        std::cerr << "Warning: R/Rot or T not found for camera " << name << std::endl;
        continue;
      }

      auto T_data = T_node["data"].as<std::vector<double>>();

      // Prefer Rodrigues vector R_XX if present
      if (Rvec_node.IsDefined()) {
        auto rvec_data = Rvec_node["data"].as<std::vector<double>>();
        if (rvec_data.size() != 3) {
          std::cerr << "Warning: Rvec size != 3 for camera " << name << std::endl;
          continue;
        }
        cv::Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = rvec_data[0];
        rvec.at<double>(1, 0) = rvec_data[1];
        rvec.at<double>(2, 0) = rvec_data[2];
        cv::Mat Rcv;
        cv::Rodrigues(rvec, Rcv);
        Eigen::Matrix3d R;
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            R(r, c) = Rcv.at<double>(r, c);
          }
        }
        cam.R = R;
      } else {
        auto Rot_data = Rot_node["data"].as<std::vector<double>>();
        cam.R = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(Rot_data.data());
      }

      cam.T = Eigen::Map<Eigen::Vector3d>(T_data.data());

      // Compute projection matrix
      cam.compute_projection_matrix();

      cameras.push_back(cam);
    }

    return cameras;
  }

  // Filter by camera names
  static std::vector<CameraParams> filter_cameras(const std::vector<CameraParams>& all_cameras,
                                                   const std::vector<std::string>& selected_names) {
    std::vector<CameraParams> filtered;
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
