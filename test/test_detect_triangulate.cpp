#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "preprocess/camera_utils.hpp"
#include "preprocess/onnx_inference.hpp"
#include "preprocess/triangulate.hpp"

using namespace marionette::preprocess;
namespace fs = std::filesystem;

struct config_t {
  std::string image_root;
  std::string intri_file;
  std::string extri_file;
  std::string output_dir;
  int start_frame;
  int end_frame;
  int frame_step;
  std::string yolo_model_path;
  std::string hrnet_model_path;
};

void save_keypoints3d(const std::vector<std::vector<Eigen::Vector4d>>& keypoints3d_all,
                      const std::string& output_file) {
  if (keypoints3d_all.empty()) {
    std::cerr << "Error: No keypoints3d data to save" << std::endl;
    return;
  }

  nlohmann::json j;
  j["type"] = "float64";

  int num_frames = keypoints3d_all.size();
  int num_joints = keypoints3d_all[0].size();
  j["shape"] = {num_frames, num_joints, 4};

  std::vector<double> data;
  data.reserve(num_frames * num_joints * 4);

  for (const auto& frame_kpts : keypoints3d_all) {
    for (const auto& kpt : frame_kpts) {
      data.push_back(kpt(0));  // x
      data.push_back(kpt(1));  // y
      data.push_back(kpt(2));  // z
      data.push_back(kpt(3));  // confidence
    }
  }
  j["data"] = data;

  fs::create_directories(fs::path(output_file).parent_path());
  std::ofstream ofs(output_file);
  ofs << j.dump(2);
  ofs.close();

  std::cout << "Saved init_params input: " << output_file << std::endl;
  std::cout << "  Shape: [" << num_frames << ", " << num_joints << ", 4]" << std::endl;
  std::cout << "  Total data points: " << data.size() << std::endl;
}

std::vector<Eigen::Vector3d> coco17_to_body25(const std::vector<Eigen::Vector3d>& coco17) {
  static const std::vector<int> COCO17_IN_BODY25 = {0, 16, 15, 18, 17, 5,  2,  6, 3,
                                                    7, 4,  12, 9,  13, 10, 14, 11};

  std::vector<Eigen::Vector3d> body25(25, Eigen::Vector3d::Zero());

  for (size_t i = 0; i < COCO17_IN_BODY25.size() && i < coco17.size(); ++i) {
    body25[COCO17_IN_BODY25[i]] = coco17[i];
  }

  // Derived joints: MidHip (8) and Neck (1)
  // XY: mean of source joints, Conf: min of source joints
  {
    const double conf_min = std::min(body25[9](2), body25[12](2));
    const Eigen::Vector2d xy_mean = (body25[9].head<2>() + body25[12].head<2>()) * 0.5;
    body25[8] = Eigen::Vector3d(xy_mean.x(), xy_mean.y(), conf_min);
  }
  {
    const double conf_min = std::min(body25[2](2), body25[5](2));
    const Eigen::Vector2d xy_mean = (body25[2].head<2>() + body25[5].head<2>()) * 0.5;
    body25[1] = Eigen::Vector3d(xy_mean.x(), xy_mean.y(), conf_min);
  }

  return body25;
}

int main() {
  config_t config;

  config.image_root = "../data/street_dance/images";
  config.intri_file = "../data/street_dance/intri.yml";
  config.extri_file = "../data/street_dance/extri.yml";
  config.output_dir = "../data/opt";
  config.start_frame = 0;
  config.end_frame = 10;
  config.frame_step = 1;
  config.yolo_model_path = "../models/yolov5m.onnx";
  config.hrnet_model_path = "../models/hrnet_w48_384x288.onnx";

  std::cout << "=== Detection and Triangulation (C++) ===" << std::endl;
  std::cout << "Image root: " << config.image_root << std::endl;
  std::cout << "Output dir: " << config.output_dir << std::endl;
  std::cout << "Frame range: " << config.start_frame << " to " << config.end_frame << " (step "
            << config.frame_step << ")" << std::endl;

  std::cout << "\nLoading camera parameters..." << std::endl;
  auto all_cameras = camera_loader_t::load_cameras(config.intri_file, config.extri_file);

  std::vector<camera_params_t> cameras = all_cameras;

  std::cout << "Loaded " << cameras.size() << " cameras: ";
  for (const auto& cam : cameras) {
    std::cout << cam.name << " ";
  }
  std::cout << std::endl;

  std::cout << "\nLoading ONNX models..." << std::endl;
  std::cout << "Note: YOLO and HRNet models need to be exported first using:" << std::endl;
  std::cout << "  python ../scripts/export_yolo_onnx.py" << std::endl;
  std::cout << "  python ../scripts/export_hrnet_onnx.py" << std::endl;

  bool models_exist = fs::exists(config.yolo_model_path) && fs::exists(config.hrnet_model_path);

  if (!models_exist) {
    std::cerr << "\nError: ONNX model files not found." << std::endl;
    std::cerr << "  YOLO:  " << config.yolo_model_path << std::endl;
    std::cerr << "  HRNet: " << config.hrnet_model_path << std::endl;
    return 1;
  }

  yolo_detector_t yolo(config.yolo_model_path);
  hrnet_pose_estimator_t hrnet(config.hrnet_model_path);

  std::cout << "Models loaded successfully." << std::endl;

  std::vector<std::vector<Eigen::Vector4d>> keypoints3d_all_frames;

  for (int frame = config.start_frame; frame < config.end_frame; frame += config.frame_step) {
    std::cout << "\n--- Processing frame " << frame << " ---" << std::endl;

    std::vector<std::vector<Eigen::Vector3d>> keypoints2d_all_views;

    for (const auto& cam : cameras) {
      std::string image_path =
          config.image_root + "/" + cam.name + "/" +
          std::to_string(frame).insert(0, 6 - std::to_string(frame).length(), '0') + ".jpg";

      if (!fs::exists(image_path)) {
        std::cerr << "Warning: Image not found: " << image_path << std::endl;
        keypoints2d_all_views.push_back({});
        continue;
      }

      cv::Mat image = cv::imread(image_path);
      if (image.empty()) {
        std::cerr << "Warning: Failed to load image: " << image_path << std::endl;
        keypoints2d_all_views.push_back({});
        continue;
      }

      auto detections = yolo.detect(image);

      std::vector<yolo_detector_t::detection_t> person_detections;
      for (const auto& det : detections) {
        if (det.class_id == 0) {
          person_detections.push_back(det);
        }
      }

      if (person_detections.empty()) {
        std::cout << "  Camera " << cam.name << ": No person detected" << std::endl;
        keypoints2d_all_views.push_back({});
        continue;
      }

      auto best_detection = person_detections[0];
      for (const auto& det : person_detections) {
        if (det.confidence > best_detection.confidence) {
          best_detection = det;
        }
      }

      std::cout << "  Camera " << cam.name
                << ": Person detected (conf: " << best_detection.confidence << ")" << std::endl;

      std::vector<float> bbox = {best_detection.x1, best_detection.y1, best_detection.x2,
                                 best_detection.y2};
      auto keypoints2d = hrnet.estimate_pose(image, bbox);

      auto keypoints2d_body25 = coco17_to_body25(keypoints2d);

      std::vector<Eigen::Vector3d> keypoints2d_undist;
      for (const auto& kpt : keypoints2d_body25) {
        if (kpt(2) > 0) {
          Eigen::Vector2d undist = cam.undistort_point(kpt.head<2>());
          keypoints2d_undist.push_back(Eigen::Vector3d(undist.x(), undist.y(), kpt(2)));
        } else {
          keypoints2d_undist.push_back(kpt);
        }
      }

      keypoints2d_all_views.push_back(keypoints2d_undist);
    }

    std::vector<Eigen::Matrix<double, 3, 4>> projection_matrices;
    for (const auto& cam : cameras) {
      projection_matrices.push_back(cam.p);
    }

    auto keypoints3d = triangulator_t::iterative_triangulate(keypoints2d_all_views,
                                                             projection_matrices, 25.0, 3, 0.1);

    keypoints3d_all_frames.push_back(keypoints3d);

    std::cout << "  Triangulated " << keypoints3d.size() << " keypoints" << std::endl;
  }

  std::cout << "\nSaving results to " << config.output_dir << " ..." << std::endl;

  std::string keypoints3d_file = config.output_dir + "/keypoints3d.json";
  save_keypoints3d(keypoints3d_all_frames, keypoints3d_file);

  std::cout << "\n=== Processing Complete ===" << std::endl;
  std::cout << "Output saved to: " << config.output_dir << std::endl;
  std::cout << "  - Keypoints 3D: " << keypoints3d_file << std::endl;
  std::cout << "You can now use keypoints3d.json with test_fitting_backward" << std::endl;

  return 0;
}
