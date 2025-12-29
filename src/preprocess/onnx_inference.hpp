#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>

namespace marionette {
namespace preprocess {

/**
 * ONNX Runtime wrapper for a YOLO-style detector.
 */
class YOLODetector {
 private:
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
  int input_height_;
  int input_width_;

 public:
  struct Detection {
    float x1, y1, x2, y2;  // bounding box (xyxy)
    float confidence;
    int class_id;
  };

  YOLODetector(const std::string& model_path, int input_size = 640)
      : input_height_(input_size), input_width_(input_size) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLODetector");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // Cache input/output node names.
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t num_input_nodes = session_->GetInputCount();
    input_names_.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      input_names_.emplace_back(input_name.get());
    }

    size_t num_output_nodes = session_->GetOutputCount();
    output_names_.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      output_names_.emplace_back(output_name.get());
    }

    input_name_ptrs_.reserve(input_names_.size());
    for (const auto& name : input_names_) {
      input_name_ptrs_.push_back(name.c_str());
    }

    output_name_ptrs_.reserve(output_names_.size());
    for (const auto& name : output_names_) {
      output_name_ptrs_.push_back(name.c_str());
    }
  }

  std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.25, 
                                 float nms_threshold = 0.45) {
    // Preprocess: letterbox resize to input_size x input_size.
    // Keeps aspect ratio, pads remaining area.
    const int orig_w = image.cols;
    const int orig_h = image.rows;
    const float r = std::min(static_cast<float>(input_width_) / static_cast<float>(orig_w),
                             static_cast<float>(input_height_) / static_cast<float>(orig_h));
    const int new_w = static_cast<int>(std::round(orig_w * r));
    const int new_h = static_cast<int>(std::round(orig_h * r));
    const float pad_x = (input_width_ - new_w) * 0.5f;
    const float pad_y = (input_height_ - new_h) * 0.5f;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    cv::Mat boxed(input_height_, input_width_, image.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(boxed(cv::Rect(static_cast<int>(std::floor(pad_x)),
                                 static_cast<int>(std::floor(pad_y)),
                                 new_w, new_h)));

    cv::cvtColor(boxed, boxed, cv::COLOR_BGR2RGB);

    // Normalize [0,255] -> [0,1]
    boxed.convertTo(boxed, CV_32F, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<float> input_tensor_values(1 * 3 * input_height_ * input_width_);
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < input_height_; ++h) {
        for (int w = 0; w < input_width_; ++w) {
          input_tensor_values[c * input_height_ * input_width_ + h * input_width_ + w] =
              boxed.at<cv::Vec3f>(h, w)[c];
        }
      }
    }

    // Inference
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_name_ptrs_.data(), &input_tensor, 1,
                                        output_name_ptrs_.data(), output_name_ptrs_.size());

    // Postprocess
    float* output = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<Detection> detections;
    
    // Output format: [batch, num_detections, num_elements]
    // Common case: num_elements = 4 (bbox) + 1 (obj_conf) + num_classes
    int num_detections = output_shape[1];
    int num_elements = output_shape[2];

    // Inverse letterbox scale.
    const float inv_r = (r > 0.0f) ? (1.0f / r) : 0.0f;

    for (int i = 0; i < num_detections; ++i) {
      float* detection = output + i * num_elements;
      
      float obj_conf = detection[4];
      if (obj_conf < conf_threshold) {
        continue;
      }

      // Pick the max class score.
      float max_class_score = 0.0f;
      int max_class_id = 0;
      for (int j = 5; j < num_elements; ++j) {
        if (detection[j] > max_class_score) {
          max_class_score = detection[j];
          max_class_id = j - 5;
        }
      }

      float confidence = obj_conf * max_class_score;
      if (confidence < conf_threshold) {
        continue;
      }

      // Map bbox from network input space back to the original image.
      Detection det;
      float cx = detection[0];
      float cy = detection[1];
      float w = detection[2];
      float h = detection[3];

      // Some exports produce normalized (0..1) coordinates; handle that case.
      if (cx <= 2.0f && cy <= 2.0f && w <= 2.0f && h <= 2.0f) {
        cx *= static_cast<float>(input_width_);
        cy *= static_cast<float>(input_height_);
        w *= static_cast<float>(input_width_);
        h *= static_cast<float>(input_height_);
      }

      // letterbox coords -> original image coords
      cx = (cx - pad_x) * inv_r;
      cy = (cy - pad_y) * inv_r;
      w = w * inv_r;
      h = h * inv_r;

      det.x1 = cx - w / 2;
      det.y1 = cy - h / 2;
      det.x2 = cx + w / 2;
      det.y2 = cy + h / 2;

      // clamp
      det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(orig_w - 1)));
      det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(orig_h - 1)));
      det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(orig_w - 1)));
      det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(orig_h - 1)));
      det.confidence = confidence;
      det.class_id = max_class_id;

      detections.push_back(det);
    }

    // NMS (Non-Maximum Suppression)
    return apply_nms(detections, nms_threshold);
  }

 private:
  std::vector<Detection> apply_nms(const std::vector<Detection>& detections,
                                    float nms_threshold) {
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
      if (suppressed[i]) {
        continue;
      }

      result.push_back(detections[i]);

      for (size_t j = i + 1; j < detections.size(); ++j) {
        if (suppressed[j]) {
          continue;
        }

        if (detections[i].class_id != detections[j].class_id) {
          continue;
        }

        float iou = compute_iou(detections[i], detections[j]);
        if (iou > nms_threshold) {
          suppressed[j] = true;
        }
      }
    }

    return result;
  }

  float compute_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    return intersection / union_area;
  }
};

/**
 * ONNX Runtime wrapper for a heatmap-based 2D pose model.
 */
class HRNetPoseEstimator {
 private:
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
  int input_height_;
  int input_width_;
  int num_joints_;

  static cv::Point2f rotate_2d(const cv::Point2f& pt, float rot_rad) {
    float sn = std::sin(rot_rad);
    float cs = std::cos(rot_rad);
    return cv::Point2f(pt.x * cs - pt.y * sn, pt.x * sn + pt.y * cs);
  }

  static void gen_trans_from_patch_cv(float c_x, float c_y,
                                      float src_width, float src_height,
                                      float dst_width, float dst_height,
                                      float scale, float rot,
                                      cv::Mat& trans, cv::Mat& inv_trans) {
    float src_w = src_width * scale;
    float src_h = src_height * scale;
    cv::Point2f src_center(c_x, c_y);

    float rot_rad = static_cast<float>(M_PI) * rot / 180.0f;
    cv::Point2f src_downdir = rotate_2d(cv::Point2f(0.0f, src_h * 0.5f), rot_rad);
    cv::Point2f src_rightdir = rotate_2d(cv::Point2f(src_w * 0.5f, 0.0f), rot_rad);

    cv::Point2f dst_center(dst_width * 0.5f, dst_height * 0.5f);
    cv::Point2f dst_downdir(0.0f, dst_height * 0.5f);
    cv::Point2f dst_rightdir(dst_width * 0.5f, 0.0f);

    cv::Point2f src[3];
    src[0] = src_center;
    src[1] = src_center + src_downdir;
    src[2] = src_center + src_rightdir;

    cv::Point2f dst[3];
    dst[0] = dst_center;
    dst[1] = dst_center + dst_downdir;
    dst[2] = dst_center + dst_rightdir;

    trans = cv::getAffineTransform(src, dst);
    inv_trans = cv::getAffineTransform(dst, src);
  }

  static cv::Point2f affine_transform(const cv::Point2f& pt, const cv::Mat& trans) {
    cv::Matx23f M;
    trans.convertTo(M, CV_32F);
    return cv::Point2f(M(0, 0) * pt.x + M(0, 1) * pt.y + M(0, 2),
                       M(1, 0) * pt.x + M(1, 1) * pt.y + M(1, 2));
  }

 public:
  HRNetPoseEstimator(const std::string& model_path, int num_joints = 17,
                     int input_height = 384, int input_width = 288)
      : input_height_(input_height), input_width_(input_width), num_joints_(num_joints) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HRNetPoseEstimator");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    // Cache input/output node names.
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t num_input_nodes = session_->GetInputCount();
    input_names_.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      input_names_.emplace_back(input_name.get());
    }

    size_t num_output_nodes = session_->GetOutputCount();
    output_names_.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      output_names_.emplace_back(output_name.get());
    }

    input_name_ptrs_.reserve(input_names_.size());
    for (const auto& name : input_names_) {
      input_name_ptrs_.push_back(name.c_str());
    }

    output_name_ptrs_.reserve(output_names_.size());
    for (const auto& name : output_names_) {
      output_name_ptrs_.push_back(name.c_str());
    }
  }

  /**
   * Estimate 2D pose keypoints from a person bounding box.
   * @param image Input image
   * @param bbox Bounding box [x1, y1, x2, y2]
   * @param bbox_scale Box scale factor
   * @return Keypoints [num_joints](x, y, confidence)
   */
  std::vector<Eigen::Vector3d> estimate_pose(const cv::Mat& image,
                                              const std::vector<float>& bbox,
                                              float bbox_scale = 1.25) {
    // Preprocess: xyxy -> (cx, cy, w, h), then match the network input aspect ratio.
    float x1 = bbox[0];
    float y1 = bbox[1];
    float x2 = bbox[2];
    float y2 = bbox[3];

    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = (x2 - x1);
    float h = (y2 - y1);

    // aspect_ratio = crop_h / crop_w
    float aspect_ratio = static_cast<float>(input_height_) / static_cast<float>(input_width_);
    if (h > aspect_ratio * w) {
      w = h / aspect_ratio;
    } else {
      h = w * aspect_ratio;
    }

    cv::Mat trans, inv_trans;
    gen_trans_from_patch_cv(cx, cy, w, h,
                            static_cast<float>(input_width_), static_cast<float>(input_height_),
                            bbox_scale, 0.0f, trans, inv_trans);

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::warpAffine(rgb, resized, trans, cv::Size(input_width_, input_height_),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // Normalize (RGB, 0-1, (x-mean)/std)
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    cv::subtract(resized, cv::Scalar(0.485, 0.456, 0.406), resized);
    cv::divide(resized, cv::Scalar(0.229, 0.224, 0.225), resized);

    // HWC -> CHW
    std::vector<float> input_tensor_values(1 * 3 * input_height_ * input_width_);
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < input_height_; ++h) {
        for (int w = 0; w < input_width_; ++w) {
          input_tensor_values[c * input_height_ * input_width_ + h * input_width_ + w] =
              resized.at<cv::Vec3f>(h, w)[c];
        }
      }
    }

    // Inference
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                      input_name_ptrs_.data(), &input_tensor, 1,
                      output_name_ptrs_.data(), output_name_ptrs_.size());

    // Decode heatmaps into 2D keypoints
    float* heatmaps = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int heatmap_height = output_shape[2];
    int heatmap_width = output_shape[3];

    std::vector<Eigen::Vector3d> keypoints(num_joints_);

    for (int j = 0; j < num_joints_; ++j) {
      float* heatmap = heatmaps + j * heatmap_height * heatmap_width;

      // Argmax
      float max_val = -1.0f;
      int max_y = 0, max_x = 0;
      for (int y = 0; y < heatmap_height; ++y) {
        for (int x = 0; x < heatmap_width; ++x) {
          float val = heatmap[y * heatmap_width + x];
          if (val > max_val) {
            max_val = val;
            max_y = y;
            max_x = x;
          }
        }
      }

      // Optional sub-pixel refinement.
      float coord_x = static_cast<float>(max_x);
      float coord_y = static_cast<float>(max_y);

      if (max_val <= 0.0f) {
        coord_x = 0.0f;
        coord_y = 0.0f;
      }

      // Simple refinement using local gradient around the peak.
      int px = static_cast<int>(std::floor(coord_x + 0.5f));
      int py = static_cast<int>(std::floor(coord_y + 0.5f));
      if (1 < px && px < heatmap_width - 1 && 1 < py && py < heatmap_height - 1) {
        float dx = heatmap[py * heatmap_width + (px + 1)] - heatmap[py * heatmap_width + (px - 1)];
        float dy = heatmap[(py + 1) * heatmap_width + px] - heatmap[(py - 1) * heatmap_width + px];
        coord_x += (dx > 0 ? 0.25f : (dx < 0 ? -0.25f : 0.0f));
        coord_y += (dy > 0 ? 0.25f : (dy < 0 ? -0.25f : 0.0f));
      }

      // heatmap coords -> input image coords (x4)
      cv::Point2f pt_in(coord_x * 4.0f, coord_y * 4.0f);

      // input image coords -> original image coords (inv_trans)
      cv::Point2f pt_ori = affine_transform(pt_in, inv_trans);

      keypoints[j](0) = static_cast<double>(pt_ori.x);
      keypoints[j](1) = static_cast<double>(pt_ori.y);
      keypoints[j](2) = static_cast<double>(max_val);
    }

    return keypoints;
  }
};

}  // namespace preprocess
}  // namespace marionette
