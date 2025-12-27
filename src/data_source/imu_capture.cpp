#include "imu_capture.hpp"

#include <spdlog/spdlog.h>

#include <thread>

#include "base64.hpp"

namespace marionette::data_source {

imu_capture::imu_capture() : running(false) {}

void imu_capture::open(std::string port_name) {
  port.open(port_name);
  port.set_baudrate(1500000);
}

void imu_capture::start(std::function<void(const pose_frame &)> frame_received) {
  std::vector<uint8_t> buf;
  bool first_frame = true;
  running = true;

  while (running.load()) {
    size_t receive_len = port.get_received_size();
    if (receive_len <= 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    std::vector<uint8_t> data(receive_len);
    port.read(data.data(), data.size());

    std::vector<std::string> lines;
    for (const auto c : data) {
      if (c == '\n') {
        lines.push_back(std::string(buf.begin(), buf.end()));
        buf.clear();
      } else {
        buf.push_back(c);
      }
    }

    for (const auto &line : lines) {
      response_header header;
      size_t header_size = 0;
      decode_base64(line, (uint8_t *)&header, sizeof(header), &header_size);
      if (header_size <= line.size()) {
        frame_data frame;
        size_t frame_size = 0;
        decode_base64(line.substr(header_size), (uint8_t *)&frame, sizeof(frame), &frame_size);
        if (header_size + frame_size + 1 /* \r */ == line.size()) {
          pose_frame pose;

          if (first_frame) {
            system_clock_start = std::chrono::system_clock::now();
            device_clock_start = frame.timestamp;
            first_frame = false;
          }

          const auto timestamp = (std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      system_clock_start.time_since_epoch())
                                      .count() +
                                  (frame.timestamp - device_clock_start)) /
                                 1000000.0;

          pose.timestamp = timestamp;
          for (int i = 0; i < NUM_SENSORS; i++) {
            pose_data data;
            data.accel_status = frame.imu[i].accel;
            data.gyro_status = frame.imu[i].gyro;
            data.mag_status = frame.imu[i].mag;
            data.orientation = glm::quat(static_cast<float>(frame.imu[i].orientation_quat.w),
                                         static_cast<float>(frame.imu[i].orientation_quat.x),
                                         static_cast<float>(frame.imu[i].orientation_quat.y),
                                         static_cast<float>(frame.imu[i].orientation_quat.z));
            pose.poses.push_back(data);
          }

          frame_received(pose);
        }
      }
    }
  }
  spdlog::info("Capture loop stopped");
}

void imu_capture::stop() { running = false; }

bool imu_capture::is_running() const { return running.load(); }

}  // namespace marionette::data_source
