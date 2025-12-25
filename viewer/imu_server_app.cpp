// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "axis_drawer.hpp"
#include "box_drawer.hpp"
#include "grid_drawer.hpp"
#include "imu_capture.hpp"
#include "stream_server.hpp"
#include "viewer.hpp"

namespace fs = std::filesystem;

using namespace marionette::data_source;

const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;

// JSON serialization helpers for glm types
namespace glm {
static void to_json(nlohmann::json &j, const glm::quat &v) { j = {v.x, v.y, v.z, v.w}; }
}  // namespace glm

// imu_server_viewer - GUI window for IMU visualization
struct imu_server_viewer : public window_base {
  std::shared_ptr<azimuth_elevation> view_controller;
  axis_drawer axis_drawer_;
  box_drawer box_drawer_;
  grid_drawer grid_drawer_;
  std::mutex mtx;
  std::vector<glm::mat3> orientations;
  std::vector<glm::vec3> positions;
  bool drawer_initialized;

  imu_server_viewer()
      : window_base("IMU Server - IMU Viewer", SCREEN_WIDTH, SCREEN_HEIGHT),
        drawer_initialized(false) {}

  virtual void initialize() override {
    window_base::initialize();
    view_controller =
        std::make_shared<azimuth_elevation>(glm::u32vec2(0, 0), glm::u32vec2(width, height));
    view_controller->set_radius(5.0f);
    view_controller->set_translation(glm::vec3(0.0f, 0.0f, 0.0f));
  }

  virtual void on_close() override {
    std::lock_guard<std::mutex> lock(mtx);
    window_manager::get_instance()->exit();
    window_base::on_close();
  }

  virtual void on_scroll(double x, double y) override {
    if (view_controller) {
      view_controller->scroll(x, y);
    }
  }

  glm::mat4 pvw;

#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

  void set_camera(float pos_x, float pos_y, float pos_z, float target_x, float target_y,
                  float target_z) {
    float fovy = 45.0f;
    float aspect = (float)(width) / height;
    float near_plane = 0.01f;
    float far_plane = 1000.0f;
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    glm::mat4 proj = glm::perspective(fovy, aspect, near_plane, far_plane);
    glm::mat4 view =
        glm::lookAt(glm::vec3(pos_x, pos_y, pos_z), glm::vec3(target_x, target_y, target_z), up);
    glm::mat4 world = glm::identity<glm::mat4>();
    pvw = proj * view * world;
  }

  virtual void show() override {
    if (!gladLoadGL()) {
      spdlog::error("Failed to load OpenGL extensions!");
      exit(-1);
    }
    window_base::show();
  }

  virtual void update() override {
    if (!drawer_initialized) {
      axis_drawer_.initialize();
      box_drawer_.initialize();
      grid_drawer_.initialize();
      drawer_initialized = true;
    }

    const auto mouse = mouse_state::get_mouse_state(handle);
    view_controller->update(mouse);
    float radius = view_controller->get_radius() * 0.2f;
    glm::vec3 forward(0.f, 0.f, 1.f);
    const auto target_pos = glm::vec3(-view_controller->get_translation_matrix()[3]);
    glm::vec3 view_pos =
        target_pos +
        glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);

    set_camera(view_pos.x, view_pos.y, view_pos.z, target_pos.x, target_pos.y, target_pos.z);

    // Draw grid
    grid_drawer_.draw(pvw);

    // Draw IMU sensors
    for (size_t i = 0; i < orientations.size(); i++) {
      glm::mat4 box_orientation;
      glm::mat4 box_position;
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (i < orientations.size()) {
          const auto orientation = orientations[i];
          const auto position = positions[i];
          box_orientation = orientation;
          box_position = glm::translate(position);
        }
      }

      float box_scale = 0.01f;
      float axis_scale = 0.1f;

      axis_drawer_.draw(pvw * box_position * box_orientation *
                        glm::scale(glm::vec3(axis_scale, axis_scale, axis_scale)));
    }
  }

  void update_imu_data(const std::vector<glm::mat3> &new_orientations,
                       const std::vector<glm::vec3> &new_positions) {
    std::lock_guard<std::mutex> lock(mtx);
    orientations = new_orientations;
    positions = new_positions;
  }
};

int main(int argc, char *argv[]) {
  spdlog::set_level(spdlog::level::info);
  spdlog::info("IMU Server Application");

  // Initialize window manager
  const auto win_mgr = window_manager::get_instance();
  win_mgr->initialize();

  // Create viewer window
  const auto viewer = std::make_shared<imu_server_viewer>();
  const auto rendering_th = std::make_shared<rendering_thread>();
  rendering_th->start(viewer.get());

  // Start gRPC streaming server
  marker_stream_server server;
  server.run();
  spdlog::info("gRPC server started on port 50052");

  // List available serial ports
  spdlog::info("Available serial ports:");
  for (const auto &name : serial_port::get_serial_port_names()) {
    spdlog::info("  - {}", name);
  }

  // Determine serial port to use
  std::string port_name;
  if (argc > 1) {
    port_name = argv[1];
  } else {
#ifdef _WIN32
    port_name = "COM3";  // Default Windows port
#else
    port_name = "/dev/ttyUSB0";  // Default Linux port
#endif
  }
  spdlog::info("Opening serial port: {}", port_name);

  // Optional: Create data directory for saving frames
  const std::string data_dir = "../data/capture";
  if (!fs::exists(data_dir)) {
    fs::create_directories(data_dir);
    spdlog::info("Created data directory: {}", data_dir);
  }

  // Open IMU device and start capturing in separate thread
  auto capture = std::make_shared<imu_capture>();

  std::thread capture_thread([capture, port_name, &server, &viewer]() {
    try {
      capture->open(port_name);
      spdlog::info("Serial port opened successfully");
    } catch (const std::exception &e) {
      spdlog::error("Failed to open serial port: {}", e.what());
      return;
    }

    // Start capture loop
    spdlog::info("Starting capture loop...");
    capture->start([capture, &server, &viewer](const imu_capture::pose_frame &frame) {
      // Print frame info to console
      std::cout << (uint64_t)(frame.timestamp * 1000);
      for (int i = 0; i < NUM_SENSORS; i++) {
        std::cout << " | " << (int)frame.poses[i].accel_status << ","
                  << (int)frame.poses[i].gyro_status << "," << (int)frame.poses[i].mag_status;
      }
      for (int i = 0; i < NUM_SENSORS; i++) {
        std::cout << " | " << glm::to_string(frame.poses[i].orientation);
      }
      std::cout << " |" << std::endl;

      // Prepare data for visualization
      std::vector<glm::quat> orientations;
      std::vector<glm::mat3> orientation_mats;
      std::vector<glm::vec3> positions;

      for (int i = 0; i < NUM_SENSORS; i++) {
        orientations.push_back(frame.poses[i].orientation);

        // Convert quaternion to matrix for visualization
        glm::mat3 orient_mat = glm::mat3_cast(frame.poses[i].orientation);
        orientation_mats.push_back(orient_mat);

        // Position sensors in a line for visualization
        positions.push_back(glm::vec3(i * 0.15f, 0.0f, 0.0f));
      }

      // Update viewer
      viewer->update_imu_data(orientation_mats, positions);

      // Stream data to connected clients via gRPC
      server.push_frame(orientations);
    });
  });

  // Main rendering loop
  while (!win_mgr->should_close()) {
    win_mgr->handle_event();
  }

  // Cleanup
  spdlog::info("Shutting down...");

  // Stop capture thread
  if (capture->is_running()) {
    spdlog::info("Stopping capture...");
    capture->stop();
  }

  if (capture_thread.joinable()) {
    spdlog::info("Waiting for capture thread...");
    capture_thread.join();
  }

  // Stop rendering
  rendering_th->stop();
  viewer->destroy();

  // Stop gRPC server
  server.stop();

  // Terminate window manager
  win_mgr->terminate();

  spdlog::info("Shutdown complete");
  return 0;
}
