// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <atomic>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "axis_drawer.hpp"
#include "bone_drawer.hpp"
#include "box_drawer.hpp"
#include "drawer2d.hpp"
#include "finger_pose_estimation.hpp"
#include "finger_tracker.hpp"
#include "grid_drawer.hpp"
#include "imu_playback_stream.hpp"
#include "model.hpp"
#include "model_drawer.hpp"
#include "remote_sensor_stream.hpp"
#include "sphere_drawer.hpp"
#include "widget_drawer.hpp"

#ifdef _WIN32
#else
#include <signal.h>
#include <unistd.h>
#endif

#include "viewer.hpp"

using namespace marionette::core;
using namespace marionette::tracking;
using namespace marionette::data_source;
using namespace marionette::pose_estimation;

const int SCREEN_WIDTH = 1680;
const int SCREEN_HEIGHT = 1050;

class rect_selector {
  mouse_state previous_state;
  glm::vec2 begin_pos;

 public:
  void update(mouse_state mouse, const std::function<void(glm::vec2, glm::vec2)>& on_selected,
              const std::function<void(glm::vec2, glm::vec2)>& on_selecting) {
    auto mouse_x = static_cast<int>(mouse.x);
    auto mouse_y = static_cast<int>(mouse.y);
    if (mouse.left_button == GLFW_PRESS) {
      if (previous_state.left_button == GLFW_RELEASE) {
        begin_pos = glm::vec2(mouse_x, mouse_y);
      }
      on_selecting(begin_pos, glm::vec2(mouse.x, mouse.y));
    }
    if (mouse.left_button == GLFW_RELEASE) {
      if (previous_state.left_button == GLFW_PRESS) {
        on_selected(begin_pos, glm::vec2(mouse.x, mouse.y));
        begin_pos = glm::vec2(0, 0);
      }
    }

    previous_state = mouse;
  }
};

struct imu_viewer : public window_base {
  std::shared_ptr<azimuth_elevation> view_controller;
  rect_selector rect_selector_;
  sphere_drawer sphere_drawer_;
  box_drawer box_drawer_;
  bone_drawer bone_drawer_;
  bone_drawer bone_drawer_r_;
  bone_drawer bone_drawer_g_;
  bone_drawer bone_drawer_b_;
  grid_drawer grid_drawer_;
  axis_drawer axis_drawer_;
  drawer2d drawer2d_;
  widget_drawer widget_drawer_;
  std::mutex mtx;
  std::vector<glm::vec3> markers;
  glm::u8vec4 color;
  glm::mat4 world;

  int selected_index = -1;
  bool show_r, show_g, show_b;

  bool drawer_initialized;

  std::map<std::string, glm::mat4> poses;

  imu_viewer()
      : window_base("IMR Viewer", SCREEN_WIDTH, SCREEN_HEIGHT),
        sphere_drawer_(36, 18, false),
        drawer_initialized(false),
        bone_drawer_r_(glm::u8vec4(255, 0, 0, 255)),
        bone_drawer_g_(glm::u8vec4(0, 255, 0, 255)),
        bone_drawer_b_(glm::u8vec4(0, 0, 255, 255)) {
    show_r = true;
    show_g = true;
    show_b = true;

    glm::mat4 basis(1.f);
    basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
    basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
    basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

    world = basis;

    widget_drawer_.check_r_changed = [this](bool show) { show_r = show; };
    widget_drawer_.check_g_changed = [this](bool show) { show_g = show; };
    widget_drawer_.check_b_changed = [this](bool show) { show_b = show; };
  }

  virtual void initialize() override {
    window_base::initialize();
    view_controller =
        std::make_shared<azimuth_elevation>(glm::u32vec2(0, 0), glm::u32vec2(width, height));
    view_controller->set_radius(5.0f);
    view_controller->set_translation(glm::vec3(0.0f, -1.0f, 0.0f));
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

  void set_camera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ) {
    float fovy = 45.0f;
    float aspect = (float)(width) / height;
    float near_plane = 0.01f;
    float far_plane = 1000.0f;
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    // set viewport to be the entire window
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    glm::mat4 proj = glm::perspective(fovy, aspect, near_plane, far_plane);
    glm::mat4 view =
        glm::lookAt(glm::vec3(posX, posY, posZ), glm::vec3(targetX, targetY, targetZ), up);
    glm::mat4 world = glm::identity<glm::mat4>();
    pvw = proj * view * world;
  }

  virtual void show() override {
    if (!gladLoadGL()) {
      printf("Failed to load OpenGL extensions!\n");
      exit(-1);
    }

    window_base::show();
  }

  virtual void update() override {
    if (handle == nullptr) {
      return;
    }

    if (!drawer_initialized) {
      sphere_drawer_.initialize();
      box_drawer_.initialize();
      axis_drawer_.initialize();
      grid_drawer_.initialize();
      widget_drawer_.initialize(handle, SCREEN_WIDTH, SCREEN_HEIGHT);
      drawer2d_.initialize();
      bone_drawer_.initialize();
      bone_drawer_r_.initialize();
      bone_drawer_g_.initialize();
      bone_drawer_b_.initialize();

      drawer_initialized = true;
    }

    const auto on_selected = [this](glm::vec2 beg, glm::vec2 end) {
      glm::vec2 rect_min(std::min(beg.x, end.x), std::min(beg.y, end.y));
      glm::vec2 rect_max(std::max(beg.x, end.x), std::max(beg.y, end.y));

      const auto clip_pos =
          glm::vec2(rect_min.x / SCREEN_WIDTH * 2 - 1, rect_max.y / SCREEN_HEIGHT * -2 + 1);
      const auto clip_size = glm::vec2(std::abs(rect_max.x - rect_min.x) / SCREEN_WIDTH * 2,
                                       std::abs(rect_max.y - rect_min.y) / SCREEN_HEIGHT * 2);

      selected_index = -1;
      widget_drawer_.selected_name = "";
    };
    const auto on_selecting = [this](glm::vec2 beg, glm::vec2 end) {
      drawer2d_.draw_rect(
          glm::vec2(beg.x / SCREEN_WIDTH, beg.y / SCREEN_HEIGHT),
          glm::vec2((end.x - beg.x) / SCREEN_WIDTH, (end.y - beg.y) / SCREEN_HEIGHT),
          glm::vec4(1, 0, 0, 1));
    };

    const auto mouse = mouse_state::get_mouse_state(handle);
    rect_selector_.update(mouse, on_selected, on_selecting);

    view_controller->update(mouse);
    float radius = view_controller->get_radius() * 0.2f;
    glm::vec3 forward(0.f, 0.f, 1.f);
    const auto target_pos = glm::vec3(-view_controller->get_translation_matrix()[3]);
    glm::vec3 view_pos =
        target_pos +
        glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);

    set_camera(view_pos.x, view_pos.y, view_pos.z, target_pos.x, target_pos.y, target_pos.z);

    std::map<std::string, glm::mat4> tmp_poses;
    {
      std::lock_guard<std::mutex> lock(mtx);
      tmp_poses = poses;
    }

    for (auto iter = tmp_poses.begin(); iter != tmp_poses.end(); iter++) {
      const auto name = iter->first;
      const auto transform = iter->second;

      if (name.find("R.") < name.size()) {
        continue;
      }

      if (name.find("Proximal.R") < name.size()) {
        bone_drawer_r_.draw(pvw * glm::translate(glm::vec3(0, 0, 0)) * transform *
                            glm::scale(glm::vec3(0.1f, 0.02f, 0.1f)));
      } else if (name.find("Intermediate.R") < name.size()) {
        bone_drawer_g_.draw(pvw * glm::translate(glm::vec3(0, 0, 0)) * transform *
                            glm::scale(glm::vec3(0.1f, 0.02f, 0.1f)));
      } else {
        bone_drawer_.draw(pvw * glm::translate(glm::vec3(0, 0, 0)) * transform *
                          glm::scale(glm::vec3(0.1f, 0.02f, 0.1f)));
      }
    }

    grid_drawer_.draw(pvw);
    widget_drawer_.draw();
  }

  virtual void on_char(unsigned int codepoint) override {
    widget_drawer_.on_char(handle, codepoint);
  }
};

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic_bool exit_flag(false);

static void shutdown() {
  std::for_each(std::rbegin(on_shutdown_handlers), std::rend(on_shutdown_handlers),
                [](auto handler) { handler(); });
  exit_flag.store(true);
}

static void sigint_handler(int) {
  shutdown();
  exit(0);
}

int imu_viewer_main() {
#if 1
  imu_playback_stream data_stream("../data/capture_20230416", 100);
#else
  remote_sensor_stream data_stream("192.168.10.105:50052");
#endif

  const auto win_mgr = window_manager::get_instance();
  win_mgr->initialize();

  on_shutdown_handlers.push_back([win_mgr]() { win_mgr->terminate(); });

  const auto viewer = std::make_shared<imu_viewer>();

  const auto rendering_th = std::make_shared<rendering_thread>();
  rendering_th->start(viewer.get());

  glm::dvec3 pos(0.0);
  double heading_vel = 0;
  uint64_t last_time_us = 0;

  model_data model;
  model.load("../data/TrackingModel.json");

  std::map<std::string, glm::mat4> poses;

  for (std::size_t j = 0; j < model.bones.size(); j++) {
    const auto pose = model.bones[j].pose;
    poses.insert(std::make_pair(model.bones[j].name, pose));
  }
  const auto recv_data_callback = [&viewer, &pos, &heading_vel, &last_time_us, &model,
                                   &poses](const std::vector<glm::quat>& orientations) {
    std::lock_guard<std::mutex> lock(viewer->mtx);

    finger_tracker tracker(model);

    std::vector<std::string> sensors = {"Cube.004", "Cube.003", "Cube.002",
                                        "Cube.001", "Cube.005", "Cube"};
    std::vector<std::string> bones = {"Little Distal.R", "Ring Distal.R",  "Middle Distal.R",
                                      "Index Distal.R",  "Thumb Distal.R", "hand.R"};
    glm::mat3 base_orientation;
    glm::vec3 base_position;
    {
      const auto i = 5;
      const auto obj = std::find_if(model.bones.begin(), model.bones.end(),
                                    [&](const auto& obj) { return obj.name == bones[i]; });
      if (obj != model.bones.end()) {
        glm::mat3 orientation = obj->pose;
        for (size_t k = 0; k < 3; k++) {
          orientation[k] = glm::normalize(orientation[k]);
        }
        glm::vec3 position = obj->pose[3];
        base_orientation = orientation;
        base_position = position;
      }
    }

    {
      for (size_t i = 0; i < orientations.size(); i++) {
        const auto obj = std::find_if(model.objects.begin(), model.objects.end(),
                                      [&](const auto& obj) { return obj.name == sensors[i]; });
        if (obj != model.objects.end()) {
          glm::mat3 default_sensor_orientation = obj->orientation;
          for (size_t k = 0; k < 3; k++) {
            default_sensor_orientation[k] = glm::normalize(default_sensor_orientation[k]);
          }
          glm::vec3 default_sensor_position = obj->position;

          glm::mat4 default_bone_pose(1.0f);
          {
            const auto obj = std::find_if(model.bones.begin(), model.bones.end(),
                                          [&](const auto& obj) { return obj.name == bones[i]; });
            if (obj != model.bones.end()) {
              glm::mat3 orientation = obj->pose;
              for (size_t k = 0; k < 3; k++) {
                orientation[k] = glm::normalize(orientation[k]);
              }
              glm::vec3 position = obj->pose[3];
              default_bone_pose[0] = glm::vec4(orientation[0], 0.0f);
              default_bone_pose[1] = glm::vec4(orientation[1], 0.0f);
              default_bone_pose[2] = glm::vec4(orientation[2], 0.0f);
              default_bone_pose[3] = glm::vec4(position, 1.0f);
            }
          }

          glm::mat4 default_bone_pose0(1.0f);
          {
            const auto i = 5;
            const auto obj = std::find_if(model.bones.begin(), model.bones.end(),
                                          [&](const auto& obj) { return obj.name == bones[i]; });
            if (obj != model.bones.end()) {
              glm::mat3 orientation = obj->pose;
              for (size_t k = 0; k < 3; k++) {
                orientation[k] = glm::normalize(orientation[k]);
              }
              glm::vec3 position = obj->pose[3];
              default_bone_pose0[0] = glm::vec4(orientation[0], 0.0f);
              default_bone_pose0[1] = glm::vec4(orientation[1], 0.0f);
              default_bone_pose0[2] = glm::vec4(orientation[2], 0.0f);
              default_bone_pose0[3] = glm::vec4(position, 1.0f);
            }
          }

          const auto sensor_orientation = glm::toMat4(glm::normalize(glm::quat(
              orientations[i].w, orientations[i].x, orientations[i].z, -orientations[i].y)));
          const auto sensor_orientation0 = glm::toMat4(glm::normalize(glm::quat(
              orientations[5].w, orientations[5].x, orientations[5].z, -orientations[5].y)));

          const auto bone_orientation = sensor_orientation * tracker.sensor_to_bone[i];
          const auto bone_orientation0 = sensor_orientation0 * tracker.sensor_to_bone[5];

          const auto sensor_pose0 =
              glm::mat4(sensor_orientation0[0], sensor_orientation0[1], sensor_orientation0[2],
                        glm::vec4(0.2 * (5 + 1), 0, 0, 1.0));
          const auto sensor_pose =
              glm::mat4(sensor_orientation[0], sensor_orientation[1], sensor_orientation[2],
                        glm::vec4(0.2 * (i + 1), 0, 0, 1.0));
          const auto bone_pose0 = sensor_pose0 * tracker.sensor_to_bone[5];
          const auto bone_pose = sensor_pose * tracker.sensor_to_bone[i];

          const auto bone_position2 = default_bone_pose[3];
          const auto bone_orientation2 = default_bone_pose0 * glm::inverse(bone_pose0) * bone_pose;

          auto bone = std::find_if(model.bones.begin(), model.bones.end(),
                                   [&](const auto& bone) { return bone.name == bones[i]; });
          if (bone != model.bones.end()) {
            poses[bone->name] = glm::mat4(bone_orientation2[0], bone_orientation2[1],
                                          bone_orientation2[2], bone_position2);
          }
        }
      }

      estimate_finger_pose(poses, model);

      viewer->poses = poses;
    }
  };

  on_shutdown_handlers.push_back([rendering_th, viewer]() {
    rendering_th->stop();
    viewer->destroy();
  });

  std::thread stream_th([&data_stream, &recv_data_callback]() {
    data_stream.subscribe_quat("", recv_data_callback);
  });

  while (!win_mgr->should_close()) {
    win_mgr->handle_event();
  }

  shutdown();

  return 0;
}

int main() { return imu_viewer_main(); }
