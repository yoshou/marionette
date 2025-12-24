#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <nlohmann/json.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "glm_json_ext.hpp"
#include "viewer.hpp"
#include "sphere_drawer.hpp"
#include "grid_drawer.hpp"
#include "widget_drawer.hpp"
#include "model_drawer.hpp"
#include "bone_drawer.hpp"
#include "drawer2d.hpp"
#include "debug.hpp"

#include "playback_stream.hpp"
#include "remote_sensor_stream.hpp"

#include "model.hpp"
#include "model_detector.hpp"
#include "frame_queue.hpp"
#include "motion_tracker.hpp"
#include "retarget.hpp"
#include "pose_computation.hpp"
#include "model_detection_worker.hpp"
#include "model_tracking_worker.hpp"

#define PLAYBACK 1

const int SCREEN_WIDTH = 1680;
const int SCREEN_HEIGHT = 1050;

class rect_selector
{
    mouse_state previous_state;
    glm::vec2 begin_pos;

public:
    void update(mouse_state mouse, const std::function<void(glm::vec2, glm::vec2)> &on_selected, const std::function<void(glm::vec2, glm::vec2)> &on_selecting)
    {
        auto mouse_x = static_cast<int>(mouse.x);
        auto mouse_y = static_cast<int>(mouse.y);
        if (mouse.left_button == GLFW_PRESS)
        {
            if (previous_state.left_button == GLFW_RELEASE)
            {
                begin_pos = glm::vec2(mouse_x, mouse_y);
            }
            on_selecting(begin_pos, glm::vec2(mouse.x, mouse.y));
        }
        if (mouse.left_button == GLFW_RELEASE)
        {
            if (previous_state.left_button == GLFW_PRESS)
            {
                on_selected(begin_pos, glm::vec2(mouse.x, mouse.y));
                begin_pos = glm::vec2(0, 0);
            }
        }

        previous_state = mouse;
    }
};

struct ir_viewer : public window_base
{
    std::shared_ptr<azimuth_elevation> view_controller;
    sphere_drawer sphere_drawer_;
    grid_drawer grid_drawer_;
    bone_drawer bone_drawer_;
    widget_drawer widget_drawer_;
    std::mutex mtx;
    std::vector<glm::vec3> markers;
    std::map<std::string, glm::mat4> poses;
    glm::u8vec4 color;
    glm::mat4 world;
    std::shared_ptr<model_drawer> model;
    retarget_model tpose_model;
    glm::mat4 pvw;

    ir_viewer()
        : window_base("Marker Viewer", SCREEN_WIDTH, SCREEN_HEIGHT), sphere_drawer_(36, 18, false)
    {
        world = glm::mat4(1.0f);

        tpose_model.load("../data/DefaultPose.json");
    }

    virtual void initialize() override
    {
        window_base::initialize();
        view_controller = std::make_shared<azimuth_elevation>(glm::u32vec2(0, 0), glm::u32vec2(width, height));
    }

    virtual void on_close() override
    {
        std::lock_guard<std::mutex> lock(mtx);
        window_manager::get_instance()->exit();
        window_base::on_close();
    }

    virtual void on_scroll(double x, double y) override
    {
        if (view_controller)
        {
            view_controller->scroll(x, y);
        }
    }

    void set_camera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ)
    {
        float fovy = 45.0f;
        float aspect = (float)(width) / height;
        float near_z = 0.1f;
        float far_z = 1000.0f;
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        // set viewport to be the entire window
        glViewport(0, 0, (GLsizei)width, (GLsizei)height);

        glm::mat4 proj = glm::perspective(fovy, aspect, near_z, far_z);
        glm::mat4 view = glm::lookAt(glm::vec3(posX, posY, posZ), glm::vec3(targetX, targetY, targetZ), up);
        glm::mat4 world = glm::identity<glm::mat4>();
        pvw = proj * view * world;
    }

    virtual void show() override
    {
        if (!gladLoadGL())
        {
            printf("Failed to load OpenGL extensions!\n");
            exit(-1);
        }

        window_base::show();
    }

    virtual void update() override
    {
        if (handle == nullptr)
        {
            return;
        }

        if (!model)
        {
            // model = std::make_shared<model_drawer>("../data/Alicia_VRM/Alicia_VRM/Alicia/VRM/AliciaSolid.vrm");
            model = std::make_shared<model_drawer>("../data/RAYNOS-chan-avatar_v1.0.2/RAYNOS-chan-avatar_v1.0.2/VRM/RAYNOS-chan_1.0.2.vrm");
            //model = std::make_shared<model_drawer>("../data/untitled.glb");

            sphere_drawer_.initialize();
            grid_drawer_.initialize();
            bone_drawer_.initialize();
            widget_drawer_.initialize(handle, SCREEN_WIDTH, SCREEN_HEIGHT);
        }

        view_controller->update(mouse_state::get_mouse_state(handle));
        float radius = view_controller->get_radius();
        glm::vec3 forward(0.f, 0.f, 1.f);
        glm::vec3 view_pos = glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);

        set_camera(view_pos.x, view_pos.y + 1.5, view_pos.z, 0, 1.5, 0);

        grid_drawer_.draw(pvw);

        const std::map<std::string, std::string> parents = {
            {"upper_arm.L", "Chest"},
            {"lower_arm.L", "upper_arm.L"},
            {"hand.L", "lower_arm.L"},

            {"upper_arm.R", "Chest"},
            {"lower_arm.R", "upper_arm.R"},
            {"hand.R", "lower_arm.R"},

            {"upper_leg.L", "Spine"},
            {"lower_leg.L", "upper_leg.L"},
            {"foot.L", "lower_leg.L"},

            {"upper_leg.R", "Spine"},
            {"lower_leg.R", "upper_leg.R"},
            {"foot.R", "lower_leg.R"},

            {"Chest", "Spine"},
        };

        {
            std::vector<glm::vec3> tmp_markers;
            std::map<std::string, glm::mat4> tmp_poses;
            glm::u8vec4 tmp_color;
            {
                std::lock_guard<std::mutex> lock(mtx);
                tmp_markers = markers;
                tmp_poses = poses;
                tmp_color = color;
            }

            const auto norm_poses = tpose_model.compute_normalized_pose(tmp_poses);
            std::map<std::string, glm::mat4> local_norm_poses;

            for (auto iter = tmp_poses.begin(); iter != tmp_poses.end(); iter++)
            {
                const auto name = iter->first;
                const auto transform = get_bone_local_transform(name, parents, norm_poses);
                local_norm_poses.insert(std::make_pair(name, transform));
            }

            const auto retargeted_pose = retarget(local_norm_poses, model->get_bone_transforms(), parents);
            model->set_bone_transforms(retargeted_pose);

            model->set_blend_weight("Joy", 1.0f);

            model->draw(pvw);
            for (const auto& name : model->get_bone_names())
            {
                const auto pose = model->get_bone_global_transform(name);
                //bone_drawer_.draw(pvw* pose* glm::scale(glm::vec3(1.f, 0.3f, 1.f)));
            }

            float lineColor[] = {
                std::clamp(tmp_color.r / 255.f, 0.f, 1.f),
                std::clamp(tmp_color.g / 255.f, 0.f, 1.f),
                std::clamp(tmp_color.b / 255.f, 0.f, 1.f), 1.f};

            for (const auto &marker : tmp_markers)
            {
                const auto pos = world * glm::vec4(marker, 1.0f);
                sphere_drawer_.drawWithLines(pvw * glm::translate(glm::vec3(pos)) * glm::scale(glm::vec3(0.01f, 0.01f, 0.01f)), lineColor);
            }

            for (auto iter = tmp_poses.begin(); iter != tmp_poses.end(); iter++)
            {
                const auto name = iter->first;
                const auto transform = get_bone_global_transform(name, parents, local_norm_poses);
                //bone_drawer_.draw(pvw *glm::translate(glm::vec3(0, 0, 1))* transform* glm::scale(glm::vec3(1.f, 0.3f, 1.f)));
            }
        }
        widget_drawer_.draw();
    }
};

static std::vector<std::function<void()>> on_shutdown_handlers;
static std::atomic_bool exit_flag(false);

static void shutdown()
{
    std::for_each(std::rbegin(on_shutdown_handlers), std::rend(on_shutdown_handlers), [](auto handler)
                  { handler(); });
    exit_flag.store(true);
}

static void sigint_handler(int)
{
    shutdown();
    exit(0);
}

int ir_viewer_main()
{
#if PLAYBACK
    const auto markers_directory = "../data/frames/";
    playback_stream data_stream(markers_directory, 90, 0);
#else
    remote_sensor_stream data_stream("192.168.10.105:50051");
#endif

    const auto win_mgr = window_manager::get_instance();
    win_mgr->initialize();

    on_shutdown_handlers.push_back([win_mgr]() {
        win_mgr->terminate();
    });

    const auto viewer = std::make_shared<ir_viewer>();

    const auto rendering_th = std::make_shared<rendering_thread>();
    rendering_th->start(viewer.get());
    
    model_data model;
    model.load("../data/TrackingModel.json");

    {
        const std::vector<std::tuple<std::string, float, float>> twist_bounds = {
            {"upper_leg.R", -90.f, 90.f},
            {"lower_leg.R", -45.f, 45.f},
            {"upper_leg.L", 0.f, 0.f},
            {"lower_leg.L", -45.f, 45.f},
            {"foot.R", 0.f, 0.f},
            {"foot.L", 0.f, 0.f},
            {"upper_arm.R", -90.f, 90.f},
            {"lower_arm.R", -90.f, 90.f},
            {"upper_arm.L", -90.f, 90.f},
            {"lower_arm.L", -90.f, 90.f},
            {"hand.R", 0.f, 0.f},
            {"hand.L", 0.f, 0.f},
            {"Chest", 0.f, 0.f},
            {"Spine", 0.f, 0.f},
            {"Neck", 0.f, 0.f},
        };

        for (const auto &[name, lb, ub] : twist_bounds)
        {
            auto &cluster = model.clusters.at(name);
            cluster.min_twist_angle = lb;
            cluster.max_twist_angle = ub;
        }
    }

    const auto world = glm::mat4(1.0f);

    model_detection_worker model_detection(model);

    std::mutex model_trackings_mtx;
    std::unordered_map<std::uint32_t, std::shared_ptr<model_tracking_worker>> model_trackings;

    const size_t max_frame_history = 1000;
    const auto frame_history = std::make_shared<frame_queue>(max_frame_history);

    model_detection.set_callback([&model, &model_trackings, &model_trackings_mtx, frame_history](const model_instance_data &model_instance, uint32_t id, const frame_data_t& frame)
                                 {
        std::lock_guard lock(model_trackings_mtx);
        if (model_trackings.find(id) == model_trackings.end())
        {
            const auto frame_found = frame_history->find(frame.frame_number);
            if (frame_found)
            {
                const auto tracking = std::make_shared<model_tracking_worker>(model, model_instance, frame_found);
                tracking->start();
                model_trackings.insert(std::make_pair(id, tracking));
            }
        }
        });

    model_detection.start();

    uint64_t frame_counter = 0;

    const auto recv_marker_callback = [&](const std::vector<glm::vec3> &markers)
    {
        frame_data_t frame;
        for (const auto marker : markers)
        {
            frame.markers.push_back(glm::vec3(world * glm::vec4(marker, 1.f)));
        }

        const std::vector<std::vector<glm::vec2>> pts;
        frame.points = pts;
        frame.frame_number = frame_counter++;

        marionette::utils::point_cloud_logger::get_logger().frame = frame.frame_number;

        frame_history->push(frame);

        model_detection.update_frame(frame);

        std::map<std::string, glm::mat4> poses;
        std::vector<glm::vec3> draw_markers;
        for (const auto marker : frame.markers)
        {
            draw_markers.push_back(glm::inverse(world) * glm::vec4(marker, 1.f));
        }

        glm::u8vec4 color(255, 0, 0, 255);
        if (model_trackings.size() > 0)
        {
            color = glm::u8vec4(0, 0, 255, 255);
            poses = model_trackings[0]->get_poses();
        }

        {
            std::lock_guard<std::mutex> lock(viewer->mtx);
            viewer->markers = draw_markers;
            viewer->color = color;
            viewer->poses = poses;
        }
    };

#if PLAYBACK
    std::thread stream_th([&data_stream, &recv_marker_callback]() {
        data_stream.subscribe_sphere("", recv_marker_callback);
    });
#else
    std::thread stream_th([&data_stream, &recv_marker_callback]() {
        data_stream.subscribe_sphere("", recv_marker_callback);
    });
#endif

    on_shutdown_handlers.push_back([rendering_th, viewer]() {
        rendering_th->stop();
        viewer->destroy();
    });

    while (!win_mgr->should_close())
    {
        win_mgr->handle_event();
    }

    model_detection.stop();
    {
        std::lock_guard lock(model_trackings_mtx);
        for (const auto& model_tracking : model_trackings)
        {
            model_tracking.second->stop();
        }
    }

    shutdown();

    return 0;
}

int main()
{
    return ir_viewer_main();
}

