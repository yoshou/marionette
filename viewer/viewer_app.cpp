#include <string>
#include <fstream>
#include <iostream>
#ifdef _WIN32
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <nlohmann/json.hpp>
#include <GL/glew.h>
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

        // set perspective viewing frustum
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(fovy, aspect, near_z, far_z); // FOV, AspectRatio, NearClip, FarClip

        // switch to modelview matrix in order to set scene
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(posX, posY, posZ, targetX, targetY, targetZ, up.x, up.y, up.z); // eye(x,y,z), focal(x,y,z), up(x,y,z)

        glm::mat4 proj = glm::perspective(fovy, aspect, near_z, far_z);
        glm::mat4 view = glm::lookAt(glm::vec3(posX, posY, posZ), glm::vec3(targetX, targetY, targetZ), up);
        glm::mat4 world = glm::identity<glm::mat4>();
        pvw = proj * view * world;
    }

    virtual void update() override
    {
        if (handle == nullptr)
        {
            return;
        }

        if (!model)
        {
            GLenum err = glewInit();
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

static std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance)
{
    const auto &keyframe_clusters = model_instance.clusters;
    std::vector<glm::mat4> poses;
    for (std::size_t i = 0; i < keyframe_clusters.size(); i++)
    {
        const auto &cluster = keyframe_clusters[i];

        const auto pose = transform_to_target_pose(cluster.cluster, cluster.fit_result);
        poses.push_back(pose);
    }

    return poses;
}

static std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance, const clusters_transform_params &params)
{
    const auto &keyframe_clusters = model_instance.clusters;
    std::vector<glm::mat4> poses;
    for (std::size_t i = 0; i < keyframe_clusters.size(); i++)
    {
        const auto &cluster = keyframe_clusters[i];

        // const auto rotation = &params.mutable_rotations[i * 3];
        const auto rotation = &params.mutable_quat_rotations[i * 4];
        const auto translation = &params.mutable_translations[i * 3];

        const auto transform_angle_axis = [](glm::mat4 m, const double *axis_angle, const double *translation)
        {
            glm::vec3 axis_angle_vec(static_cast<float>(axis_angle[0]),
                                     static_cast<float>(axis_angle[1]),
                                     static_cast<float>(axis_angle[2]));

            glm::vec3 trans_vec(static_cast<float>(translation[0]),
                                static_cast<float>(translation[1]),
                                static_cast<float>(translation[2]));

            const auto angle = glm::length(axis_angle_vec);

            if (angle > std::numeric_limits<float>::epsilon())
            {
                const auto axis = glm::normalize(axis_angle_vec);
                const auto quat = glm::angleAxis(angle, axis);

                return glm::translate(trans_vec) * glm::toMat4(quat) * m;
            }
            else
            {
                return glm::translate(trans_vec) * m;
            }
        };

        const auto transform_quat = [](glm::mat4 m, const double* rotation, const double* translation)
        {
            glm::quat quat(static_cast<float>(rotation[0]),
                static_cast<float>(rotation[1]),
                static_cast<float>(rotation[2]),
                static_cast<float>(rotation[3]));

            glm::vec3 trans(static_cast<float>(translation[0]),
                static_cast<float>(translation[1]),
                static_cast<float>(translation[2]));

            return glm::translate(trans) * glm::toMat4(quat) * m;
        };

        const auto pose = transform_to_target_pose(cluster.cluster, cluster.fit_result);
        const auto updated_pose = transform_quat(pose, rotation, translation);
        
        poses.push_back(updated_pose);
    }

    return poses;
}

class model_detection_worker
{
    std::unique_ptr<std::thread> th;
    std::atomic_bool running;
    std::mutex mtx;

    model_data model;
    frame_data_t frame;

    std::atomic_bool is_callback_valid;
    std::function<void(const model_instance_data&, std::uint32_t, const frame_data_t&)> callback;
    std::mutex callback_mtx;

    void process()
    {
        running = true;
        while (running)
        {
            frame_data_t current_frame;
            {
                std::lock_guard lock(mtx);
                current_frame = frame;
            }

            std::cout << "=============== Detecting Frame : " << current_frame.frame_number << " =================" << std::endl;
            const auto model_instance = detect_model(model, current_frame);

            if (model_instance.clusters.size() > 0)
            {
                if (is_callback_valid)
                {
                    callback(model_instance, 0, current_frame);
                }
            }
        }
    }
public:
    model_detection_worker(const model_data& model)
        : running(false), model(model), is_callback_valid(false)
    {}

    void update_frame(const frame_data_t& frame)
    {
        std::lock_guard lock(mtx);
        this->frame = frame;
    }

    void set_callback(std::function<void(const model_instance_data &, std::uint32_t, const frame_data_t&)> func)
    {
        callback = func;
        is_callback_valid = true;
    }

    void start()
    {
        th.reset(new std::thread(&model_detection_worker::process, this));
    }

    void stop()
    {
        running = false;
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

class model_tracking_worker
{
    std::unique_ptr<std::thread> th;
    std::atomic_bool running;
    std::mutex mtx;

    model_data model;
    model_data instanced_model;
    model_instance_data model_instance;
    std::shared_ptr<frame_cursor> frame_cursor;

    mutable std::mutex poses_mtx;
    std::map<std::string, glm::mat4> poses;

    void process()
    {
        {
            auto& clusters = model_instance.clusters;
            const auto frame = frame_cursor->get_frame();

            point_cloud target_cloud(frame.markers);
            target_cloud.build_index();
            for (auto& cluster : clusters)
            {
                cluster.target = find_target_points(cluster.cluster, cluster.fit_result, target_cloud);
            }

            for (std::size_t i = 0; i < clusters.size(); i++)
            {
                const auto sources = find_source_points(clusters[i].cluster, clusters[i].fit_result, target_cloud);
                instanced_model.clusters[clusters[i].cluster.name].points = sources;
            }
        }

        motion_tracker tracker;

        running = true;
        while (running)
        {
            frame_cursor->wait_next(running);
            frame_cursor = frame_cursor->get_next();

            std::cout << "=============== Tracking Frame : " << frame_cursor->get_frame().frame_number << " =================" << std::endl;

            tracker.track_frame(instanced_model, frame_cursor->get_frame(), model_instance);

            std::map<std::string, glm::mat4> poses;
            const auto intraframe_clusters = compute_pose(tracker.keyframe_clusters, tracker.params);
            for (std::size_t j = 0; j < intraframe_clusters.size(); j++)
            {
                const auto& fit_result = intraframe_clusters[j];

                const auto pose = fit_result;
                poses.insert(std::make_pair(tracker.keyframe_clusters.clusters[j].cluster.name, pose));
            }

            {
                std::lock_guard lock(poses_mtx);
                this->poses = poses;
            }
#if 0
            {
                nlohmann::json j;

                j["poses"] = poses;
                j["points"] = frame_cursor->get_frame().markers;

                std::ofstream ofs;
                ofs.open("pose_" + std::to_string(frame_cursor->get_frame().frame_number) + ".json", std::ios::out);
                ofs << j.dump(2);
            }
#endif
        }
    }

public:
    model_tracking_worker(const model_data& model, const model_instance_data& model_instance, const std::shared_ptr<::frame_cursor>& frame_cursor)
        : running(false), model(model), instanced_model(model), model_instance(model_instance), frame_cursor(frame_cursor)
    {
    }

    void start()
    {
        th.reset(new std::thread(&model_tracking_worker::process, this));
    }

    void stop()
    {
        running = false;
        if (th && th->joinable())
        {
            th->join();
        }
    }

    std::map<std::string, glm::mat4> get_poses() const
    {
        std::lock_guard lock(poses_mtx);
        return poses;
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

    //model_optimizer model_optim(model);

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

        point_cloud_logger::get_logger().frame = frame.frame_number;

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
#if 0
        {
            point_cloud_debug_drawer debug_point;
            for (const auto& marker : frame.markers)
            {
                debug_point.add(marker, glm::u8vec3(255, 0, 0));
            }
            if (frame.markers.size() > 0)
            {
                debug_point.save("temp_" + std::to_string(frame.frame_number) + ".pcd");
            }
        }
#endif
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

