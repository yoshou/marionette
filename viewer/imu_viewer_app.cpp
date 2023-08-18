#include <iostream>
#include <map>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <atomic>
#include <thread>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "sphere_drawer.hpp"
#include "box_drawer.hpp"
#include "axis_drawer.hpp"
#include "grid_drawer.hpp"
#include "bone_drawer.hpp"
#include "widget_drawer.hpp"
#include "model_drawer.hpp"
#include "drawer2d.hpp"
#include "model.hpp"

#ifdef _WIN32
#else
#include <signal.h>
#include <unistd.h>
#endif

#include "viewer.hpp"

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

struct imu_viewer : public window_base
{
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
    std::vector<glm::mat3> orientations;
    std::vector<glm::vec3> positions;

    int selected_index = -1;
    bool show_r, show_g, show_b;

    bool drawer_initialized;

    std::map<std::string, glm::mat4> poses;

    imu_viewer()
        : window_base("IMR Viewer", SCREEN_WIDTH, SCREEN_HEIGHT), sphere_drawer_(36, 18, false), drawer_initialized(false), bone_drawer_r_(glm::u8vec4(255, 0, 0, 255)), bone_drawer_g_(glm::u8vec4(0, 255, 0, 255)), bone_drawer_b_(glm::u8vec4(0, 0, 255, 255))
    {
        show_r = true;
        show_g = true;
        show_b = true;

        glm::mat4 basis(1.f);
        basis[0] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
        basis[1] = glm::vec4(0.f, 0.f, 1.f, 0.f);
        basis[2] = glm::vec4(0.f, 1.f, 0.f, 0.f);

        world = basis;

        widget_drawer_.check_r_changed = [this](bool show)
        {
            show_r = show;
        };
        widget_drawer_.check_g_changed = [this](bool show)
        {
            show_g = show;
        };
        widget_drawer_.check_b_changed = [this](bool show)
        {
            show_b = show;
        };
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

    glm::mat4 pvw;
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

    void set_camera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ)
    {
        float fovy = 45.0f;
        float aspect = (float)(width) / height;
        float near = 0.01f;
        float far = 1000.0f;
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        // set viewport to be the entire window
        glViewport(0, 0, (GLsizei)width, (GLsizei)height);

        // set perspective viewing frustum
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(fovy, aspect, near, far); // FOV, AspectRatio, NearClip, FarClip

        // switch to modelview matrix in order to set scene
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(posX, posY, posZ, targetX, targetY, targetZ, up.x, up.y, up.z); // eye(x,y,z), focal(x,y,z), up(x,y,z)

        glm::mat4 proj = glm::perspective(fovy, aspect, near, far);
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

        if (!drawer_initialized)
        {
            GLenum err = glewInit();

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

        const auto on_selected = [this](glm::vec2 beg, glm::vec2 end)
        {
            glm::vec2 rect_min(std::min(beg.x, end.x), std::min(beg.y, end.y));
            glm::vec2 rect_max(std::max(beg.x, end.x), std::max(beg.y, end.y));

            const auto clip_pos = glm::vec2(rect_min.x / SCREEN_WIDTH * 2 - 1, rect_max.y / SCREEN_HEIGHT * -2 + 1);
            const auto clip_size = glm::vec2(std::abs(rect_max.x - rect_min.x) / SCREEN_WIDTH * 2, std::abs(rect_max.y - rect_min.y) / SCREEN_HEIGHT * 2);

            selected_index = -1;
            widget_drawer_.selected_name = "";
        };
        const auto on_selecting = [this](glm::vec2 beg, glm::vec2 end)
        {
            drawer2d_.draw_rect(glm::vec2(beg.x / SCREEN_WIDTH, beg.y / SCREEN_HEIGHT), glm::vec2((end.x - beg.x) / SCREEN_WIDTH, (end.y - beg.y) / SCREEN_HEIGHT), glm::vec4(1, 0, 0, 1));
        };

        const auto mouse = mouse_state::get_mouse_state(handle);
        rect_selector_.update(mouse, on_selected, on_selecting);

        view_controller->update(mouse);
        float radius = view_controller->get_radius() * 0.2f;
        glm::vec3 forward(0.f, 0.f, 1.f);
        const auto target_pos = glm::vec3(-view_controller->get_translation_matrix()[3]);
        glm::vec3 view_pos = target_pos + glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);

        set_camera(view_pos.x, view_pos.y, view_pos.z, target_pos.x, target_pos.y, target_pos.z);

        for (size_t i = 0; i < orientations.size(); i++)
        {
            glm::mat4 box_orientation;
            glm::mat4 box_position;
            {
                std::lock_guard<std::mutex> lock(mtx);
                const auto orientation = orientations[i];
                const auto position = positions[i];
                box_orientation = orientation;
                box_position = glm::translate(position);
            }

            float box_scale = 0.01f;
            float axis_scale = 0.1f;

            //box_drawer_.draw(pvw * box_position * box_orientation * glm::scale(glm::vec3(box_scale, box_scale, box_scale)));
            axis_drawer_.draw(pvw * box_position * box_orientation * glm::scale(glm::vec3(axis_scale, axis_scale, axis_scale)));

        }

        std::map<std::string, glm::mat4> tmp_poses;
        glm::u8vec4 tmp_color;
        {
            std::lock_guard<std::mutex> lock(mtx);
            tmp_poses = poses;
        }

        for (auto iter = tmp_poses.begin(); iter != tmp_poses.end(); iter++)
        {
            const auto name = iter->first;
            const auto transform = iter->second;

            if (name.find("R.") < name.size())
            {
                continue;
            }

            if (name.find("Proximal.R") < name.size())
            {
                bone_drawer_r_.draw(pvw * glm::translate(glm::vec3(0, 0, 0)) * transform * glm::scale(glm::vec3(0.1f, 0.02f, 0.1f)));
            }
            else if (name.find("Intermediate.R") < name.size())
            {
                bone_drawer_g_.draw(pvw * glm::translate(glm::vec3(0, 0, 0)) * transform * glm::scale(glm::vec3(0.1f, 0.02f, 0.1f)));
            }
            //else if (name.find("Distal.R") < name.size())
            //{
             //   bone_drawer_b_.draw(pvw * glm::translate(glm::vec3(0, 0, 0)) * transform * glm::scale(glm::vec3(0.1f, 0.02f, 0.1f)));
            //}
            else
            {
                bone_drawer_.draw(pvw * glm::translate(glm::vec3(0, 0, 0)) * transform * glm::scale(glm::vec3(0.1f, 0.02f, 0.1f)));
            }
        }

        grid_drawer_.draw(pvw);
        widget_drawer_.draw();
    }

    virtual void on_char(unsigned int codepoint) override
    {
        widget_drawer_.on_char(handle, codepoint);
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

#include "qprobe_playback_stream.hpp"
#include "remote_sensor_stream.hpp"

class finger_tracker
{
    model_data model;

public:

    std::vector<glm::mat4> sensor_to_bone;
    finger_tracker(const model_data& model)
        : model(model)
    {
        initialize();
    }

    void initialize()
    {
        std::vector<std::string> sensors = { "Cube.004", "Cube.003" , "Cube.002" , "Cube.001" , "Cube.005" , "Cube" };
        std::vector<std::string> bones = { "Little Distal.R", "Ring Distal.R", "Middle Distal.R", "Index Distal.R", "Thumb Distal.R", "hand.R" };

        for (size_t i = 0; i < sensors.size(); i++)
        {
            const auto sensor = std::find_if(model.objects.begin(), model.objects.end(), [&](const auto& obj) { return obj.name == sensors[i];  });
            if (sensor == model.objects.end())
            {
                throw std::runtime_error("Invalid model");
            }
            const auto bone = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& obj) { return obj.name == bones[i];  });
            if (bone == model.bones.end())
            {
                throw std::runtime_error("Invalid model");
            }
            glm::mat4 sensor_pose = sensor->orientation;
            for (size_t k = 0; k < 3; k++)
            {
                sensor_pose[k] = glm::normalize(sensor_pose[k]);
            }
            sensor_pose[3] = glm::vec4(sensor->position, 1.f);
#if 0
            glm::quat sensor_orientation = glm::quat_cast(sensor_pose);
            const auto sensor_position = sensor_pose[3];
            sensor_orientation = glm::quat(sensor_orientation.w, sensor_orientation.x, sensor_orientation.z, -sensor_orientation.y);
            sensor_pose = glm::toMat4(sensor_orientation);
            sensor_pose[3] = sensor_position;
#endif

            glm::mat4 bone_pose = bone->pose;
            for (size_t k = 0; k < 3; k++)
            {
                bone_pose[k] = glm::normalize(bone_pose[k]);
            }
            //glm::quat bone_orientation = glm::quat_cast(bone_pose);
            //bone_orientation = glm::toMat4(glm::normalize(glm::quat(bone_orientation.w, bone_orientation.x, bone_orientation.z, -bone_orientation.y)));
            sensor_to_bone.push_back(glm::inverse(sensor_pose) * bone_pose);
        }
    }

    void track(const std::vector<glm::quat>& poses)
    {
    }
};

#include <glm/gtc/type_ptr.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct local_angle_error
{
    local_angle_error()
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        const T* const rotation,
        const T* const child_rotation,
        T* residuals) const
    {
#if 0
        Eigen::Map<const Eigen::Vector4<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Vector4<T>> q3(child_rotation);

        const Eigen::Vector4<T> diff = (q1 + q3) * 0.5 - q2;
#else
        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q3(child_rotation);

        const Eigen::Quaternion<T> q4 = Eigen::Quaternion<T>(q1).slerp(T(0.5), q3);
        const Eigen::Vector4<T> diff = Eigen::Vector4<T>(q4.x(), q4.y(), q4.z(), q4.w()) - q2;
#endif

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();
        residuals[3] = diff.w();

        return true;
    }

    static ceres::CostFunction* create()
    {
        return (new ceres::AutoDiffCostFunction<local_angle_error, 4, 4, 4, 4>(
            new local_angle_error()));
    }
};

struct local_angle_error2
{
    const glm::dquat parent_rotation;

    local_angle_error2(const glm::dquat& parent_rotation)
        : parent_rotation(parent_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const rotation,
        const T* const child_rotation,
        T* residuals) const
    {
        const T parent_rotation[] = { T(this->parent_rotation.x), T(this->parent_rotation.y), T(this->parent_rotation.z), T(this->parent_rotation.w)};

#if 0
        Eigen::Map<const Eigen::Vector4<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Vector4<T>> q3(child_rotation);

        const Eigen::Vector4<T> diff = (q1 + q3) * 0.5 - q2;
#else
        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q3(child_rotation);

        const Eigen::Quaternion<T> q4 = Eigen::Quaternion<T>(q1).slerp(T(0.5), q3);
        const Eigen::Vector4<T> diff = Eigen::Vector4<T>(q4.x(), q4.y(), q4.z(), q4.w()) - q2;
#endif

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();
        residuals[3] = diff.w();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& parent_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_angle_error2, 4, 4, 4>(
            new local_angle_error2(parent_rotation)));
    }
};

struct local_angle_error3
{
    const glm::dquat child_rotation;

    local_angle_error3(const glm::dquat& child_rotation)
        : child_rotation(child_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        const T* const rotation,
        T* residuals) const
    {
        const T child_rotation[] = { T(this->child_rotation.x), T(this->child_rotation.y), T(this->child_rotation.z), T(this->child_rotation.w)};

#if 0
        Eigen::Map<const Eigen::Vector4<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Vector4<T>> q3(child_rotation);

        const Eigen::Vector4<T> diff = (q1 + q3) * 0.5 - q2;
#else
        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q3(child_rotation);

        const Eigen::Quaternion<T> q4 = Eigen::Quaternion<T>(q1).slerp(T(0.5), q3);
        const Eigen::Vector4<T> diff = Eigen::Vector4<T>(q4.x(), q4.y(), q4.z(), q4.w()) - q2;
#endif

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();
        residuals[3] = diff.w();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& child_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_angle_error3, 4, 4, 4>(
            new local_angle_error3(child_rotation)));
    }
};

struct articulation_error
{
    glm::vec3 local_child_position;

    articulation_error(const glm::vec3& local_child_position)
        : local_child_position(local_child_position)
    {}

    template <typename T>
    bool operator()(
        const T* const parent_translation,
        const T* const parent_rotation,
        const T* const child_translation,
        T* residuals) const
    {
        T point[3] = {
            T(static_cast<double>(this->local_child_position.x)),
            T(static_cast<double>(this->local_child_position.y)),
            T(static_cast<double>(this->local_child_position.z)) };

        T p[3];
        ceres::QuaternionRotatePoint(parent_rotation, point, p);
        p[0] += parent_translation[0];
        p[1] += parent_translation[1];
        p[2] += parent_translation[2];

        residuals[0] = p[0] - child_translation[0];
        residuals[1] = p[1] - child_translation[1];
        residuals[2] = p[2] - child_translation[2];

        return true;
    }

    static ceres::CostFunction* create(const glm::vec3& local_child_positionn)
    {
        return (new ceres::AutoDiffCostFunction<articulation_error, 3, 3, 4, 3>(
            new articulation_error(local_child_positionn)));
    }
};

struct articulation_error2
{
    glm::vec3 local_child_position;

    articulation_error2(const glm::vec3& local_child_position)
        : local_child_position(local_child_position)
    {}

    template <typename T>
    bool operator()(
        const T* const child_translation,
        T* residuals) const
    {
        T p[3] = {
            T(static_cast<double>(this->local_child_position.x)),
            T(static_cast<double>(this->local_child_position.y)),
            T(static_cast<double>(this->local_child_position.z)) };

        residuals[0] = p[0] - child_translation[0];
        residuals[1] = p[1] - child_translation[1];
        residuals[2] = p[2] - child_translation[2];

        return true;
    }

    static ceres::CostFunction* create(const glm::vec3& local_child_positionn)
    {
        return (new ceres::AutoDiffCostFunction<articulation_error2, 3, 3>(
            new articulation_error2(local_child_positionn)));
    }
};

struct local_1dof_constraint_error
{
    local_1dof_constraint_error()
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        const T* const child_rotation,
        T* residuals) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q2(child_rotation);

        const Eigen::Vector3<T> axis(T(1.0), T(0.0), T(0.0));

        const Eigen::Vector3<T> v1 = q1 * axis;
        const Eigen::Vector3<T> v2 = q2 * axis;

        const Eigen::Vector3<T> diff = (v1 - v2) * T(100.0);

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();

        return true;
    }

    static ceres::CostFunction* create()
    {
        return (new ceres::AutoDiffCostFunction<local_1dof_constraint_error, 3, 4, 4>(
            new local_1dof_constraint_error()));
    }
};
struct local_1dof_constraint_error2
{
    const glm::dquat parent_rotation;

    local_1dof_constraint_error2(const glm::dquat& parent_rotation)
        : parent_rotation(parent_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const child_rotation,
        T* residuals) const
    {
        const T parent_rotation[] = { T(this->parent_rotation.x), T(this->parent_rotation.y), T(this->parent_rotation.z), T(this->parent_rotation.w) };

        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q2(child_rotation);

        const Eigen::Vector3<T> axis(T(1.0), T(0.0), T(0.0));

        const Eigen::Vector3<T> v1 = q1 * axis;
        const Eigen::Vector3<T> v2 = q2 * axis;

        const Eigen::Vector3<T> diff = v1 - v2;

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& parent_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_1dof_constraint_error2, 3, 4>(
            new local_1dof_constraint_error2(parent_rotation)));
    }
};
struct local_1dof_constraint_error3
{
    const glm::dquat child_rotation;

    local_1dof_constraint_error3(const glm::dquat& child_rotation)
        : child_rotation(child_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        T* residuals) const
    {
        const T child_rotation[] = { T(this->child_rotation.x), T(this->child_rotation.y), T(this->child_rotation.z), T(this->child_rotation.w) };

        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q2(child_rotation);

        const Eigen::Vector3<T> axis(T(1.0), T(0.0), T(0.0));

        const Eigen::Vector3<T> v1 = q1 * axis;
        const Eigen::Vector3<T> v2 = q2 * axis;

        const Eigen::Vector3<T> diff = (v1 - v2) * T(100.0);

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& child_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_1dof_constraint_error3, 3, 4>(
            new local_1dof_constraint_error3(child_rotation)));
    }
};

static void estimate_finger_pose(std::map<std::string, glm::mat4>& poses, const model_data& model)
{
    ceres::Problem problem;

    std::vector<std::tuple<std::string, std::string, std::string>> interpolations = {
        {"Thumb Proximal.R", "hand.R", "Thumb Intermediate.R"},
        {"Index Proximal.R", "hand.R", "Index Intermediate.R"},
        {"Middle Proximal.R", "hand.R", "Middle Intermediate.R"},
        {"Ring Proximal.R", "hand.R", "Ring Intermediate.R"},
        {"Little Proximal.R", "hand.R", "Little Intermediate.R"},
        {"Thumb Intermediate.R", "Thumb Proximal.R", "Thumb Distal.R"},
        {"Index Intermediate.R", "Index Proximal.R", "Index Distal.R"},
        {"Middle Intermediate.R", "Middle Proximal.R", "Middle Distal.R"},
        {"Ring Intermediate.R", "Ring Proximal.R", "Ring Distal.R"},
        {"Little Intermediate.R", "Little Proximal.R", "Little Distal.R"},
    };

    std::vector<std::pair<std::string, std::string>> joints = {
        {"Thumb Proximal.R", "hand.R"},
        {"Index Proximal.R", "hand.R"},
        {"Middle Proximal.R", "hand.R"},
        {"Ring Proximal.R", "hand.R"},
        {"Little Proximal.R", "hand.R"},
        {"Thumb Intermediate.R", "Thumb Proximal.R"},
        {"Index Intermediate.R", "Index Proximal.R"},
        {"Middle Intermediate.R", "Middle Proximal.R"},
        {"Ring Intermediate.R", "Ring Proximal.R"},
        {"Little Intermediate.R", "Little Proximal.R"},
        {"Thumb Distal.R", "Thumb Intermediate.R"},
        {"Index Distal.R", "Index Intermediate.R"},
        {"Middle Distal.R", "Middle Intermediate.R"},
        {"Ring Distal.R", "Ring Intermediate.R"},
        {"Little Distal.R", "Little Intermediate.R"},
    };

    std::vector<std::string> bones = {
        "Little Intermediate.R", "Ring Intermediate.R", "Middle Intermediate.R", "Index Intermediate.R", "Thumb Intermediate.R",
        "Little Proximal.R", "Ring Proximal.R", "Middle Proximal.R", "Index Proximal.R", "Thumb Proximal.R" };

    std::vector<std::string> target_bones = { "Little Distal.R", "Ring Distal.R", "Middle Distal.R", "Index Distal.R", "Thumb Distal.R" };

    std::vector<double> rotation_params(bones.size() * 4);
    std::vector<double> translation_params(bones.size() * 3);

    std::map<std::string, double*> bone_rotations;
    std::map<std::string, double*> bone_translations;
    for (std::size_t i = 0; i < bones.size(); i++)
    {
        const auto& bone = bones[i];
        double* rotation_param = &rotation_params[4 * i];
        const glm::dquat orientation = glm::quat_cast(poses.at(bone));
        const glm::vec3 position = glm::vec3(poses.at(bone)[3]);

        rotation_param[0] = orientation.x;
        rotation_param[1] = orientation.y;
        rotation_param[2] = orientation.z;
        rotation_param[3] = orientation.w;

        bone_rotations.insert(std::make_pair(bone, rotation_param));

        double* translation_param = &translation_params[3 * i];

        translation_param[0] = position.x;
        translation_param[1] = position.y;
        translation_param[2] = position.z;

        bone_translations.insert(std::make_pair(bone, translation_param));
    }

    {
        for (const auto& [bone, parent_bone, child_bone] : interpolations)
        {
            if (std::find(bones.begin(), bones.end(), parent_bone) == bones.end())
            {
                const glm::dquat parent_pose = glm::quat_cast(poses.at(parent_bone));
                double* rotation_param = bone_rotations.at(bone);
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_angle_error2::create(parent_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    rotation_param,
                    child_rotation_param);
            }
            else if (std::find(bones.begin(), bones.end(), child_bone) == bones.end())
            {
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                double* rotation_param = bone_rotations.at(bone);
                const glm::dquat child_pose = glm::quat_cast(poses.at(child_bone));

                ceres::CostFunction* cost_function =
                    local_angle_error3::create(child_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param,
                    rotation_param);
            }
            else
            {
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                double* rotation_param = bone_rotations.at(bone);
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_angle_error::create();

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param,
                    rotation_param,
                    child_rotation_param);
            }
        }
    }

    {
        for (const auto& [child_bone, parent_bone] : joints)
        {
            if (parent_bone == "hand.R")
            {
                continue;
            }
            if (parent_bone.find("Thumb") < parent_bone.size())
            {
                continue;
            }

            if (std::find(bones.begin(), bones.end(), parent_bone) == bones.end())
            {
                const glm::dquat parent_pose = glm::quat_cast(poses.at(parent_bone));
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_1dof_constraint_error2::create(parent_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    child_rotation_param);
            }
            else if (std::find(bones.begin(), bones.end(), child_bone) == bones.end())
            {
#if 1
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                const glm::dquat child_pose = glm::quat_cast(poses.at(child_bone));

                ceres::CostFunction* cost_function =
                    local_1dof_constraint_error3::create(child_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param);
#endif
            }
            else
            {
#if 1
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_1dof_constraint_error::create();

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param,
                    child_rotation_param);
#endif
            }
        }
    }

#if 0
    {
        for (const auto& [child_bone, parent_bone] : joints)
        {
            const auto global_child_position = glm::vec3(poses.at(child_bone)[3]);
            const auto parent_pose = poses.at(parent_bone);

            const auto local_child_position = glm::vec3(glm::inverse(parent_pose) * glm::vec4(global_child_position, 1.0));

            if (parents.find(parent_bone) != parents.end())
            {
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                double* parent_translation_param = bone_translations.at(parent_bone);
                double* child_translation_param = bone_translations.at(child_bone);

                ceres::CostFunction* cost_function =
                    articulation_error::create(local_child_position);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_translation_param,
                    parent_rotation_param,
                    child_translation_param);
            }
            else
            {
                double* child_translation_param = bone_translations.at(child_bone);

                ceres::CostFunction* cost_function =
                    articulation_error2::create(local_child_position);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    child_translation_param);
            }
        }
    }
#endif

    for (std::size_t i = 0; i < bones.size(); i++)
    {
        const auto& bone = bones[i];
        double* rotation_param = &rotation_params[4 * i];
        double* translation_param = &translation_params[3 * i];

        ceres::LocalParameterization* parameterization =
            new ceres::EigenQuaternionParameterization();
        problem.SetParameterization(rotation_param, parameterization);
    }

    {
        const auto start = std::chrono::system_clock::now();

        ceres::Solver::Options options;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Total elapsed for minimizer : " << elapsed / 1000.0 << " [ms]" << std::endl;
    }

    for (std::size_t i = 0; i < bones.size(); i++)
    {
        const auto& bone = bones[i];
        double* rotation_param = &rotation_params[4 * i];
        double* translation_param = &translation_params[3 * i];
        const auto orientation = glm::toMat3(glm::quat(rotation_param[3], rotation_param[0], rotation_param[1], rotation_param[2]));
        poses.at(bone) = glm::mat4(glm::vec4(orientation[0], 0.0), glm::vec4(orientation[1], 0.0), glm::vec4(orientation[2], 0.0), poses.at(bone)[3]);
    }

    for (const auto& [target_bone, parent_bone] : joints)
    {
        if (std::find(bones.begin(), bones.end(), parent_bone) != bones.end())
        {
            const auto bone_pose = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& bone) { return bone.name == target_bone; });
            const auto parent_bone_pose = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& bone) { return bone.name == parent_bone; });
            const auto bone_position = poses.at(parent_bone) * glm::inverse(parent_bone_pose->pose) * bone_pose->pose[3];

            poses.at(target_bone)[3] = bone_position;
        }
    }
}

int imu_viewer_main()
{
#if 1
    //imu_data_stream data_stream("COM3", 115200);
    //imu_data_stream data_stream("COM5", 1500000);
    qprobe_playback_stream data_stream("../data/hand_poses", 100);
    //qprobe_playback_stream data_stream("../data/capture", 100);
#else
    remote_sensor_stream data_stream("192.168.10.105:50052");
#endif

    const auto win_mgr = window_manager::get_instance();
    win_mgr->initialize();

    on_shutdown_handlers.push_back([win_mgr]()
                                   { win_mgr->terminate(); });

    const auto viewer = std::make_shared<imu_viewer>();

    const auto rendering_th = std::make_shared<rendering_thread>();
    rendering_th->start(viewer.get());

    glm::dvec3 pos(0.0);
    double heading_vel = 0;
    uint64_t last_time_us = 0;

    model_data model;
    model.load("../data/TrackingModel.json");

#if 0
    const auto recv_data_callback = [&viewer, &pos, &heading_vel, &last_time_us](const FrameData &data)
    {
        std::cout << (int)data.imu.accel << ", " << (int)data.imu.gyro << ", " << (int)data.imu.mag << std::endl;
        std::cout << data.imu.linear_accel.x << ", " << data.imu.linear_accel.y << ", " << data.imu.linear_accel.z << std::endl;

        if (last_time_us == 0)
        {
            last_time_us = data.timestamp;
        }

        const auto delta_time_us = data.timestamp - last_time_us;
        last_time_us = data.timestamp;

        const auto delta_time_ms = delta_time_us / 1000.0;
        // velocity = accel*dt (dt in seconds)
        // position = 0.5*accel*dt^2
        const auto accel_to_vel = delta_time_ms / 1000.0;
        const auto accel_to_pos = 0.5 * accel_to_vel * accel_to_vel;
        const auto deg_to_rad = 0.01745329251; // trig functions require radians, BNO055 outputs degrees

        const auto delta_pos = accel_to_pos * glm::dvec3(data.imu.linear_accel.x, data.imu.linear_accel.y, data.imu.linear_accel.z);

        glm::dquat ori(data.imu.orientation_quat.w, data.imu.orientation_quat.x, data.imu.orientation_quat.y, data.imu.orientation_quat.z);

        pos += glm::inverse(ori) * delta_pos;

        // velocity of sensor in the direction it's facing
        heading_vel = accel_to_vel * data.imu.linear_accel.x / cos(deg_to_rad * data.imu.orientation.x);

        glm::vec3 position(pos.x, pos.z, pos.y);

        glm::quat orientation(data.imu.orientation_quat.w, data.imu.orientation_quat.x, data.imu.orientation_quat.z, data.imu.orientation_quat.y);

        std::lock_guard<std::mutex> lock(viewer->mtx);
        viewer->orientation = orientation;
        viewer->position = position;

        std::cout << delta_time_us << ", " << glm::to_string(position) << std::endl;
    };
#endif
    std::map<std::string, glm::mat4> poses;

    for (std::size_t j = 0; j < model.bones.size(); j++)
    {
        const auto pose = model.bones[j].pose;
        poses.insert(std::make_pair(model.bones[j].name, pose));
    }
    const auto recv_data_callback = [&viewer, &pos, &heading_vel, &last_time_us, &model, &poses](const std::vector<glm::quat>& orientations)
    {
        std::lock_guard<std::mutex> lock(viewer->mtx);
        viewer->orientations.clear();
        viewer->positions.clear();

        finger_tracker tracker(model);

        std::vector<std::string> sensors = { "Cube.004", "Cube.003" , "Cube.002" , "Cube.001" , "Cube.005" , "Cube" };
        std::vector<std::string> bones = { "Little Distal.R", "Ring Distal.R", "Middle Distal.R", "Index Distal.R", "Thumb Distal.R", "hand.R" };
        glm::mat3 base_orientation;
        glm::vec3 base_position;
        {
            const auto i = 5;
            const auto obj = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& obj) { return obj.name == bones[i];  });
            if (obj != model.bones.end())
            {
                glm::mat3 orientation = obj->pose;
                for (size_t k = 0; k < 3; k++)
                {
                    orientation[k] = glm::normalize(orientation[k]);
                }
                glm::vec3 position = obj->pose[3];
                base_orientation = orientation;
                base_position = position;
            }
        }

        {
            for (size_t i = 0; i < orientations.size(); i++)
            {
                const auto obj = std::find_if(model.objects.begin(), model.objects.end(), [&](const auto& obj) { return obj.name == sensors[i];  });
                if (obj != model.objects.end())
                {
                    glm::mat3 default_sensor_orientation = obj->orientation;
                    for (size_t k = 0; k < 3; k++)
                    {
                        default_sensor_orientation[k] = glm::normalize(default_sensor_orientation[k]);
                    }
                    glm::vec3 default_sensor_position = obj->position;

                    viewer->orientations.push_back(default_sensor_orientation);
                    viewer->positions.push_back(default_sensor_position);

                    //

                    glm::mat4 default_bone_pose(1.0f);
                    {
                        const auto obj = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& obj) { return obj.name == bones[i];  });
                        if (obj != model.bones.end())
                        {
                            glm::mat3 orientation = obj->pose;
                            for (size_t k = 0; k < 3; k++)
                            {
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
                        const auto obj = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& obj) { return obj.name == bones[i];  });
                        if (obj != model.bones.end())
                        {
                            glm::mat3 orientation = obj->pose;
                            for (size_t k = 0; k < 3; k++)
                            {
                                orientation[k] = glm::normalize(orientation[k]);
                            }
                            glm::vec3 position = obj->pose[3];
                            default_bone_pose0[0] = glm::vec4(orientation[0], 0.0f);
                            default_bone_pose0[1] = glm::vec4(orientation[1], 0.0f);
                            default_bone_pose0[2] = glm::vec4(orientation[2], 0.0f);
                            default_bone_pose0[3] = glm::vec4(position, 1.0f);
                        }
                    }

                    //

                    const auto sensor_orientation = glm::toMat4(glm::normalize(glm::quat(orientations[i].w, orientations[i].x, orientations[i].z, -orientations[i].y)));
                    const auto sensor_orientation0 = glm::toMat4(glm::normalize(glm::quat(orientations[5].w, orientations[5].x, orientations[5].z, -orientations[5].y)));

                    const auto bone_orientation = sensor_orientation * tracker.sensor_to_bone[i];
                    const auto bone_orientation0 = sensor_orientation0 * tracker.sensor_to_bone[5];

                    const auto sensor_pose0 = glm::mat4(sensor_orientation0[0], sensor_orientation0[1], sensor_orientation0[2], glm::vec4(0.2 * (5 + 1), 0, 0, 1.0));
                    const auto sensor_pose = glm::mat4(sensor_orientation[0], sensor_orientation[1], sensor_orientation[2], glm::vec4(0.2 * (i + 1), 0, 0, 1.0));
                    const auto bone_pose0 = sensor_pose0 * tracker.sensor_to_bone[5];
                    const auto bone_pose = sensor_pose * tracker.sensor_to_bone[i];

#if 0
                    const auto bone_position2 = (bone_pose0 * glm::inverse(default_bone_pose0) * default_bone_pose)[3];
                    const auto bone_orientation2 = bone_pose;
#elif 0
                    const auto bone_position2 = (glm::inverse(default_bone_pose0) * default_bone_pose)[3];
                    const auto bone_orientation2 = glm::inverse(bone_pose0) * bone_pose;
#else
                    const auto bone_position2 = default_bone_pose[3];
                    const auto bone_orientation2 = default_bone_pose0 * glm::inverse(bone_pose0) * bone_pose;
#endif

                    {
                        viewer->orientations.push_back(sensor_pose);
                        viewer->positions.push_back(sensor_pose[3]);
                    }

                    auto bone = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& bone) { return bone.name == bones[i];  });
                    if (bone != model.bones.end())
                    {
                        poses[bone->name] = glm::mat4(bone_orientation2[0], bone_orientation2[1], bone_orientation2[2], bone_position2);
                    }
                }
            }

            estimate_finger_pose(poses, model);

            viewer->poses = poses;
        }
    };

    on_shutdown_handlers.push_back([rendering_th, viewer]()
                                   {
        rendering_th->stop();
        viewer->destroy(); });

    std::thread stream_th([&data_stream, &recv_data_callback]() {
        data_stream.subscribe_quat("", recv_data_callback);
    });

    while (!win_mgr->should_close())
    {
        win_mgr->handle_event();
    }

    //data_stream.stop();
    //if (stream_th.joinable())
    //{
    //    stream_th.join();
    //}

    shutdown();

    return 0;
}

int main()
{
    return imu_viewer_main();
}

