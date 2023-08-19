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
#include "bone_drawer.hpp"
#include "widget_drawer.hpp"
#include "drawer2d.hpp"
#include "debug.hpp"

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

struct log_viewer : public window_base
{
    std::shared_ptr<azimuth_elevation> view_controller;
    rect_selector rect_selector_;
    sphere_drawer sphere_drawer_;
    bone_drawer bone_drawer_;
    grid_drawer grid_drawer_;
    drawer2d drawer2d_;
    widget_drawer widget_drawer_;
    std::mutex mtx;
    std::vector<glm::vec3> markers;
    std::map<std::string, glm::mat4> poses;
    glm::u8vec4 color;
    glm::mat4 world;
    bool initialized = false;

    point_cloud_debug_drawer point_cloud_;

    int selected_index = -1;

    bool show_r, show_g, show_b;

    std::vector<std::vector<glm::vec3>> marker_frames;

    log_viewer()
        : window_base("Marker Viewer", SCREEN_WIDTH, SCREEN_HEIGHT), sphere_drawer_(36, 18, false), world(1.f)
    {
        show_r = true;
        show_g = true;
        show_b = true;

#if 1
        std::vector<std::string> filenames;
        std::string path = "../data/markers/";
        std::vector<std::size_t> iter;
        for (const auto &entry : fs::directory_iterator(path))
        {
            const std::string s = entry.path().filename().string();
            filenames.push_back(s);
        }

        widget_drawer_.frame_no_changed = [this, filenames, path](int frame_no)
        {
            std::string filename;
            {
                filename = (fs::path(path) / ("capture_markers_" + std::to_string(frame_no) + ".pcd")).string();
            }
            std::cout << filename << std::endl;
            point_cloud_.load(filename);

            std::cout << "Num markers: " << point_cloud_.size() << std::endl;
        };
#elif 1
        widget_drawer_.frame_no_changed = [this](int frame_no)
        {
            std::ifstream ifs;
            ifs.open("pose_" + std::to_string(frame_no - 1700) + ".json", std::ios::in);
            nlohmann::json j = nlohmann::json::parse(ifs);
            const auto points = j["points"].get<std::vector<glm::vec3>>();

            point_cloud_.clear();
            point_cloud_.add(points, glm::u8vec3(255, 0, 0));

            std::vector<glm::mat4> poses;
            for (const auto& [name, pose] : j["poses"].get<std::map<std::string, glm::mat4>>())
            {
                poses.push_back(pose);
            }

            point_cloud_.poses = poses;
        };
#else
        {
            std::ifstream ifs;
            ifs.open("../data/capture_pose_points.json", std::ios::in);
            nlohmann::json j_frames = nlohmann::json::parse(ifs);
            for (const auto &j_frame : j_frames)
            {
                const auto markers = j_frame["markers"].get<std::vector<glm::vec3>>();
                marker_frames.push_back(markers);
            }
        }
        widget_drawer_.frame_no_changed = [this](int frame_no)
        {
            point_cloud_.clear();
            point_cloud_.add(marker_frames[frame_no], glm::u8vec3(255, 0, 0));
        };
#endif

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
        float near = 1.0f;
        float far = 1000.0f;
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        // set viewport to be the entire window
        glViewport(0, 0, (GLsizei)width, (GLsizei)height);

        glm::mat4 proj = glm::perspective(fovy, aspect, near, far);
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

        if (!initialized)
        {
            sphere_drawer_.initialize();
            bone_drawer_.initialize();
            grid_drawer_.initialize();
            widget_drawer_.initialize(handle, SCREEN_WIDTH, SCREEN_HEIGHT);
            drawer2d_.initialize();

            initialized = true;
        }

        const auto on_selected = [this](glm::vec2 beg, glm::vec2 end)
        {
            glm::vec2 rect_min(std::min(beg.x, end.x), std::min(beg.y, end.y));
            glm::vec2 rect_max(std::max(beg.x, end.x), std::max(beg.y, end.y));

            const auto clip_pos = glm::vec2(rect_min.x / SCREEN_WIDTH * 2 - 1, rect_max.y / SCREEN_HEIGHT * -2 + 1);
            const auto clip_size = glm::vec2(std::abs(rect_max.x - rect_min.x) / SCREEN_WIDTH * 2, std::abs(rect_max.y - rect_min.y) / SCREEN_HEIGHT * 2);

            for (std::size_t i = 0; i < point_cloud_.size(); i++)
            {
                glm::vec3 marker;
                glm::u8vec3 color;
                point_cloud_.get(i, marker, color);

                if (glm::u8vec3(color) != glm::u8vec3(0, 255, 0))
                {
                    continue;
                }

                auto pos = pvw * world * glm::vec4(marker, 1.0f);
                pos.x /= pos.w;
                pos.y /= pos.w;
                if (pos.x >= clip_pos.x && pos.y >= clip_pos.y && pos.x <= (clip_pos.x + clip_size.x) && pos.y <= (clip_pos.y + clip_size.y))
                {
                    widget_drawer_.selected_name = glm::to_string(pos);
                    selected_index = i;
                    return;
                }
            }
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
        float radius = view_controller->get_radius();
        glm::vec3 forward(0.f, 0.f, 1.f);
        const auto target_pos = glm::vec3(-view_controller->get_translation_matrix()[3]);
        glm::vec3 view_pos = target_pos + glm::rotate(glm::inverse(view_controller->get_rotation_quaternion()), forward * radius);

        set_camera(view_pos.x, view_pos.y, view_pos.z, target_pos.x, target_pos.y, target_pos.z);

        grid_drawer_.draw(pvw);

        {
            for (std::size_t i = 0; i < point_cloud_.size(); i++)
            {
                glm::vec3 marker;
                glm::u8vec3 color;
                point_cloud_.get(i, marker, color);

                if (!show_r && glm::u8vec3(color) == glm::u8vec3(255, 0, 0))
                {
                    continue;
                }
                if (!show_g && glm::u8vec3(color) == glm::u8vec3(0, 255, 0))
                {
                    continue;
                }
                if (!show_b && glm::u8vec3(color) == glm::u8vec3(0, 0, 255))
                {
                    continue;
                }

                if (( glm::u8vec3(color) != glm::u8vec3(255, 0, 0))
                 && ( glm::u8vec3(color) != glm::u8vec3(0, 255, 0))
                 && (glm::u8vec3(color) != glm::u8vec3(0, 0, 255)))
                {
                    continue;
                }

                if (i == selected_index)
                {
                    color = glm::u8vec3(0, 255, 255);
                }

                float lineColor[] = {
                    std::clamp(color.r / 255.f, 0.f, 1.f),
                    std::clamp(color.g / 255.f, 0.f, 1.f),
                    std::clamp(color.b / 255.f, 0.f, 1.f), 1.f};
                const auto pos = world * glm::vec4(marker, 1.0f);
                sphere_drawer_.drawWithLines(pvw * glm::translate(glm::vec3(pos)) * glm::scale(glm::vec3(0.01f, 0.01f, 0.01f)), lineColor);
            }
        }

        {
            
            for (std::size_t i = 0; i < point_cloud_.poses.size(); i++)
            {
                const auto pose = point_cloud_.poses[i];
                bone_drawer_.draw(pvw* pose* glm::scale(glm::vec3(1.f, 0.3f, 1.f)));
            }
        }

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

int log_viewer_main()
{
    const auto win_mgr = window_manager::get_instance();
    win_mgr->initialize();

    on_shutdown_handlers.push_back([win_mgr]()
                                   { win_mgr->terminate(); });

    const auto viewer = std::make_shared<log_viewer>();

    const auto rendering_th = std::make_shared<rendering_thread>();
    rendering_th->start(viewer.get());

    on_shutdown_handlers.push_back([rendering_th, viewer]()
                                   {
        rendering_th->stop();
        viewer->destroy(); });

    while (!win_mgr->should_close())
    {
        win_mgr->handle_event();
    }

    shutdown();

    return 0;
}

int main()
{
    return log_viewer_main();
}

