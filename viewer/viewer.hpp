#pragma once

#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>

struct window_base;

struct graphics_context
{
    window_base *window;

    graphics_context(window_base *window)
        : window(window)
    {
    }

    virtual void attach();
    virtual void detach();
    virtual void clear();
    virtual void swap_buffer();
    virtual ~graphics_context();
};

struct window_base
{
public:
    void *handle;
    std::string name;
    std::size_t width;
    std::size_t height;

    std::mutex mtx;

    window_base(std::string name, std::size_t width, std::size_t height);

    void *get_handle() const;

    virtual bool is_closed() const;

    virtual void on_close();
    virtual void on_key(int key, int scancode, int action, int mods);
    virtual void on_char(unsigned int codepoint);
    virtual void on_scroll(double x, double y);
    virtual void on_mouse_click(int button, int action, int mods);
    virtual void on_mouse(double x, double y);
    virtual void on_enter(int entered);
    virtual void on_resize(int width, int height);
    virtual void show();
    virtual void initialize();
    virtual void destroy();
    virtual void update();
    virtual ~window_base() = default;

    graphics_context create_graphics_context();
};

struct window_manager
{
    std::thread::id thread_id;
    std::atomic_bool should_close_flag;

    struct action_func_base
    {
        virtual void invoke(){};
        virtual ~action_func_base() = default;
    };

    template <typename Func>
    struct action_func : public action_func_base
    {
        Func func;
        action_func(Func &&_func) : func(std::move(_func)) {}
        action_func(action_func &&other) : func(std::move(other.func)) {}
        action_func &operator=(action_func &&other)
        {
            func = std::move(other.func);
            return *this;
        }
        virtual void invoke()
        {
            func();
        };
        virtual ~action_func() = default;
    };

    std::deque<std::unique_ptr<action_func_base>> queue;

    window_manager();

    bool should_close();
    virtual void handle_event();
    void *create_window_handle(std::string name, int width, int height, window_base *window);
    void destroy_window_handle(void *handle);
    void show_window(void *handle);
    void hide_window(void *handle);
    virtual void initialize();
    virtual void exit();
    virtual void terminate();

    virtual ~window_manager() = default;

    static std::shared_ptr<window_manager> get_instance();
};

struct rendering_thread
{
    std::unique_ptr<std::thread> th;

public:
    rendering_thread()
    {}

    void start(window_base* window)
    {
        th = std::make_unique<std::thread>([window]()
        {
            window->initialize();
            auto graphics_ctx = window->create_graphics_context();
            graphics_ctx.attach();
            window->show();
            while (!window->is_closed())
            {
                graphics_ctx.clear();
                window->update();
                graphics_ctx.swap_buffer();
            }
        });
    }

    void stop()
    {
        if (th && th->joinable())
        {
            th->join();
        }
    }
};

struct mouse_state
{
    double x, y;
    int right_button;
    int middle_button;
    int left_button;

    static mouse_state get_mouse_state(void *handle);
};

class azimuth_elevation
{
public:
    azimuth_elevation(glm::u32vec2 screen_offset, glm::u32vec2 screen_size)
        : screen_offset(screen_offset), screen_size(screen_size), start_position(0.0f, 0.0f, 0.0f), current_position(0.0f, 0.0f, 0.0f), drag_rotation(false), drag_transition(false)
    {
        reset();
    }

    glm::mat4 get_rotation_matrix()
    {
        return glm::toMat4(current_rotation);
    }

    glm::quat get_rotation_quaternion()
    {
        return current_rotation;
    }

    glm::mat4 get_translation_matrix()
    {
        return translation_matrix;
    }

    glm::mat4 get_translation_delta_matrix()
    {
        return translation_delta_matrix;
    }

    void set_radius_translation(float value)
    {
        radius_translation = value;
    }

    void set_radius(float value)
    {
        radius = value;
    }
    float get_radius() const
    {
        return radius;
    }

    float get_screen_w() const
    {
        return (float)screen_size.x;
    }
    float get_screen_h() const
    {
        return (float)screen_size.y;
    }
    float get_screen_x() const
    {
        return (float)screen_offset.x;
    }
    float get_screen_y() const
    {
        return (float)screen_offset.y;
    }

    glm::quat quat_from_screen(glm::vec3 from, glm::vec3 to)
    {
        const auto vector = (to - from) * 1.f /*radius*/;

        angle.x += vector.y / get_screen_w();
        angle.y += vector.x / get_screen_h();

        return glm::quat_cast(glm::rotate(angle.x, glm::vec3(1.f, 0.f, 0.f)) * glm::rotate(angle.t, glm::vec3(0.f, 1.f, 0.f)));
    }
    glm::vec3 screen_to_vector(float sx, float sy)
    {
        return glm::vec3(sx, sy, 0);
    }

    bool on_target(int x, int y) const
    {
        x -= screen_offset.x;
        y -= screen_offset.y;

        return (x >= 0) && (y >= 0) && (x < static_cast<int>(screen_size.x)) && (y < static_cast<int>(screen_size.y));
    }

    void begin_rotation(int x, int y)
    {
        if (on_target(x, y))
        {
            drag_rotation = true;
            previous_rotation = current_rotation;
            start_position = screen_to_vector((float)x, (float)y);
        }
    }
    void update_rotation(float x, float y)
    {
        if (drag_rotation)
        {
            current_position = screen_to_vector(x, y);
            current_rotation = quat_from_screen(start_position, current_position);
            start_position = current_position;
        }
    }
    void end_rotation()
    {
        drag_rotation = false;
    }

    void begin_transition(int x, int y)
    {
        if (on_target(x, y))
        {
            drag_transition = true;
            previsou_position.x = (float)x;
            previsou_position.y = (float)y;
        }
    }
    void update_transition(int x, int y, bool zoom)
    {
        if (drag_transition)
        {
            float delta_x = (previsou_position.x - (float)x) * radius_translation / get_screen_w();
            float delta_y = (previsou_position.y - (float)y) * radius_translation / get_screen_h();

            if (!zoom)
            {
                translation_delta_matrix = glm::translate(glm::vec3(-2 * delta_x, 2 * delta_y, 0.0f));
                translation_matrix = translation_delta_matrix * translation_matrix;
            }
            else
            {
                translation_delta_matrix = glm::translate(glm::vec3(0.0f, 0.0f, 5 * delta_y));
                translation_matrix = translation_delta_matrix * translation_matrix;
            }

            previsou_position.x = (float)x;
            previsou_position.y = (float)y;
        }
    }
    void end_transition()
    {
        translation_delta_matrix = glm::identity<glm::mat4>();
        drag_transition = false;
    }

    void reset()
    {
        angle = glm::vec2(0.f, 0.f);
        previous_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
        current_rotation = glm::angleAxis(glm::radians(30.f), glm::vec3(1.f, 0.f, 0.f));
        translation_matrix = glm::translate(glm::vec3(0.f, 1.f, 0.f));
        translation_delta_matrix = glm::identity<glm::mat4>();
        drag_rotation = false;
        radius_translation = 1.0f;
        radius = 5.0f;
    }

    void update(mouse_state mouse);

    void scroll(double x, double y)
    {
        radius -= (static_cast<float>(y) * 1.0f);
    }

private:
    static glm::vec2 get_center(int width, int height)
    {
        return glm::vec2(width * 0.5f, height * 0.5f);
    }
    static glm::vec2 get_center(const glm::u32vec2 &screen)
    {
        return glm::vec2(screen.x * 0.5f, screen.y * 0.5f);
    }
    glm::u32vec2 screen_offset;
    glm::u32vec2 screen_size;

    float radius;
    float radius_translation;

    glm::vec2 angle;
    glm::vec3 start_position;
    glm::vec3 previsou_position;
    glm::vec3 current_position;
    glm::quat previous_rotation;
    glm::quat current_rotation;

    glm::mat4 translation_matrix;
    glm::mat4 translation_delta_matrix;

    bool drag_rotation, drag_transition;

    mouse_state previous_state;
};
