#pragma once

#include <memory>
#include <functional>

class widget_drawer
{
public:
    void initialize(void *handle, std::size_t width, std::size_t height);
    void draw();

    widget_drawer();
    ~widget_drawer();

    std::function<void(int)> frame_no_changed;
    std::function<void(bool)> check_r_changed;
    std::function<void(bool)> check_g_changed;
    std::function<void(bool)> check_b_changed;

    std::string selected_name;

    void on_char(void* handle, unsigned int codepoint);

private:
    struct widget_drawer_data;

    std::unique_ptr<widget_drawer_data> pimpl;
};
