#include <cstdlib>
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <atomic>
#include <thread>
#include <future>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include "widget_drawer.hpp"

#ifdef _WIN32
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#define NK_KEYSTATE_BASED_INPUT
#include "nuklear.h"
#include "nuklear_glfw_gl3.h"
#else
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL4_IMPLEMENTATION
#define NK_KEYSTATE_BASED_INPUT
#include "nuklear.h"
#include "nuklear_glfw_gl4.h"
#endif

enum theme
{
    THEME_BLACK,
    THEME_WHITE,
    THEME_RED,
    THEME_BLUE,
    THEME_DARK
};

static void
set_style(struct nk_context *ctx, enum theme theme)
{
    struct nk_color table[NK_COLOR_COUNT];
    if (theme == THEME_WHITE)
    {
        table[NK_COLOR_TEXT] = nk_rgba(70, 70, 70, 255);
        table[NK_COLOR_WINDOW] = nk_rgba(175, 175, 175, 255);
        table[NK_COLOR_HEADER] = nk_rgba(175, 175, 175, 255);
        table[NK_COLOR_BORDER] = nk_rgba(0, 0, 0, 255);
        table[NK_COLOR_BUTTON] = nk_rgba(185, 185, 185, 255);
        table[NK_COLOR_BUTTON_HOVER] = nk_rgba(170, 170, 170, 255);
        table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(160, 160, 160, 255);
        table[NK_COLOR_TOGGLE] = nk_rgba(150, 150, 150, 255);
        table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(120, 120, 120, 255);
        table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(175, 175, 175, 255);
        table[NK_COLOR_SELECT] = nk_rgba(190, 190, 190, 255);
        table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(175, 175, 175, 255);
        table[NK_COLOR_SLIDER] = nk_rgba(190, 190, 190, 255);
        table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(80, 80, 80, 255);
        table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(70, 70, 70, 255);
        table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(60, 60, 60, 255);
        table[NK_COLOR_PROPERTY] = nk_rgba(175, 175, 175, 255);
        table[NK_COLOR_EDIT] = nk_rgba(150, 150, 150, 255);
        table[NK_COLOR_EDIT_CURSOR] = nk_rgba(0, 0, 0, 255);
        table[NK_COLOR_COMBO] = nk_rgba(175, 175, 175, 255);
        table[NK_COLOR_CHART] = nk_rgba(160, 160, 160, 255);
        table[NK_COLOR_CHART_COLOR] = nk_rgba(45, 45, 45, 255);
        table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
        table[NK_COLOR_SCROLLBAR] = nk_rgba(180, 180, 180, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(140, 140, 140, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(150, 150, 150, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(160, 160, 160, 255);
        table[NK_COLOR_TAB_HEADER] = nk_rgba(180, 180, 180, 255);
        nk_style_from_table(ctx, table);
    }
    else if (theme == THEME_RED)
    {
        table[NK_COLOR_TEXT] = nk_rgba(190, 190, 190, 255);
        table[NK_COLOR_WINDOW] = nk_rgba(30, 33, 40, 215);
        table[NK_COLOR_HEADER] = nk_rgba(181, 45, 69, 220);
        table[NK_COLOR_BORDER] = nk_rgba(51, 55, 67, 255);
        table[NK_COLOR_BUTTON] = nk_rgba(181, 45, 69, 255);
        table[NK_COLOR_BUTTON_HOVER] = nk_rgba(190, 50, 70, 255);
        table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(195, 55, 75, 255);
        table[NK_COLOR_TOGGLE] = nk_rgba(51, 55, 67, 255);
        table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(45, 60, 60, 255);
        table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(181, 45, 69, 255);
        table[NK_COLOR_SELECT] = nk_rgba(51, 55, 67, 255);
        table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(181, 45, 69, 255);
        table[NK_COLOR_SLIDER] = nk_rgba(51, 55, 67, 255);
        table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(181, 45, 69, 255);
        table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(186, 50, 74, 255);
        table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(191, 55, 79, 255);
        table[NK_COLOR_PROPERTY] = nk_rgba(51, 55, 67, 255);
        table[NK_COLOR_EDIT] = nk_rgba(51, 55, 67, 225);
        table[NK_COLOR_EDIT_CURSOR] = nk_rgba(190, 190, 190, 255);
        table[NK_COLOR_COMBO] = nk_rgba(51, 55, 67, 255);
        table[NK_COLOR_CHART] = nk_rgba(51, 55, 67, 255);
        table[NK_COLOR_CHART_COLOR] = nk_rgba(170, 40, 60, 255);
        table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
        table[NK_COLOR_SCROLLBAR] = nk_rgba(30, 33, 40, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(64, 84, 95, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(70, 90, 100, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(75, 95, 105, 255);
        table[NK_COLOR_TAB_HEADER] = nk_rgba(181, 45, 69, 220);
        nk_style_from_table(ctx, table);
    }
    else if (theme == THEME_BLUE)
    {
        table[NK_COLOR_TEXT] = nk_rgba(20, 20, 20, 255);
        table[NK_COLOR_WINDOW] = nk_rgba(202, 212, 214, 215);
        table[NK_COLOR_HEADER] = nk_rgba(137, 182, 224, 220);
        table[NK_COLOR_BORDER] = nk_rgba(140, 159, 173, 255);
        table[NK_COLOR_BUTTON] = nk_rgba(137, 182, 224, 255);
        table[NK_COLOR_BUTTON_HOVER] = nk_rgba(142, 187, 229, 255);
        table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(147, 192, 234, 255);
        table[NK_COLOR_TOGGLE] = nk_rgba(177, 210, 210, 255);
        table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(182, 215, 215, 255);
        table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(137, 182, 224, 255);
        table[NK_COLOR_SELECT] = nk_rgba(177, 210, 210, 255);
        table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(137, 182, 224, 255);
        table[NK_COLOR_SLIDER] = nk_rgba(177, 210, 210, 255);
        table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(137, 182, 224, 245);
        table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(142, 188, 229, 255);
        table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(147, 193, 234, 255);
        table[NK_COLOR_PROPERTY] = nk_rgba(210, 210, 210, 255);
        table[NK_COLOR_EDIT] = nk_rgba(210, 210, 210, 225);
        table[NK_COLOR_EDIT_CURSOR] = nk_rgba(20, 20, 20, 255);
        table[NK_COLOR_COMBO] = nk_rgba(210, 210, 210, 255);
        table[NK_COLOR_CHART] = nk_rgba(210, 210, 210, 255);
        table[NK_COLOR_CHART_COLOR] = nk_rgba(137, 182, 224, 255);
        table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
        table[NK_COLOR_SCROLLBAR] = nk_rgba(190, 200, 200, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(64, 84, 95, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(70, 90, 100, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(75, 95, 105, 255);
        table[NK_COLOR_TAB_HEADER] = nk_rgba(156, 193, 220, 255);
        nk_style_from_table(ctx, table);
    }
    else if (theme == THEME_DARK)
    {
        table[NK_COLOR_TEXT] = nk_rgba(210, 210, 210, 255);
        table[NK_COLOR_WINDOW] = nk_rgba(57, 67, 71, 215);
        table[NK_COLOR_HEADER] = nk_rgba(51, 51, 56, 220);
        table[NK_COLOR_BORDER] = nk_rgba(46, 46, 46, 255);
        table[NK_COLOR_BUTTON] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_BUTTON_HOVER] = nk_rgba(58, 93, 121, 255);
        table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(63, 98, 126, 255);
        table[NK_COLOR_TOGGLE] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(45, 53, 56, 255);
        table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_SELECT] = nk_rgba(57, 67, 61, 255);
        table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_SLIDER] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(48, 83, 111, 245);
        table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(53, 88, 116, 255);
        table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(58, 93, 121, 255);
        table[NK_COLOR_PROPERTY] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_EDIT] = nk_rgba(50, 58, 61, 225);
        table[NK_COLOR_EDIT_CURSOR] = nk_rgba(210, 210, 210, 255);
        table[NK_COLOR_COMBO] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_CHART] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_CHART_COLOR] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
        table[NK_COLOR_SCROLLBAR] = nk_rgba(50, 58, 61, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(48, 83, 111, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(53, 88, 116, 255);
        table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(58, 93, 121, 255);
        table[NK_COLOR_TAB_HEADER] = nk_rgba(48, 83, 111, 255);
        nk_style_from_table(ctx, table);
    }
    else
    {
        nk_style_default(ctx);
    }
}

#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024

namespace component
{

class nk_property_int
{
    std::string label;
    int value, min, max;
    int step;
    float inc_per_pixel;

public:
    std::function<void()> changed;

    nk_property_int(std::string label, int value, int min, int max, int step, float inc_per_pixel)
        : label(label), value(value), min(min), max(max), step(step), inc_per_pixel(inc_per_pixel)
    {
    }

    void update(nk_context *ctx)
    {
        int prev = value;
        ::nk_property_int(ctx, label.c_str(), min, &value, max, step, inc_per_pixel);

        if (prev != value)
        {
            if (changed)
            {
                changed();
            }
        }
    }

    int get_value() const
    {
        return value;
    }
    int get_min_value() const
    {
        return min;
    }
    int get_max_value() const
    {
        return max;
    }
    void set_min_value(int value)
    {
        min = value;
    }
    void set_max_value(int value)
    {
        max = value;
    }

    static std::shared_ptr<nk_property_int> create(std::string label, int value, int min, int max, int step, float inc_per_pixel)
    {
        std::shared_ptr<nk_property_int> component(new nk_property_int(label, value, min, max, step, inc_per_pixel));
        return component;
    }
};
};

struct widget_drawer::widget_drawer_data
{
    /* Platform */
    nk_glfw glfw;
    GLFWwindow *win = nullptr;
    int width = 0, height = 0;
    struct nk_context *ctx = nullptr;
    struct nk_colorf bg;
    struct nk_image img;
    std::shared_ptr<component::nk_property_int> frame_prop;

    nk_bool checked_r = 1;
    nk_bool checked_g = 1;
    nk_bool checked_b = 1;
};

widget_drawer::widget_drawer()
    : pimpl(new widget_drawer_data())
{
    pimpl->frame_prop = component::nk_property_int::create("Frame", 2000, 0, std::numeric_limits<int>::max(), 1, 1);
    pimpl->frame_prop->changed = [this]()
    {
        if (frame_no_changed)
        {
            frame_no_changed(pimpl->frame_prop->get_value());
        }
    };
}

widget_drawer::~widget_drawer() = default;

void widget_drawer::on_char(void *handle, unsigned int codepoint)
{
#if _WIN32
    nk_glfw3_char_callback((nk_glfw *)handle, codepoint);
#else
    nk_glfw3_char_callback((GLFWwindow *)handle, codepoint);
#endif
}

void widget_drawer::initialize(void* handle, std::size_t width, std::size_t height)
{
    pimpl->win = (GLFWwindow *)handle;
#if _WIN32
    pimpl->ctx = nk_glfw3_init(&pimpl->glfw, pimpl->win, NK_GLFW3_INSTALL_CALLBACKS);

    /* Load Fonts: if none of these are loaded a default font will be used  */
    /* Load Cursor: if you uncomment cursor loading please hide the cursor */
    {
        struct nk_font_atlas* atlas;
        nk_glfw3_font_stash_begin(&pimpl->glfw, &atlas);
        struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", 16, 0);
        nk_glfw3_font_stash_end(&pimpl->glfw);
        nk_style_set_font(pimpl->ctx, &roboto->handle);
    }

    /*set_style(ctx, THEME_WHITE);*/
    /*set_style(ctx, THEME_RED);*/
    /*set_style(ctx, THEME_BLUE);*/
    /*set_style(ctx, THEME_DARK);*/
#else
    pimpl->ctx = nk_glfw3_init(pimpl->win, NK_GLFW3_DEFAULT, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);

    /* Load Fonts: if none of these are loaded a default font will be used  */
    /* Load Cursor: if you uncomment cursor loading please hide the cursor */
    {
        struct nk_font_atlas *atlas;
        nk_glfw3_font_stash_begin(&atlas);
        struct nk_font_config config = nk_font_config(0);
        config.range = nk_font_chinese_glyph_ranges();
        struct nk_font *robot = nk_font_atlas_add_from_file(atlas, "../fonts/mplus/fonts/ttf/Mplus2-Regular.ttf", (float)16, &config);
        nk_glfw3_font_stash_end();

        // for (size_t i = 0; i < NK_CURSOR_COUNT; i++)
        // {
        //     atlas->cursors[i].size.x = (float)window->scaleh((int)atlas->cursors[i].size.x);
        //     atlas->cursors[i].size.y = (float)window->scalev((int)atlas->cursors[i].size.y);
        // }

        // nk_style_load_all_cursors(ctx, atlas->cursors);
        nk_style_set_font(pimpl->ctx, &robot->handle);
    }

    /* style.c */
    /*set_style(ctx, THEME_WHITE);*/
    /*set_style(ctx, THEME_RED);*/
    /*set_style(ctx, THEME_BLUE);*/
    /*set_style(ctx, THEME_DARK);*/

    /* Create bindless texture.
     * The index returned is not the opengl resource id.
     * IF you need the GL resource id use: nk_glfw3_get_tex_ogl_id() */

    {
        int tex_index = 0;
        enum
        {
            tex_width = 256,
            tex_height = 256
        };
        char pixels[tex_width * tex_height * 4];
        memset(pixels, 128, sizeof(pixels));
        tex_index = nk_glfw3_create_texture(pixels, tex_width, tex_height);
        pimpl->img = nk_image_id(tex_index);
    }
#endif
}

void widget_drawer::draw()
{
    auto ctx = pimpl->ctx;
    auto win = pimpl->win;
    auto bg = pimpl->bg;

    if (ctx == nullptr)
    {
        return;
    }
    if (win == nullptr)
    {
        return;
    }

#ifdef _WIN32
    nk_glfw3_new_frame(&pimpl ->glfw);
#else
    nk_glfw3_new_frame();
#endif

    if (nk_begin(ctx, "Panel", nk_rect(0, 0, 1280, 30), NK_WINDOW_NO_SCROLLBAR))
    {
        nk_menubar_begin(ctx);
        nk_layout_row_static(ctx, 30, 50, 1);
        nk_menu_item_text(ctx, "File", 4, NK_TEXT_ALIGN_CENTERED);
        nk_menubar_end(ctx);
    }
    nk_end(ctx);

    /* GUI */
    if (nk_begin(ctx, "Demo", nk_rect(50, 50, 230, 250),
                 NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
                     NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE))
    {
        enum
        {
            EASY,
            HARD
        };
        static int op = EASY;
        static int property = 20;
        nk_layout_row_static(ctx, 30, 80, 1);
        if (nk_button_label(ctx, "button"))
            fprintf(stdout, "button pressed\n");

        nk_layout_row_dynamic(ctx, 25, 1);
        pimpl->frame_prop->update(ctx);

        nk_bool prev_checked_r = pimpl->checked_r;
        nk_bool prev_checked_g = pimpl->checked_g;
        nk_bool prev_checked_b = pimpl->checked_b;
        nk_layout_row_static(ctx, 30, 80, 1);
        nk_checkbox_label(ctx, "Show R", &pimpl->checked_r);
        nk_layout_row_static(ctx, 30, 80, 1);
        nk_checkbox_label(ctx, "Show G", &pimpl->checked_g);
        nk_layout_row_static(ctx, 30, 80, 1);
        nk_checkbox_label(ctx, "Show B", &pimpl->checked_b);
        if (prev_checked_r != pimpl->checked_r)
        {
            if (check_r_changed)
            {
                check_r_changed(pimpl->checked_r != 0);
            }
        }
        if (prev_checked_g != pimpl->checked_g)
        {
            if (check_g_changed)
            {
                check_g_changed(pimpl->checked_g != 0);
            }
        }
        if (prev_checked_b != pimpl->checked_b)
        {
            if (check_b_changed)
            {
                check_b_changed(pimpl->checked_b != 0);
            }
        }

        nk_layout_row_dynamic(ctx, 20, 1);

        std::string text = "selected: " + selected_name;
        nk_label(ctx, text.c_str(), NK_TEXT_LEFT);
    }
    nk_end(ctx);

#ifdef _WIN32
    nk_glfw3_render(&pimpl->glfw, NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
#else
    nk_glfw3_render(NK_ANTI_ALIASING_ON);
#endif
}
