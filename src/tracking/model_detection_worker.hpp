#pragma once

#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <functional>
#include "model_detector.hpp"
#include "frame.hpp"

namespace marionette::tracking
{

using namespace marionette::core;

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

    void process();

public:
    model_detection_worker(const model_data& model);

    void update_frame(const frame_data_t& frame);

    void set_callback(std::function<void(const model_instance_data &, std::uint32_t, const frame_data_t&)> func);

    void start();

    void stop();
};

} // namespace marionette::tracking
