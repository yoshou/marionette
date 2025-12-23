#include "model_detection_worker.hpp"
#include <iostream>

model_detection_worker::model_detection_worker(const model_data& model)
    : running(false), model(model), is_callback_valid(false)
{}

void model_detection_worker::update_frame(const frame_data_t& frame)
{
    std::lock_guard lock(mtx);
    this->frame = frame;
}

void model_detection_worker::set_callback(std::function<void(const model_instance_data &, std::uint32_t, const frame_data_t&)> func)
{
    callback = func;
    is_callback_valid = true;
}

void model_detection_worker::start()
{
    th.reset(new std::thread(&model_detection_worker::process, this));
}

void model_detection_worker::stop()
{
    running = false;
    if (th && th->joinable())
    {
        th->join();
    }
}

void model_detection_worker::process()
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
