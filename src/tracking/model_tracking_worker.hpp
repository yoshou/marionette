#pragma once

#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <map>
#include <string>
#include <glm/glm.hpp>
#include "model.hpp"
#include "model_detector.hpp"
#include "frame_queue.hpp"
#include "motion_tracker.hpp"

namespace marionette::tracking
{

using namespace marionette::core;

class model_tracking_worker
{
    std::unique_ptr<std::thread> th;
    std::atomic_bool running;
    std::mutex mtx;

    model_data model;
    model_data instanced_model;
    model_instance_data model_instance;
    std::shared_ptr<::frame_cursor> frame_cursor;

    mutable std::mutex poses_mtx;
    std::map<std::string, glm::mat4> poses;

    void process();

public:
    model_tracking_worker(const model_data& model, const model_instance_data& model_instance, const std::shared_ptr<::frame_cursor>& frame_cursor);

    void start();

    void stop();

    std::map<std::string, glm::mat4> get_poses() const;
};

} // namespace marionette::tracking
