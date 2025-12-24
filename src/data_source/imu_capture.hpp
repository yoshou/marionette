#pragma once

#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "serial_port.hpp"
#include "imu_data.hpp"

namespace marionette::data_source {

class imu_capture
{
    serial_port port;
    std::chrono::system_clock::time_point system_clock_start;
    uint64_t device_clock_start;
    std::atomic_bool running;

public:
    struct pose_data
    {
        uint8_t accel_status;
        uint8_t gyro_status;
        uint8_t mag_status;
        glm::quat orientation;
    };
    
    struct pose_frame
    {
        double timestamp;
        std::vector<pose_data> poses;
    };

    imu_capture();

    void open(std::string port_name);

    void start(std::function<void(const pose_frame &)> frame_received);

    void stop();

    bool is_running() const;
};

} // namespace marionette::data_source
