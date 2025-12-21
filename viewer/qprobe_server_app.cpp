#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "serial_port.hpp"
#include "imu_data.hpp"
#include "base64.hpp"
#include "stream_server.hpp"

namespace fs = std::filesystem;

// qprobe_capture class - captures IMU data from serial port
class qprobe_capture
{
    serial_port port;
    std::chrono::system_clock::time_point system_clock_start;
    uint64_t device_clock_start;

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

    void open(std::string port_name)
    {
        port.open(port_name);
        port.set_baudrate(1500000);
    }

    void start(std::function<void(const pose_frame &)> frame_received)
    {
        std::vector<uint8_t> buf;
        bool first_frame = true;
        
        while (true)
        {
            size_t receive_len = port.get_received_size();
            if (receive_len <= 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            std::vector<uint8_t> data(receive_len);
            size_t read_len = port.read(data.data(), data.size());

            std::vector<std::string> lines;
            for (const auto c : data)
            {
                if (c == '\n')
                {
                    lines.push_back(std::string(buf.begin(), buf.end()));
                    buf.clear();
                }
                else
                {
                    buf.push_back(c);
                }
            }

            for (const auto &line : lines)
            {
                ResponseHeader header;
                size_t header_size = 0;
                decode_base64(line, (uint8_t *)&header, sizeof(header), &header_size);
                if (header_size <= line.size())
                {
                    FrameData frame;
                    size_t frame_size = 0;
                    decode_base64(line.substr(header_size), (uint8_t *)&frame, sizeof(frame), &frame_size);
                    if (header_size + frame_size + 1 /* \r */ == line.size())
                    {
                        pose_frame pose;

                        if (first_frame)
                        {
                            system_clock_start = std::chrono::system_clock::now();
                            device_clock_start = frame.timestamp;
                            first_frame = false;
                        }

                        const auto timestamp = (std::chrono::duration_cast<std::chrono::nanoseconds>(system_clock_start.time_since_epoch()).count() +
                                                (frame.timestamp - device_clock_start)) /
                                               1000000.0;

                        pose.timestamp = timestamp;
                        for (int i = 0; i < NUM_SENSORS; i++)
                        {
                            pose_data data;
                            data.accel_status = frame.imu[i].accel;
                            data.gyro_status = frame.imu[i].gyro;
                            data.mag_status = frame.imu[i].mag;
                            data.orientation = glm::quat(frame.imu[i].orientation_quat.w, 
                                                        frame.imu[i].orientation_quat.x, 
                                                        frame.imu[i].orientation_quat.y, 
                                                        frame.imu[i].orientation_quat.z);
                            pose.poses.push_back(data);
                        }

                        frame_received(pose);
                    }
                }
            }
        }
    }
};

// JSON serialization helpers for glm types
namespace glm
{
    static void to_json(nlohmann::json &j, const glm::quat &v)
    {
        j = {v.x, v.y, v.z, v.w};
    }
}

int main(int argc, char* argv[])
{
    spdlog::set_level(spdlog::level::info);
    spdlog::info("qprobe Server Application");

    // Start gRPC streaming server
    marker_stream_server server;
    server.run();
    spdlog::info("gRPC server started on port 50052");

    // List available serial ports
    spdlog::info("Available serial ports:");
    for (const auto& name : serial_port::get_serial_port_names())
    {
        spdlog::info("  - {}", name);
    }

    // Determine serial port to use
    std::string port_name;
    if (argc > 1)
    {
        port_name = argv[1];
    }
    else
    {
#ifdef _WIN32
        port_name = "COM3";  // Default Windows port
#else
        port_name = "/dev/ttyUSB0";  // Default Linux port
#endif
    }
    spdlog::info("Opening serial port: {}", port_name);

    // Optional: Create data directory for saving frames
    const std::string data_dir = "../data/capture";
    if (!fs::exists(data_dir))
    {
        fs::create_directories(data_dir);
        spdlog::info("Created data directory: {}", data_dir);
    }

    // Open qprobe device and start capturing
    qprobe_capture capture;
    try
    {
        capture.open(port_name);
        spdlog::info("Serial port opened successfully");
    }
    catch (const std::exception& e)
    {
        spdlog::error("Failed to open serial port: {}", e.what());
        return 1;
    }

    // Start capture loop
    spdlog::info("Starting capture loop...");
    capture.start([&](const qprobe_capture::pose_frame &frame) {
        // Print frame info to console
        std::cout << (uint64_t)(frame.timestamp * 1000);
        for (int i = 0; i < NUM_SENSORS; i++)
        {
            std::cout << " | " << (int)frame.poses[i].accel_status 
                      << "," << (int)frame.poses[i].gyro_status 
                      << "," << (int)frame.poses[i].mag_status;
        }
        for (int i = 0; i < NUM_SENSORS; i++)
        {
            std::cout << " | " << glm::to_string(frame.poses[i].orientation);
        }
        std::cout << " |" << std::endl;

        // Prepare data for JSON (optional - can be disabled for performance)
        nlohmann::json j;
        j["timestamp"] = frame.timestamp;

        std::vector<glm::quat> orientations;
        std::vector<nlohmann::json> poses;
        
        for (int i = 0; i < NUM_SENSORS; i++)
        {
            nlohmann::json pose;
            pose["accel_status"] = (int)frame.poses[i].accel_status;
            pose["gyro_status"] = (int)frame.poses[i].gyro_status;
            pose["mag_status"] = (int)frame.poses[i].mag_status;
            pose["orientation"] = frame.poses[i].orientation;

            orientations.push_back(frame.poses[i].orientation);
            poses.push_back(pose);
        }
        j["poses"] = poses;

        // Optional: Save to file (commented out for performance)
        // const auto j_str = j.dump(2);
        // const auto path = data_dir + "/" + std::to_string((uint64_t)(frame.timestamp * 1000)) + ".json";
        // std::ofstream ofs(path, std::ios::out | std::ios::binary);
        // ofs.write(j_str.c_str(), j_str.size());

        // Stream data to connected clients via gRPC
        server.push_frame(orientations);
    });

    server.stop();
    return 0;
}
