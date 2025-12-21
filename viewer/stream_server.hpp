#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <memory>
#include <condition_variable>

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "sensor.grpc.pb.h"

class SensorServiceImpl;

class marker_stream_server
{
    using frame_type = std::vector<glm::quat>;
    std::atomic_bool running;
    std::shared_ptr<std::thread> server_th;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<SensorServiceImpl> service;

public:
    marker_stream_server();
    virtual ~marker_stream_server();

    void push_frame(const frame_type &frame);
    void run();
    void stop();
};
