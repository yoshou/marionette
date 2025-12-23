#include "stream_server.hpp"
#include <iostream>
#include <unordered_map>
#include <spdlog/spdlog.h>

class SensorServiceImpl final : public stargazer::Sensor::Service
{
    std::mutex mtx;
    std::unordered_map<std::string, grpc::ServerWriter<stargazer::QuatMessage> *> writers;

public:
    void notify_quat(const std::vector<glm::quat> &quats)
    {
        stargazer::QuatMessage response;
        for (const auto &quat : quats)
        {
            auto mutable_quat = response.add_values();
            mutable_quat->set_x(quat.x);
            mutable_quat->set_y(quat.y);
            mutable_quat->set_z(quat.z);
            mutable_quat->set_w(quat.w);
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            for (const auto &[name, writer] : writers)
            {
                writer->Write(response);
            }
        }
    }

    grpc::Status SubscribeQuat(grpc::ServerContext *context,
                               const stargazer::SubscribeRequest *request,
                               grpc::ServerWriter<stargazer::QuatMessage> *writer) override
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            writers.insert(std::make_pair(request->name(), writer));
        }
        while (!context->IsCancelled())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (const auto iter = writers.find(request->name()); iter != writers.end())
            {
                writers.erase(iter);
            }
        }
        return grpc::Status::OK;
    }
};

marker_stream_server::marker_stream_server()
    : service(new SensorServiceImpl()) {}
marker_stream_server::~marker_stream_server() = default;

void marker_stream_server::push_frame(const frame_type &frame)
{
    if (!running)
    {
        return;
    }

    service->notify_quat(frame);
}

void marker_stream_server::run()
{
    running = true;
    server_th.reset(new std::thread([this]()
                                    {
        std::string server_address("0.0.0.0:50052");

        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(service.get());
        server = builder.BuildAndStart();
        spdlog::info("Server listening on " + server_address);
        server->Wait(); }));
}

void marker_stream_server::stop()
{
    if (running.load())
    {
        running.store(false);
        server->Shutdown();
        if (server_th && server_th->joinable())
        {
            server_th->join();
        }
    }
}
