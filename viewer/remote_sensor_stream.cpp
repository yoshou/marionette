#include "remote_sensor_stream.hpp"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "sensor.grpc.pb.h"

remote_sensor_stream::remote_sensor_stream(std::string endpoint)
    : stub_(stargazer::Sensor::NewStub(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials())))
{
}

void remote_sensor_stream::subscribe_sphere(const std::string& name, std::function<void(const std::vector<glm::vec3>&)> callback)
{
    grpc::ClientContext context;
    stargazer::SubscribeRequest request;
    request.set_name(name);

    std::unique_ptr<grpc::ClientReader<stargazer::SphereResponse>> reader(
        stub_->SubscribeSphere(&context, request));

    stargazer::SphereResponse response;
    while (reader->Read(&response))
    {
        std::vector<glm::vec3> markers;

        for (std::size_t i = 0; i < static_cast<std::size_t>(response.mutable_spheres()->values_size()); i++)
        {
            const auto sphere = response.mutable_spheres()->values(i);
            markers.push_back(glm::vec3(sphere.point().x(), sphere.point().y(), sphere.point().z()));
        }

        callback(markers);
    }
    grpc::Status status = reader->Finish();
    if (!status.ok())
    {
        throw std::runtime_error("subscribe_sphere rpc failed.");
    }
}

void remote_sensor_stream::subscribe_quat(const std::string &name, std::function<void(const std::vector<glm::quat> &)> callback)
{
    grpc::ClientContext context;
    stargazer::SubscribeRequest request;
    request.set_name(name);

    std::unique_ptr<grpc::ClientReader<stargazer::QuatResponse>> reader(
        stub_->SubscribeQuat(&context, request));

    stargazer::QuatResponse response;
    while (reader->Read(&response))
    {
        std::vector<glm::quat> quats;

        for (std::size_t i = 0; i < static_cast<std::size_t>(response.mutable_quats()->values_size()); i++)
        {
            const auto quat = response.mutable_quats()->values(i);
            quats.push_back(glm::quat(quat.w(), quat.x(), quat.y(), quat.z()));
        }

        callback(quats);
    }
    grpc::Status status = reader->Finish();
    if (!status.ok())
    {
        throw std::runtime_error("subscribe_sphere rpc failed.");
    }
}
