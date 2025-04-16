#include "remote_sensor_stream.hpp"

#include <fstream>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "sensor.grpc.pb.h"

#define USE_SECURE_CREDENTIALS

static std::string read_text_file(const std::string& path)
{
    std::ifstream ifs;
    ifs.open(path, std::ios::binary | std::ios::in);

    std::istreambuf_iterator<char> beg(ifs);
    std::istreambuf_iterator<char> end;
    std::vector<char> str_data;
    std::copy(beg, end, std::back_inserter(str_data));
    std::string str(str_data.begin(), str_data.end());

    return str;
}

static std::shared_ptr<grpc::ChannelCredentials> create_channel_credentials()
{
#ifdef USE_SECURE_CREDENTIALS
    grpc::SslCredentialsOptions ssl_opts;
    std::string ca_crt_content = read_text_file("../data/ca.crt");
    std::string client_crt_content = read_text_file("../data/client.crt");
    std::string client_key_content = read_text_file("../data/client.key");

    ssl_opts.pem_cert_chain = client_crt_content;
    ssl_opts.pem_private_key = client_key_content;
    ssl_opts.pem_root_certs = ca_crt_content;
    return grpc::SslCredentials(ssl_opts);
#else
    return grpc::InsecureChannelCredentials();
#endif
}

remote_sensor_stream::remote_sensor_stream(std::string endpoint)
    : stub_(stargazer::Sensor::NewStub(grpc::CreateChannel(endpoint, create_channel_credentials())))
{
}

void remote_sensor_stream::subscribe_sphere(const std::string& name, std::function<void(const std::vector<glm::vec3>&)> callback)
{
    grpc::ClientContext context;
    stargazer::SubscribeRequest request;
    request.set_name(name);

    std::unique_ptr<grpc::ClientReader<stargazer::SphereMessage>> reader(
        stub_->SubscribeSphere(&context, request));

    stargazer::SphereMessage response;
    while (reader->Read(&response))
    {
        std::vector<glm::vec3> markers;

        for (std::size_t i = 0; i < static_cast<std::size_t>(response.values_size()); i++)
        {
            const auto sphere = response.values(i);
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

    std::unique_ptr<grpc::ClientReader<stargazer::QuatMessage>> reader(
        stub_->SubscribeQuat(&context, request));

    stargazer::QuatMessage response;
    while (reader->Read(&response))
    {
        std::vector<glm::quat> quats;

        for (std::size_t i = 0; i < static_cast<std::size_t>(response.values_size()); i++)
        {
            const auto quat = response.values(i);
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
