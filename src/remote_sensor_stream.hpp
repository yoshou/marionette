#pragma once

#include "sensor.grpc.pb.h"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <functional>

class remote_sensor_stream
{
public:
    remote_sensor_stream(std::string endpoint);
    void subscribe_sphere(const std::string& name, std::function<void(const std::vector<glm::vec3>&)> callback);
    void subscribe_quat(const std::string& name, std::function<void(const std::vector<glm::quat>&)> callback);

private:
    std::unique_ptr<stargazer::Sensor::Stub> stub_;
};
