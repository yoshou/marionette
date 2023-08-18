#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "camera_info.hpp"

glm::vec3 triangulation(const glm::mat4 &m1, const glm::mat4 &m2, const glm::vec2 &p1, const glm::vec2 &p2, glm::vec2 &s);

glm::vec3 triangulate(const glm::vec2 pt1, const glm::vec2 pt2, const camera_t &cm1, const camera_t &cm2);

glm::vec3 triangulate(const std::vector<glm::vec2>& points, const std::vector<camera_t> &cameras);
