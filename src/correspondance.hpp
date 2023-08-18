#pragma once

#include <tuple>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "camera_info.hpp"

namespace reconstruction
{
template <typename T>
using triple = std::tuple<T, T, T>;

std::map<triple<size_t>, triple<cv::Mat>> estimate_triple_fundamental_mat(cv::Mat m, const std::vector<triple<size_t>>& groups);

glm::mat3 calculate_fundametal_matrix(const glm::mat3 &camera_mat1, const glm::mat3 &camera_mat2,
    const glm::mat4 &camera_pose1, const glm::mat4 &camera_pose2);

glm::mat3 calculate_fundametal_matrix(const camera_t &camera1, const camera_t &camera2);

glm::vec3 compute_correspond_epiline(const glm::mat3 &F, const glm::vec2 &p);

glm::vec3 compute_correspond_epiline(const camera_t &camera1, const camera_t &camera2, const glm::vec2 &p);

size_t find_nearest_point(const glm::vec2 &pt, const std::vector<glm::vec2> &pts,
    double thresh, float &dist);

struct node_t
{
    glm::vec2 pt;
    std::size_t camera_idx;
    std::size_t point_idx;
};

using adj_list_t = std::vector<std::vector<uint32_t>>;

void compute_connected_graphs(const std::vector<std::vector<uint32_t>> &adj, std::vector<std::vector<std::size_t>> &connected_graphs);

bool can_identify_group(const std::vector<node_t> &nodes, const std::vector<std::size_t> &graph);

void save_graphs(const std::vector<node_t> &nodes, const adj_list_t &adj, const std::string &prefix);

bool cut_ambiguous_edge(const std::vector<node_t> &nodes, adj_list_t &adj, const std::vector<camera_t> &cameras, double world_thresh = 0.03);

void find_correspondance(const std::vector<std::vector<glm::vec2>> &pts,
                         const std::vector<camera_t> &cameras, std::vector<node_t> &nodes, adj_list_t &adj, double screen_thresh);
}
