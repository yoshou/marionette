#pragma once

#include "model.hpp"
#include "global_registeration.hpp"

#include <glm/glm.hpp>

struct articulation_results
{
    rigid_cluster parent_cluster;
    rigid_cluster child_cluster;
    std::vector<find_fit_result> parent_results;
    std::vector<find_fit_result> child_results;

    articulation_results() {}
    articulation_results(const rigid_cluster &parent_cluster, const rigid_cluster &child_cluster,
                         const std::vector<find_fit_result> &parent_results, const std::vector<find_fit_result> &child_results)
        : parent_cluster(parent_cluster), child_cluster(child_cluster), parent_results(parent_results), child_results(child_results)
    {}
};

std::tuple<std::map<std::string, std::size_t>, float, float> solve_articulation_chain_constraint(
    const model_data &model,
    const std::vector<articulation_results> &articulations,
    const std::string &from_cluster,
    const std::string &to_cluster);

std::tuple<glm::vec3, glm::mat4, glm::vec3, glm::mat4> compute_articuated_pair_points(
    const rigid_cluster &parent_cluster,
    const rigid_cluster &child_cluster,
    const find_fit_result &parent_result,
    const find_fit_result &child_result);
