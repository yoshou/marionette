#include "model.hpp"
#include "global_registeration.hpp"
#include "articulation_solver.hpp"
#include "registration.hpp"

#include <iostream>
#include <glm/glm.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

static auto solve_articuated_pair(
    const rigid_cluster &parent_cluster, const rigid_cluster &child_cluster,
    const std::vector<find_fit_result> &parent_results, const std::vector<find_fit_result> &child_results)
{
    std::size_t min_i;
    std::size_t min_j;
    float min_dist = std::numeric_limits<float>::max();
    glm::vec3 vec1, vec2;
    glm::mat4 rot1, rot2;

    const auto articulation_pos = child_cluster.pose[3];
    const auto articulation_rot = child_cluster.pose;
    for (std::size_t i = 0; i < parent_results.size(); i++)
    {
        const auto &parent_result = parent_results[i];

        const auto anchor_pos = twist(parent_cluster.points[0], parent_cluster.pose, glm::inverse(parent_cluster.pose), parent_result.twist_angle);
        const auto transform = glm::translate(glm::mat4(1.f), parent_result.translation) * glm::mat4(parent_result.rotation) * glm::translate(glm::mat4(1.f), -anchor_pos);

        const auto articulation_pos_a = transform * articulation_pos;
        const auto articulation_rot_a = transform * articulation_rot;

        for (std::size_t j = 0; j < child_results.size(); j++)
        {
            const auto &child_result = child_results[j];

            const auto anchor_pos = twist(child_cluster.points[0], child_cluster.pose, glm::inverse(child_cluster.pose), child_result.twist_angle);
            const auto transform = glm::translate(glm::mat4(1.f), child_result.translation) * glm::mat4(child_result.rotation) * glm::translate(glm::mat4(1.f), -anchor_pos);

            const auto articulation_pos_b = transform * articulation_pos;
            const auto articulation_rot_b = transform * articulation_rot;

            const auto dist = glm::distance(glm::vec3(articulation_pos_a), glm::vec3(articulation_pos_b));
            const auto error = dist + parent_result.error + child_result.error;
            if (error < min_dist)
            {
                min_dist = error;
                min_i = i;
                min_j = j;
                vec1 = articulation_pos_a;
                vec2 = articulation_pos_b;
                rot1 = articulation_rot_a;
                rot2 = articulation_rot_b;
            }
        }
    }

    return std::make_tuple(min_i, min_j, vec1, vec2);
}

struct vertex_property
{
    std::string cluster_name;
    std::size_t candidate_idx;
};

struct edge_property
{
    float distance;
};

typedef boost::adjacency_list<
    boost::listS, boost::vecS, boost::undirectedS,
    vertex_property,
    edge_property>
    graph_t;

typedef graph_t::edge_descriptor edge_t;
typedef graph_t::vertex_descriptor vertex_t;
class my_visitor : boost::default_bfs_visitor
{
protected:
    vertex_t dest;

public:
    my_visitor(vertex_t dest)
        : dest(dest){};
    void initialize_vertex(const vertex_t &s, const graph_t &g) const {}
    void discover_vertex(const vertex_t &s, const graph_t &g) const {}
    void examine_vertex(const vertex_t &s, const graph_t &g) const {}
    void examine_edge(const edge_t &e, const graph_t &g) const {}
    void edge_relaxed(const edge_t &e, const graph_t &g) const {}
    void edge_not_relaxed(const edge_t &e, const graph_t &g) const {}
    void finish_vertex(const vertex_t &s, const graph_t &g) const
    {
        if (dest == s)
        {
            throw std::exception();
        }
    }
};

static std::deque<vertex_t> get_path(vertex_t from, vertex_t to, std::vector<vertex_t> parents)
{
    std::deque<vertex_t> path;
    for (vertex_t v = to;; v = parents[v])
    {
        path.push_front(v);
        if (v == from)
        {
            break;
        }
    }
    return path;
}

static void print_path(const graph_t& g, const std::deque<vertex_t> &path, float distance_to)
{
    for (const auto &v : path)
    {
        std::cout << g[v].cluster_name << ", " << g[v].candidate_idx;
        if (v != path.back())
        {
            std::cout << " -> ";
        }
    }
    std::cout << std::endl;
    std::cout << "Total distance : " << distance_to << std::endl;
}

std::tuple<std::map<std::string, std::size_t>, float, float> solve_articulation_chain_constraint(const model_data &model, const std::vector<articulation_results> &articulations, const std::string &from_cluster, const std::string &to_cluster)
{
    graph_t g;
    std::unordered_map<std::string, std::vector<vertex_t>> node_ids;

    const auto from = boost::add_vertex({"Source", 0}, g);
    const auto to = boost::add_vertex({"Distination", 0}, g);

    // Set candidates as vertices.
    for (std::size_t k = 0; k < articulations.size(); k++)
    {
        {
            const auto &parent_cluster = articulations[k].parent_cluster;
            const auto &parent_results = articulations[k].parent_results;

            if (node_ids.find(parent_cluster.name) == node_ids.end())
            {
                std::vector<vertex_t> ids;
                for (vertex_t i = 0; i < parent_results.size(); i++)
                {
                    ids.push_back(boost::add_vertex({parent_cluster.name, i}, g));
                }
                node_ids.insert(std::make_pair(parent_cluster.name, ids));
            }
        }

        {
            const auto &child_cluster = articulations[k].child_cluster;
            const auto &child_results = articulations[k].child_results;

            if (node_ids.find(child_cluster.name) == node_ids.end())
            {
                std::vector<vertex_t> ids;
                for (vertex_t i = 0; i < child_results.size(); i++)
                {
                    ids.push_back(boost::add_vertex({child_cluster.name, i}, g));
                }
                node_ids.insert(std::make_pair(child_cluster.name, ids));
            }
        }
    }
    
    // Set articuation distance as edge value.
    for (std::size_t k = 0; k < articulations.size(); k++)
    {
        const auto &parent_cluster = articulations[k].parent_cluster;
        const auto &child_cluster = articulations[k].child_cluster;

        const auto &parent_results = articulations[k].parent_results;
        const auto &child_results = articulations[k].child_results;

        const auto articulation_pos = child_cluster.pose[3];
        const auto articulation_rot = child_cluster.pose;

        std::vector<glm::vec3> articulation_pos_a_candidates(parent_results.size());
        std::vector<glm::vec3> articulation_pos_b_candidates(child_results.size());
        std::vector<glm::mat3> articulation_rot_a_candidates(parent_results.size());
        std::vector<glm::mat3> articulation_rot_b_candidates(child_results.size());

        const auto parent_inv_pose = glm::inverse(parent_cluster.pose);
        const auto child_inv_pose = glm::inverse(child_cluster.pose);

        auto angle_axis_a = to_line(parent_cluster.pose);
        auto angle_axis_b = to_line(child_cluster.pose);

        for (std::size_t i = 0; i < parent_results.size(); i++)
        {
            const auto &parent_result = parent_results[i];

            const auto anchor_pos = twist(parent_cluster.points[parent_result.anchor_index], angle_axis_a, parent_result.twist_angle);
            const auto transform_a = glm::translate(parent_result.translation) * glm::mat4(parent_result.rotation) * glm::translate(-anchor_pos);

            const auto articulation_pos_a = transform_a * articulation_pos;
            const auto articulation_rot_a = transform_a * articulation_rot;

            articulation_pos_a_candidates[i] = articulation_pos_a;
            articulation_rot_a_candidates[i] = articulation_rot_a;
        }

        for (std::size_t j = 0; j < child_results.size(); j++)
        {
            const auto &child_result = child_results[j];

            const auto anchor_pos = twist(child_cluster.points[child_result.anchor_index], angle_axis_b, child_result.twist_angle);
            const auto transform_b = glm::translate(child_result.translation) * glm::mat4(child_result.rotation) * glm::translate(-anchor_pos);

            const auto articulation_pos_b = transform_b * articulation_pos;
            const auto articulation_rot_b = transform_b * articulation_rot;

            articulation_pos_b_candidates[j] = articulation_pos_b;
            articulation_rot_b_candidates[j] = articulation_rot_b;
        }
        point_cloud articulation_pos_b_candidates_cloud(articulation_pos_b_candidates);
        articulation_pos_b_candidates_cloud.build_index();

        const auto &node_idx_a = node_ids.at(parent_cluster.name);
        const auto &node_idx_b = node_ids.at(child_cluster.name);
        for (std::size_t i = 0; i < parent_results.size(); i++)
        {
            const auto &parent_result = parent_results[i];

            const auto &articulation_pos_a = articulation_pos_a_candidates[i];
            const auto &articulation_rot_a = articulation_rot_a_candidates[i];

            std::vector<glm::vec3> parent_points;
            transform_to_target(parent_cluster, parent_result, parent_points);

            std::vector<std::pair<point_cloud::index_type, float>> result;
            articulation_pos_b_candidates_cloud.radius_search(articulation_pos_a, 0.01f, result);

            // for (std::size_t j = 0; j < child_results.size(); j++)
            for (std::size_t k = 0; k < result.size(); k++)
            {
                const auto j = result[k].first;
                const auto &child_result = child_results[j];

                const auto &articulation_pos_b = articulation_pos_b_candidates[j];
                const auto &articulation_rot_b = articulation_rot_b_candidates[j];

                std::vector<glm::vec3> child_points;
                transform_to_target(child_cluster, child_result, child_points);

                float constraint_error = 0.f;
                std::size_t num_constraints = 0;
                for (std::size_t i = 0; i < model.constraints.size(); i++)
                {
                    const auto &point1 = model.constraints[i].point1;
                    const auto &point2 = model.constraints[i].point2;

                    const auto parent_name = point1.cluster_name;
                    const auto child_name = point2.cluster_name;

                    if (parent_cluster.name != parent_name || child_cluster.name != child_name)
                    {
                        continue;
                    }

                    const auto pos1 = parent_points[point1.index];
                    const auto pos2 = child_points[point2.index];

                    constraint_error += glm::distance(pos1, pos2);
                    num_constraints++;
                }

                if (num_constraints > 0)
                {
                    constraint_error /= num_constraints;
                }

                const auto articulation_dist_sq = result[k].second;
                const auto articulation_dist = std::sqrt(articulation_dist_sq);
                // const auto dist = (parent_result.error + child_result.error + constraint_error * 0.5 + articulation_dist);
                const auto dist = ((parent_result.error + child_result.error) + articulation_dist * 0.5f + constraint_error * 0.1f);

                const auto u = node_idx_a[i];
                const auto v = node_idx_b[j];
                boost::add_edge(u, v, {dist}, g);
            }
        }
    }

    // Set source and distination vertex
    {
        const auto u = from;
        for (const auto& v : node_ids[from_cluster])
        {
            boost::add_edge(u, v, {0.f}, g);
        }
    }

    {
        const auto u = to;
        for (const auto &v : node_ids[to_cluster])
        {
            boost::add_edge(u, v, {0.f}, g);
        }
    }

    // Solve the combination of the smallest distance condidates by Dijkstra algorithm.
    my_visitor vis(to);
    std::vector<vertex_t> parents(num_vertices(g));
    std::vector<float> distance(num_vertices(g));

    try {
        boost::dijkstra_shortest_paths(g, from,
            boost::weight_map(boost::get(&edge_property::distance, g)).
                  predecessor_map(&parents[0]).
                  distance_map(&distance[0]).
                  visitor(vis)
        );
    } catch (std::exception& e) {
    }

    if (parents[to] == to)
    {
        return std::make_tuple(std::map<std::string, std::size_t>{}, std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    }
    else
    {
        const auto path = get_path(from, to, parents);

        std::map<std::string, std::size_t> result;
        for (const auto& v : path)
        {
            const auto cluster_name = g[v].cluster_name;
            const auto candidate_idx = g[v].candidate_idx;

            result.insert(std::make_pair(cluster_name, candidate_idx));
        }

        float max_dist = 0.f;
        for (std::size_t i = 1; i < path.size() - 2; i++)
        {
            const auto u = path[i];
            const auto v = path[i + 1];
            const auto e = boost::edge(u, v, g);
            const auto dist = g[e.first].distance;

            max_dist = std::max(max_dist, dist);
        }

        return std::make_tuple(result, distance[to] / (path.size() - 3), max_dist);
    }
}

std::tuple<glm::vec3, glm::mat4, glm::vec3, glm::mat4> compute_articuated_pair_points(
    const rigid_cluster &parent_cluster, const rigid_cluster &child_cluster,
    const find_fit_result &parent_result, const find_fit_result &child_result)
{
    glm::vec3 articulation_pos_a, articulation_pos_b;
    glm::mat4 articulation_rot_a, articulation_rot_b;

    const auto articulation_pos = child_cluster.pose[3];
    const auto articulation_rot = child_cluster.pose;
    {
        const auto anchor_pos = twist(parent_cluster.points[0], parent_cluster.pose, glm::inverse(parent_cluster.pose), parent_result.twist_angle);
        const auto transform = glm::translate(parent_result.translation) * glm::mat4(parent_result.rotation) * glm::translate(glm::mat4(1.f), -anchor_pos);

        articulation_pos_a = transform * articulation_pos;
        articulation_rot_a = transform * articulation_rot;
    }

    {
        const auto anchor_pos = twist(child_cluster.points[0], child_cluster.pose, glm::inverse(child_cluster.pose), child_result.twist_angle);
        const auto transform = glm::translate(child_result.translation) * glm::mat4(child_result.rotation) * glm::translate(glm::mat4(1.f), -anchor_pos);

        articulation_pos_b = transform * articulation_pos;
        articulation_rot_b = transform * articulation_rot;
    }

    return std::make_tuple(articulation_pos_a, articulation_rot_a, articulation_pos_b, articulation_rot_b);
}

static auto compute_paired_points(const glm::vec3& point1, const glm::vec3& point2,
    const rigid_cluster &parent_cluster, const rigid_cluster &child_cluster,
    const find_fit_result &parent_result, const find_fit_result &child_result)
{
    glm::vec3 pos_a, pos_b;

    {
        const auto anchor_pos = twist(parent_cluster.points[0], parent_cluster.pose, glm::inverse(parent_cluster.pose), parent_result.twist_angle);
        const auto transform = glm::translate(parent_result.translation) * glm::mat4(parent_result.rotation) * glm::translate(glm::mat4(1.f), -anchor_pos);

        pos_a = transform_coordinate(point1, transform);
    }

    {
        const auto anchor_pos = twist(child_cluster.points[0], child_cluster.pose, glm::inverse(child_cluster.pose), child_result.twist_angle);
        const auto transform = glm::translate(child_result.translation) * glm::mat4(child_result.rotation) * glm::translate(glm::mat4(1.f), -anchor_pos);

        pos_b = transform_coordinate(point2, transform);
    }

    return std::make_tuple(pos_a, pos_b);
}
