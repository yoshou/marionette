#include <tuple>
#include <map>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <glm/gtx/string_cast.hpp>

#include "correspondance.hpp"
#include "camera_info.hpp"
#include "triangulation.hpp"
#include "utils.hpp"
#include "tuple_hash.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/graphviz.hpp>

namespace reconstruction
{
    std::map<triple<size_t>, triple<cv::Mat>> estimate_triple_fundamental_mat(cv::Mat m, const std::vector<triple<size_t>> &groups)
    {
        size_t cols = static_cast<size_t>(m.rows);

        std::map<std::tuple<size_t, size_t, size_t>, std::tuple<cv::Mat, cv::Mat, cv::Mat>> mats;

        for (auto group : groups)
        {
            std::vector<std::pair<size_t, size_t>> pairs = {
                std::make_pair(std::get<0>(group), std::get<1>(group)),
                std::make_pair(std::get<1>(group), std::get<2>(group)),
                std::make_pair(std::get<2>(group), std::get<0>(group)),
            };

            std::vector<cv::Mat> pair_mats;

            for (auto pair : pairs)
            {
                std::vector<cv::Point2f> points1;
                std::vector<cv::Point2f> points2;

                for (size_t row = 0; row < static_cast<size_t>(m.rows); row++)
                {
                    double x1 = m.at<double>(row, pair.first * 2);
                    double y1 = m.at<double>(row, pair.first * 2 + 1);
                    double x2 = m.at<double>(row, pair.second * 2);
                    double y2 = m.at<double>(row, pair.second * 2 + 1);

                    if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0)
                    {
                        continue;
                    }
                    points1.push_back(cv::Point2f(x1, y1));
                    points2.push_back(cv::Point2f(x2, y2));
                }

                cv::Mat mat = cv::findFundamentalMat(points1, points2, cv::FM_LMEDS);

                pair_mats.push_back(mat);
            }

            mats.insert(std::make_pair(group, std::make_tuple(pair_mats[0], pair_mats[1], pair_mats[2])));
        }

        return mats;
    }

    glm::mat3 calculate_fundametal_matrix(const glm::mat3 &camera_mat1, const glm::mat3 &camera_mat2,
                                          const glm::mat4 &camera_pose1, const glm::mat4 &camera_pose2)
    {
        const auto pose = camera_pose2 * glm::inverse(camera_pose1);

        const auto R = glm::mat3(pose);
        const auto t = glm::vec3(pose[3]);

        const auto Tx = glm::mat3(
            0, -t[2], t[1],
            t[2], 0, -t[0],
            -t[1], t[0], 0);

        const auto E = Tx * R;

        const auto F = glm::inverse(glm::transpose(camera_mat2)) * E * glm::inverse(camera_mat1);

        return F;
    }

    glm::mat3 calculate_fundametal_matrix(const camera_t &camera1, const camera_t &camera2)
    {
        const auto camera_mat1 = camera1.intrin.get_matrix();
        const auto camera_mat2 = camera2.intrin.get_matrix();

        return calculate_fundametal_matrix(camera_mat1, camera_mat2, camera1.extrin.rotation, camera2.extrin.rotation);
    }

    static glm::vec3 normalize_line(const glm::vec3 &v)
    {
        const auto c = std::sqrt(v.x * v.x + v.y * v.y);
        return v / c;
    }

    glm::vec3 compute_correspond_epiline(const glm::mat3 &F, const glm::vec2 &p)
    {
        const auto l = F * glm::vec3(p, 1.f);
        return normalize_line(l);
    }

    glm::vec3 compute_correspond_epiline(const camera_t &camera1, const camera_t &camera2, const glm::vec2 &p)
    {
        const auto F = calculate_fundametal_matrix(camera1, camera2);
        return compute_correspond_epiline(F, p);
    }

    size_t find_nearest_point(const glm::vec2 &pt, const std::vector<glm::vec2> &pts,
                              double thresh, float &dist)
    {
        size_t idx = pts.size();
        auto min_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < pts.size(); i++)
        {
            auto pt2 = pts[i];

            if (pt2.x < 0 || pt2.y < 0)
            {
                continue;
            }

            const auto dist = glm::distance(pt, pt2);

            if (dist < min_dist && dist < thresh)
            {
                min_dist = dist;
                idx = i;
            }
        }

        dist = min_dist;

        return idx;
    }

    struct edge_t
    {
        std::size_t u, v;
        float w;
    };

    template <typename F>
    static void dfs(const std::vector<std::vector<uint32_t>> &adj, std::size_t v, std::vector<std::uint32_t> &visited, F pred)
    {
        visited[v] = 1;
        pred(v);

        for (auto u : adj[v])
        {
            if (!visited[u])
            {
                dfs(adj, u, visited, pred);
            }
        }
    }

    void compute_connected_graphs(const std::vector<std::vector<uint32_t>> &adj, std::vector<std::vector<std::size_t>> &connected_graphs)
    {
        const auto n = adj.size();
        std::vector<std::uint32_t> visited(n, 0);

        connected_graphs.clear();
        for (std::size_t v = 0; v < n; v++)
        {
            if (!visited[v])
            {
                std::vector<std::size_t> connected_nodes;

                dfs(adj, v, visited, [&](std::size_t u)
                    { connected_nodes.push_back(u); });

                connected_graphs.push_back(connected_nodes);
            }
        }
    }

    static std::size_t get_node_group(std::size_t v, std::vector<std::size_t> offsets)
    {
        for (std::size_t i = 1; i < offsets.size(); i++)
        {
            if (v < offsets[i])
            {
                return i - 1;
            }
        }
        return offsets.size() - 1;
    }

    bool can_identify_group(const std::vector<node_t> &nodes, const std::vector<std::size_t> &graph)
    {
        std::unordered_set<uint32_t> groups;

        for (const auto v : graph)
        {
            const auto group = nodes[v].camera_idx;
            const auto [it, inserted] = groups.insert(group);
            if (!inserted)
            {
                return false;
            }
        }

        return true;
    }

    template <class Name>
    class label_writer
    {
    public:
        label_writer(Name _name) : name(_name) {}
        template <class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &v) const
        {
            out << "[label=\"" << name.at(v) << "\"]";
        }

    private:
        Name name;
    };

    static float compute_diff_camera_angle(const camera_t &camera1, const camera_t &camera2)
    {
        const auto r1 = glm::mat3(camera1.extrin.rotation);
        const auto r2 = glm::mat3(camera2.extrin.rotation);

        const auto r = glm::transpose(r1) * r2;
        const auto r_quat = glm::quat_cast(r);
        return glm::angle(r_quat);
    }

    using adj_list_t = std::vector<std::vector<uint32_t>>;

    void save_graphs(const std::vector<node_t> &nodes, const adj_list_t &adj, const std::string &prefix)
    {
        std::vector<std::vector<std::size_t>> connected_graphs;
        compute_connected_graphs(adj, connected_graphs);

        std::size_t count = 0;
        for (std::size_t i = 0; i < connected_graphs.size(); i++)
        {
            const auto &connected_graph = connected_graphs[i];
            if (connected_graph.size() < 2)
            {
                continue;
            }

            using graph_t = boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS>;
            graph_t g;

            std::map<graph_t::vertex_descriptor, std::string> labels;

            std::map<std::size_t, graph_t::vertex_descriptor> g_nodes;
            for (const auto v : connected_graph)
            {
                const auto node = boost::add_vertex(g);
                g_nodes.insert(std::make_pair(v, node));
                const auto c = nodes[v].camera_idx;
                const auto i = nodes[v].point_idx;
                labels[node] = std::to_string(c) + "," + std::to_string(i);
            }

            for (const auto v : connected_graph)
            {
                for (const auto u : adj[v])
                {
                    boost::add_edge(g_nodes[u], g_nodes[v], g);
                }
            }

            std::ofstream file(prefix + std::to_string(count++) + ".dot");
            boost::write_graphviz(file, g, label_writer<decltype(labels)>(labels));
        }
    }

    bool cut_ambiguous_edge(const std::vector<node_t> &nodes, adj_list_t &adj, const std::vector<camera_t> &cameras, double world_thresh)
    {
        std::vector<std::vector<std::size_t>> connected_graphs;
        compute_connected_graphs(adj, connected_graphs);

        // Remove ambiguous edges
        // If there is a path between the points of inconsitency, there is a combination of edges
        // with a large 3D point distance from the 2D point pair in the edge connecting to a node.
        for (const auto &g : connected_graphs)
        {
            const auto can_ident = can_identify_group(nodes, g);
            if (can_ident)
            {
                continue;
            }

            std::vector<std::pair<std::size_t, std::size_t>> remove_edges;
            for (const auto v : g)
            {
                std::vector<std::pair<std::size_t, std::size_t>> edges;
                for (const auto u : adj[v])
                {
                    edges.push_back(std::make_pair(u, v));
                }

                for (std::size_t i = 0; i < edges.size(); i++)
                {
                    for (std::size_t j = i + 1; j < edges.size(); j++)
                    {
                        const auto e1 = edges[i];
                        const auto e2 = edges[j];

                        const auto get_camera_point = [&nodes, &cameras](const std::size_t v)
                        {
                            return std::make_pair(nodes[v].pt, cameras[nodes[v].camera_idx]);
                        };

                        glm::vec3 marker1, marker2;
                        {
                            const auto [pt1, camera1] = get_camera_point(e1.first);
                            const auto [pt2, camera2] = get_camera_point(e1.second);
                            marker1 = triangulate(pt1, pt2, camera1, camera2);
                        }
                        {
                            const auto [pt1, camera1] = get_camera_point(e2.first);
                            const auto [pt2, camera2] = get_camera_point(e2.second);
                            marker2 = triangulate(pt1, pt2, camera1, camera2);
                        }

                        if (glm::distance(marker1, marker2) > world_thresh)
                        {
                            remove_edges.push_back(e1);
                            remove_edges.push_back(e2);
                        }
                    }
                }
            }

            for (const auto &e : remove_edges)
            {
                const auto u = e.first;
                const auto v = e.second;
                {
                    auto iter = std::remove_if(adj[v].begin(), adj[v].end(), [u](const size_t u_)
                                               { return u == u_; });
                    adj[v].erase(iter, adj[v].end());
                }
                {
                    auto iter = std::remove_if(adj[u].begin(), adj[u].end(), [v](const size_t v_)
                                               { return v == v_; });
                    adj[u].erase(iter, adj[u].end());
                }
            }
        }

        std::vector<std::vector<std::size_t>> new_connected_graphs;
        compute_connected_graphs(adj, new_connected_graphs);

        for (std::size_t i = 0; i < new_connected_graphs.size(); i++)
        {
            const auto &connected_graph = new_connected_graphs[i];
            const auto can_ident = can_identify_group(nodes, connected_graph);
            if (!can_ident)
            {
                return false;
            }
        }

        return true;
    }

    static cv::Mat glm2cv_mat3(const glm::mat4 &m)
    {
        cv::Mat ret(3, 3, CV_32F);
        for (std::size_t i = 0; i < 3; i++)
        {
            for (std::size_t j = 0; j < 3; j++)
            {
                ret.at<float>(i, j) = m[j][i];
            }
        }
        return ret;
    }

    static cv::Mat glm2cv_mat4(const glm::mat4 &m)
    {
        cv::Mat ret(4, 4, CV_32F);
        memcpy(ret.data, glm::value_ptr(m), 16 * sizeof(float));
        return ret;
    }

    static glm::vec2 undistort(const glm::vec2 &pt, const camera_t &camera)
    {
        auto pts = std::vector<cv::Point2f>{cv::Point2f(pt.x, pt.y)};
        cv::Mat m = glm2cv_mat3(camera.intrin.get_matrix());
        cv::Mat coeffs(5, 1, CV_32F);
        for (int i = 0; i < 5; i++)
        {
            coeffs.at<float>(i) = camera.intrin.coeffs[i];
        }

        std::vector<cv::Point2f> norm_pts;
        cv::undistortPoints(pts, norm_pts, m, coeffs);

        const auto p = camera.intrin.get_matrix() * glm::vec3(norm_pts[0].x, norm_pts[0].y, 1.0f);
        return glm::vec2(p.x / p.z, p.y / p.z);
    }

    void find_correspondance(const std::vector<std::vector<glm::vec2>> &pts,
                             const std::vector<camera_t> &cameras, std::vector<node_t> &nodes, adj_list_t &adj, double screen_thresh)
    {
        std::vector<std::pair<std::size_t, std::size_t>> camera_pairs;

        const auto num_cameras = cameras.size();

        for (std::size_t c1 = 0; c1 < cameras.size(); c1++)
        {
            for (std::size_t c2 = c1 + 1; c2 < cameras.size(); c2++)
            {
                camera_pairs.push_back(std::make_pair(c1, c2));
            }
        }

        std::vector<edge_t> edges;
        std::vector<std::size_t> node_offsets;
        for (std::size_t i = 0; i < pts.size(); i++)
        {
            node_offsets.push_back(nodes.size());

            for (std::size_t j = 0; j < pts[i].size(); j++)
            {
                nodes.push_back(node_t{
                    pts[i][j], i, j});
            }
        }

        const auto min_angle = 5;

        for (auto [c1, c2] : camera_pairs)
        {
            if (compute_diff_camera_angle(cameras[c1], cameras[c2]) < glm::pi<double>() / 180.0 * min_angle)
            {
                continue;
            }

            for (std::size_t i = 0; i < pts[c1].size(); i++)
            {
                const auto pt1 = undistort(pts[c1][i], cameras[c1]);

                if (pt1.x < 0 || pt1.y < 0)
                {
                    continue;
                }

                const auto line = compute_correspond_epiline(cameras[c1], cameras[c2], pt1);

                for (std::size_t j = 0; j < pts[c2].size(); j++)
                {
                    const auto pt2 = undistort(pts[c2][j], cameras[c2]);

                    if (pt2.x < 0 || pt2.y < 0)
                    {
                        continue;
                    }

                    const auto dist = distance_line_point(line, pt2);

                    if (dist > screen_thresh)
                    {
                        continue;
                    }

                    const auto marker = triangulate(pts[c1][i], pts[c2][j], cameras[c1], cameras[c2]);

                    float nearest_pt_dist_acc = dist;
                    size_t nearest_pt_dist_count = 1;

                    std::vector<std::size_t> observed_cameras = {c1, c2};

                    for (size_t c3 = 0; c3 < cameras.size(); c3++)
                    {
                        bool is_dup_camera = false;
                        for (const auto obs_camera : observed_cameras)
                        {
                            if (obs_camera == c3)
                            {
                                is_dup_camera = true;
                                break;
                            }
                            if (compute_diff_camera_angle(cameras[obs_camera], cameras[c3]) < glm::pi<double>() / 180.0 * min_angle)
                            {
                                is_dup_camera = true;
                                break;
                            }
                        }

                        if (is_dup_camera)
                        {
                            continue;
                        }

                        const auto pt = project(cameras[c3], marker);

                        if (pt.x < 0 || pt.y < 0)
                        {
                            continue;
                        }

                        float nearest_pt_dist;
                        const auto nearest_pt_idx = find_nearest_point(pt, pts[c3], screen_thresh, nearest_pt_dist);

                        if (nearest_pt_idx == pts[c3].size())
                        {
                            continue;
                        }

                        nearest_pt_dist_acc += nearest_pt_dist;
                        nearest_pt_dist_count++;

                        observed_cameras.push_back(c3);
                    }

                    float nearest_pt_dist = nearest_pt_dist_acc / nearest_pt_dist_count;
                    if (nearest_pt_dist_count <= 1)
                    {
                        nearest_pt_dist = std::numeric_limits<float>::max();
                    }

                    if (nearest_pt_dist > screen_thresh)
                    {
                        continue;
                    }

                    const auto u = node_offsets[c1] + i;
                    const auto v = node_offsets[c2] + j;
                    edges.push_back(edge_t{u, v, static_cast<float>(nearest_pt_dist_count)});
                }
            }
        }

        adj.clear();
        adj.resize(nodes.size());
        for (const auto &edge : edges)
        {
            adj[edge.u].push_back(edge.v);
            adj[edge.v].push_back(edge.u);
        }
    }
}
