#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <tuple>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

template <typename T>
static std::string get_typename()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return "float32";
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return "float64";
    }
    else if constexpr (std::is_same_v<T, uint32_t>)
    {
        return "uint32";
    }
    throw std::runtime_error("Invalid type");
}

template <typename T>
using MatrixX = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
using MatrixXf = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

template <typename T>
static std::tuple<std::vector<T>, std::vector<uint32_t>> load_tensor(const std::string &filename)
{
    std::ifstream ifs;
    ifs.open(filename, std::ios::in);
    nlohmann::json j = nlohmann::json::parse(ifs);
    if (j["type"].get<std::string>() != get_typename<T>())
    {
        throw std::runtime_error("Invalid type");
    }
    const auto data = j["data"].get<std::vector<T>>();
    const auto shape = j["shape"].get<std::vector<uint32_t>>();
    return std::forward_as_tuple(data, shape);
}

template <typename T>
static MatrixX<T> load_dense_matrix(const std::string &filename)
{
    std::ifstream ifs;
    ifs.open(filename, std::ios::in);
    nlohmann::json j = nlohmann::json::parse(ifs);
    if (j["type"].get<std::string>() != get_typename<T>())
    {
        throw std::runtime_error("Invalid type");
    }
    const auto data = j["data"].get<std::vector<T>>();
    const auto shape = j["shape"].get<std::vector<uint32_t>>();
    if (shape.size() != 2)
    {
        throw std::runtime_error("Invalid shape");
    }

    MatrixX<T> dense = MatrixX<T>::Zero(shape[0], shape[1]);
    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            dense(i, j) = data[i * shape[1] + j];
        }
    }
    return dense;
}

template <typename T>
static MatrixX<T> load_sparse_matrix(const std::string &filename)
{
    std::ifstream ifs;
    ifs.open(filename, std::ios::in);
    nlohmann::json j = nlohmann::json::parse(ifs);
    if (j["type"].get<std::string>() != get_typename<T>())
    {
        throw std::runtime_error("Invalid type");
    }
    const auto data = j["data"].get<std::vector<T>>();
    const auto row = j["row"].get<std::vector<uint32_t>>();
    const auto col = j["col"].get<std::vector<uint32_t>>();
    const auto shape = j["shape"].get<std::vector<uint32_t>>();
    if (shape.size() != 2)
    {
        throw std::runtime_error("Invalid shape");
    }

    MatrixX<T> dense = MatrixX<T>::Zero(shape[0], shape[1]);
    for (size_t i = 0; i < data.size(); i++)
    {
        dense(row[i], col[i]) = data[i];
    }
    return dense;
}

static std::vector<size_t> calculate_steps(const std::vector<uint32_t> &shape)
{
    std::vector<size_t> steps(shape.size() - 1);
    std::partial_sum(shape.rbegin(), shape.rend() - 1, steps.rbegin(), std::multiplies{});
    steps.push_back(1);
    return steps;
}

static size_t calculate_size(const std::vector<uint32_t> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies{});
}

template <typename T>
static std::tuple<std::vector<T>, std::vector<uint32_t>> blend_shapes(
    const std::vector<T> &betas,
    const std::vector<uint32_t> &betas_shape,
    const std::vector<float> &shape_disps,
    const std::vector<uint32_t> &shape_disps_shape)
{
    const auto betas_steps = calculate_steps(betas_shape);
    const auto shape_disps_steps = calculate_steps(shape_disps_shape);

    const std::vector<uint32_t> blend_shapes_shape = {betas_shape[0], shape_disps_shape[0], shape_disps_shape[1]};
    const auto blend_shapes_steps = calculate_steps(blend_shapes_shape);

    std::vector<T> blend_shapes(std::accumulate(blend_shapes_shape.begin(), blend_shapes_shape.end(), 1, std::multiplies{}));

    for (size_t b = 0; b < betas_shape[0]; b++)
    {
        for (size_t m = 0; m < shape_disps_shape[0]; m++)
        {
            for (size_t k = 0; k < shape_disps_shape[1]; k++)
            {
                for (size_t l = 0; l < shape_disps_shape[2]; l++)
                {
                    const auto dst_idx = b * blend_shapes_steps[0] + m * blend_shapes_steps[1] + k * blend_shapes_steps[2];
                    blend_shapes[dst_idx] += betas[b * betas_steps[0] + l * betas_steps[1]] * T(shape_disps[m * shape_disps_steps[0] + k * shape_disps_steps[1] + l * shape_disps_steps[2]]);
                }
            }
        }
    }

    return std::forward_as_tuple(blend_shapes, blend_shapes_shape);
}

template <typename T>
static std::tuple<std::vector<T>, std::vector<uint32_t>> vertices2joints(
    const MatrixXd &j_regressor,
    const std::vector<T> &vertices,
    const std::vector<uint32_t> &vertices_shape)
{
    const auto vertices_steps = calculate_steps(vertices_shape);

    const std::vector<uint32_t> joints_shape = {vertices_shape[0], static_cast<uint32_t>(j_regressor.rows()), vertices_shape[2]};
    const auto joints_steps = calculate_steps(joints_shape);

    std::vector<T> joints(calculate_size(joints_shape));

    for (size_t b = 0; b < vertices_shape[0]; b++)
    {
        for (size_t j = 0; j < static_cast<uint32_t>(j_regressor.rows()); j++)
        {
            for (size_t k = 0; k < vertices_shape[2]; k++)
            {
                for (size_t i = 0; i < vertices_shape[1]; i++)
                {
                    const auto dst_idx = b * joints_steps[0] + j * joints_steps[1] + k * joints_steps[2];
                    joints[dst_idx] += vertices[b * vertices_steps[0] + i * vertices_steps[1] + k * vertices_steps[2]] * T(j_regressor(j, i));
                }
            }
        }
    }

    return std::forward_as_tuple(joints, joints_shape);
}

template <typename T>
inline T dot_product(const T x[3], const T y[3])
{
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template <typename T>
inline void rodrigues(const T angle_axis[3],
                      T result[9])
{
    static const T one = T(1.0);
    const T theta2 = dot_product(angle_axis, angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon()))
    {
        // We want to be careful to only evaluate the square root if the
        // norm of the angle_axis vector is greater than zero. Otherwise
        // we get a division by zero.
        const T theta = sqrt(theta2);
        const T wx = angle_axis[0] / theta;
        const T wy = angle_axis[1] / theta;
        const T wz = angle_axis[2] / theta;

        const T costheta = cos(theta);
        const T sintheta = sin(theta);

        result[0] = costheta + wx * wx * (one - costheta);
        result[3] = wz * sintheta + wx * wy * (one - costheta);
        result[6] = -wy * sintheta + wx * wz * (one - costheta);
        result[1] = wx * wy * (one - costheta) - wz * sintheta;
        result[4] = costheta + wy * wy * (one - costheta);
        result[7] = wx * sintheta + wy * wz * (one - costheta);
        result[2] = wy * sintheta + wx * wz * (one - costheta);
        result[5] = -wx * sintheta + wy * wz * (one - costheta);
        result[8] = costheta + wz * wz * (one - costheta);
    }
    else
    {
        // Near zero, we switch to using the first order Taylor expansion.
        result[0] = one;
        result[3] = angle_axis[2];
        result[6] = -angle_axis[1];
        result[1] = -angle_axis[2];
        result[4] = one;
        result[7] = angle_axis[0];
        result[2] = angle_axis[1];
        result[5] = -angle_axis[0];
        result[8] = one;
    }
}

template <typename T>
static std::tuple<std::vector<T>, std::vector<uint32_t>> batch_rodrigues(
    const std::vector<T> &rot_vecs,
    const std::vector<uint32_t> &rot_vecs_shape)
{
    const auto num_rots = rot_vecs_shape.back() / 3;
    const auto num_batch = rot_vecs_shape.front();

    const auto rot_vecs_steps = calculate_steps(rot_vecs_shape);

    std::vector<uint32_t> rot_mats_shape = rot_vecs_shape;
    rot_mats_shape.back() = num_rots * 9;
    const auto rot_mats_steps = calculate_steps(rot_mats_shape);

    std::vector<T> rot_mats(calculate_size(rot_mats_shape));
    for (size_t b = 0; b < num_batch; b++)
    {
        for (size_t i = 0; i < num_rots; i++)
        {
            std::array<T, 3> rot_vec;
            for (size_t j = 0; j < 3; j++)
            {
                rot_vec[j] = rot_vecs[b * rot_vecs_steps[0] + i * 3 + j];
            }

            std::array<T, 9> rot_mat;
            rodrigues(rot_vec.data(), rot_mat.data());

            for (size_t j = 0; j < 9; j++)
            {
                rot_mats[b * rot_mats_steps[0] + i * 9 + j] = rot_mat[j];
            }
        }
    }

    return std::forward_as_tuple(rot_mats, rot_mats_shape);
}

template <typename T, typename T1, typename T2>
static std::tuple<std::vector<T>, std::vector<T>> batch_rigid_transform(
    const std::vector<T1> &rot_mats,
    const std::vector<uint32_t> &rot_mats_shape,
    const std::vector<T2> &joints,
    const std::vector<uint32_t> &joints_shape,
    const std::vector<int32_t> &parents)
{
    const auto joints_steps = calculate_steps(joints_shape);
    const auto rot_mats_steps = calculate_steps(rot_mats_shape);
    const auto batch_size = rot_mats_shape[0];
    const auto joint_size = static_cast<uint32_t>(parents.size());
    assert(joints_shape[0] == 1);
    assert(joints_shape[1] == joint_size);
    assert(joints_shape[2] == 3);
    assert(rot_mats_shape[1] / 9 + 1 == joint_size);

    std::vector<uint32_t> transform_mats_shape = {batch_size, joint_size, 16};
    const auto transform_mats_steps = calculate_steps(transform_mats_shape);
    std::vector<T> transform_mats(calculate_size(transform_mats_shape));
    std::vector<T> rel_transform_mats(calculate_size(transform_mats_shape));

    std::vector<uint32_t> posed_joints_shape = {batch_size, joint_size, 3};
    const auto posed_joints_steps = calculate_steps(posed_joints_shape);
    std::vector<T> posed_joints(calculate_size(posed_joints_shape));

    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < joint_size; j++)
        {
            std::array<T, 3> joint;
            for (size_t k = 0; k < 3; k++)
            {
                joint[k] = joints[j * joints_steps[1] + k * joints_steps[2]];
            }

            std::array<T, 3> parent_joint;
            if (j != 0)
            {
                assert(parents[j] >= 0);
                for (size_t k = 0; k < 3; k++)
                {
                    parent_joint[k] = joints[parents[j] * joints_steps[1] + k * joints_steps[2]];
                }
            }
            else
            {
                for (size_t k = 0; k < 3; k++)
                {
                    parent_joint[k] = T(0.0);
                }
            }

            std::array<T, 3> rel_joint;
            for (size_t k = 0; k < 3; k++)
            {
                rel_joint[k] = joint[k] - parent_joint[k];
            }

            std::array<std::array<T, 4>, 4> transform_mat;
            for (size_t m = 0; m < 4; m++)
            {
                for (size_t n = 0; n < 4; n++)
                {
                    if (m == n)
                    {
                        transform_mat[m][n] = T(1.0);
                    }
                    else
                    {
                        transform_mat[m][n] = T(0.0);
                    }
                }
            }
            if (j != 0)
            {
                for (size_t m = 0; m < 3; m++)
                {
                    for (size_t n = 0; n < 3; n++)
                    {
                        transform_mat[m][n] = T(rot_mats[i * rot_mats_steps[0] + (j - 1) * 9 + m * 3 + n]);
                    }
                    transform_mat[m][3] = rel_joint[m];
                }
            }
            for (size_t m = 0; m < 3; m++)
            {
                transform_mat[m][3] = rel_joint[m];
            }

            std::array<std::array<T, 4>, 4> parent_transform_mat;
            for (size_t m = 0; m < 4; m++)
            {
                for (size_t n = 0; n < 4; n++)
                {
                    if (m == n)
                    {
                        parent_transform_mat[m][n] = T(1.0);
                    }
                    else
                    {
                        parent_transform_mat[m][n] = T(0.0);
                    }
                }
            }
            if (j != 0)
            {
                assert(parents[j] >= 0);
                for (size_t m = 0; m < 4; m++)
                {
                    for (size_t n = 0; n < 4; n++)
                    {
                        parent_transform_mat[m][n] = transform_mats[i * transform_mats_steps[0] + parents[j] * transform_mats_steps[1] + m * 4 + n];
                    }
                }
            }

            std::array<std::array<T, 4>, 4> global_transform_mat;
            for (size_t m = 0; m < 4; m++)
            {
                for (size_t n = 0; n < 4; n++)
                {
                    T acc(0.0);
                    for (size_t k = 0; k < 4; k++)
                    {
                        acc += parent_transform_mat[m][k] * transform_mat[k][n];
                    }
                    global_transform_mat[m][n] = acc;
                }
            }

            for (size_t m = 0; m < 4; m++)
            {
                for (size_t n = 0; n < 4; n++)
                {
                    transform_mats[i * transform_mats_steps[0] + j * transform_mats_steps[1] + m * 4 + n] = global_transform_mat[m][n];
                }
            }

            for (size_t k = 0; k < 3; k++)
            {
                posed_joints[i * posed_joints_steps[0] + j * posed_joints_steps[1] + k * posed_joints_steps[2]] = global_transform_mat[k][3];
            }

            auto rel_transform_mat = global_transform_mat;

            for (size_t m = 0; m < 3; m++)
            {
                T acc(0.0);
                for (size_t k = 0; k < 3; k++)
                {
                    acc += global_transform_mat[m][k] * joint[k];
                }
                rel_transform_mat[m][3] -= acc;
            }
            for (size_t m = 0; m < 4; m++)
            {
                for (size_t n = 0; n < 4; n++)
                {
                    rel_transform_mats[i * transform_mats_steps[0] + j * transform_mats_steps[1] + m * 4 + n] = rel_transform_mat[m][n];
                }
            }
        }
    }
    return std::forward_as_tuple(posed_joints, rel_transform_mats);
}

#include <filesystem>

class smpl_model
{
    std::vector<int32_t> parents;
    MatrixXd j_regressor;
    MatrixXf j_regressor_body25;
    std::vector<uint32_t> j_shapedirs_shape;
    std::vector<float> j_shapedirs;
    std::vector<uint32_t> j_posedirs_shape;
    std::vector<float> j_posedirs;
    MatrixXd j_weights;
    MatrixXd j_v_template;
    MatrixXd j_j_regressor;

public:
    smpl_model()
    {
        using path = std::filesystem::path;

        path param_dir("../data/opt/");

        const auto v_template = load_dense_matrix<double>((param_dir / "SMPL_NEUTRAL_v_template.json").generic_string());
        const auto weights = load_dense_matrix<double>((param_dir / "SMPL_NEUTRAL_weights.json").generic_string());
        j_regressor = load_sparse_matrix<double>((param_dir / "SMPL_NEUTRAL_j_regressor.json").generic_string());
        j_regressor_body25 = load_dense_matrix<float>((param_dir / "SMPL_NEUTRAL_j_regressor_body25.json").generic_string());

        const auto [posedirs_data, posedirs_shape] = load_tensor<double>((param_dir / "SMPL_NEUTRAL_posedirs.json").generic_string());
        const auto posedirs_steps = calculate_steps(posedirs_shape);

        const auto [shapedirs_data, shapedirs_shape] = load_tensor<double>((param_dir / "SMPL_NEUTRAL_shapedirs.json").generic_string());
        const auto shapedirs_steps = calculate_steps(shapedirs_shape);

        const auto [kintree_table_data, kintree_table_shape] = load_tensor<uint32_t>((param_dir / "SMPL_NEUTRAL_kintree_table.json").generic_string());

        std::copy_n(kintree_table_data.begin(), kintree_table_shape.back(), std::back_inserter(parents));

        MatrixXd j_regressor_ext(j_regressor.rows() + j_regressor_body25.rows(), j_regressor.cols());
        j_regressor_ext << j_regressor, j_regressor_body25.cast<double>();

        j_weights = j_regressor_ext * weights;
        j_v_template = j_regressor_ext * v_template;
        j_j_regressor = MatrixXd::Identity(j_regressor.rows(), j_regressor_ext.rows());

        j_shapedirs_shape = {static_cast<uint32_t>(j_regressor_ext.rows()), shapedirs_shape[1], shapedirs_shape[2]};
        const auto j_shapedirs_steps = calculate_steps(j_shapedirs_shape);

        j_shapedirs.resize(calculate_size(j_shapedirs_shape));

        for (size_t i = 0; i < j_shapedirs_shape[0]; i++)
        {
            for (size_t j = 0; j < j_shapedirs_shape[1]; j++)
            {
                for (size_t k = 0; k < j_shapedirs_shape[2]; k++)
                {
                    for (size_t v = 0; v < shapedirs_shape[0]; v++)
                    {
                        const auto dst_idx = i * j_shapedirs_steps[0] + j * j_shapedirs_steps[1] + k * j_shapedirs_steps[2];
                        j_shapedirs[dst_idx] += j_regressor_ext(i, v) * shapedirs_data[v * shapedirs_steps[0] + j * shapedirs_steps[1] + k * shapedirs_steps[2]];
                    }
                }
            }
        }

        j_posedirs_shape = {static_cast<uint32_t>(j_regressor_ext.rows()), posedirs_shape[1], posedirs_shape[2]};
        const auto j_posedirs_steps = calculate_steps(j_posedirs_shape);

        j_posedirs.resize(calculate_size(j_posedirs_shape));
        for (size_t i = 0; i < j_posedirs_shape[0]; i++)
        {
            for (size_t j = 0; j < j_posedirs_shape[1]; j++)
            {
                for (size_t k = 0; k < j_posedirs_shape[2]; k++)
                {
                    for (size_t v = 0; v < posedirs_shape[0]; v++)
                    {
                        const auto dst_idx = i * j_posedirs_steps[0] + j * j_posedirs_steps[1] + k * j_posedirs_steps[2];
                        j_posedirs[dst_idx] += j_regressor_ext(i, v) * posedirs_data[v * posedirs_steps[0] + j * posedirs_steps[1] + k * posedirs_steps[2]];
                    }
                }
            }
        }
    }

    template <typename T, typename T1, typename T2>
    std::tuple<std::vector<T>, std::vector<T>> lbs(const std::vector<T1> &shapes_data,
                                                   const std::vector<uint32_t> &shapes_shape,
                                                   const std::vector<T2> &poses_data,
                                                   const std::vector<uint32_t> &poses_shape)
    {
        const auto batch_size = poses_shape[0];

        const auto [blend_shape, blend_shape_shape] = blend_shapes(shapes_data, shapes_shape, j_shapedirs, j_shapedirs_shape);

        std::vector<T> v_shaped(calculate_size(blend_shape_shape));
        std::transform(blend_shape.begin(), blend_shape.end(), j_v_template.reshaped<Eigen::RowMajor>().begin(), v_shaped.begin(), [](const auto a, const auto b)
                       { return T(a) + T(b); });

        const auto [joints, joints_shape] = vertices2joints(j_j_regressor, v_shaped, blend_shape_shape);
        const auto [rot_mats, rot_mats_shape] = batch_rodrigues(poses_data, poses_shape);

        auto pose_feature = rot_mats;
        for (size_t i = 0; i < pose_feature.size(); i += 9)
        {
            pose_feature[i] -= T2(1.0f);
            pose_feature[i + 4] -= T2(1.0f);
            pose_feature[i + 8] -= T2(1.0f);
        }

        const auto [pose_offset, v_posed_shape] = blend_shapes(pose_feature, rot_mats_shape, j_posedirs, j_posedirs_shape);

        std::vector<T> v_posed(calculate_size(v_posed_shape));

        for (size_t b = 0; b < batch_size; b++)
        {
            std::transform(v_shaped.begin(), v_shaped.end(), pose_offset.begin() + (b * pose_offset.size() / batch_size), v_posed.begin() + (b * pose_offset.size() / batch_size), [](const auto a, const auto b)
                           { return a + T(b); });
        }

        const auto [j_transformed, transform_mats] = batch_rigid_transform<T>(rot_mats, rot_mats_shape, joints, joints_shape, parents);

        const auto joint_size = static_cast<uint32_t>(parents.size());
        std::vector<uint32_t> transform_mats_shape = {batch_size, joint_size, 16};
        const auto transform_mats_steps = calculate_steps(transform_mats_shape);

        std::vector<uint32_t> blend_transform_mats_shape = {batch_size, static_cast<uint32_t>(j_weights.rows()), 16};
        const auto blend_transform_mats_steps = calculate_steps(blend_transform_mats_shape);
        std::vector<T> blend_transform_mats(calculate_size(blend_transform_mats_shape));
        for (size_t b = 0; b < batch_size; b++)
        {
            for (size_t i = 0; i < static_cast<uint32_t>(j_weights.rows()); i++)
            {
                for (size_t j = 0; j < 16; j++)
                {
                    for (size_t k = 0; k < joint_size; k++)
                    {
                        const auto dst_idx = b * blend_transform_mats_steps[0] + i * blend_transform_mats_steps[1] + j * blend_transform_mats_steps[2];
                        blend_transform_mats[dst_idx] += j_weights(i, k) * transform_mats[b * transform_mats_steps[0] + k * transform_mats_steps[1] + j * transform_mats_steps[2]];
                    }
                }
            }
        }

        const auto v_posed_steps = calculate_steps(v_posed_shape);
        std::vector<T> verts(calculate_size(v_posed_shape));

        for (size_t b = 0; b < batch_size; b++)
        {
            for (size_t i = 0; i < v_posed_shape[1]; i++)
            {
                std::array<T, 4> v;
                for (size_t j = 0; j < 3; j++)
                {
                    v[j] = v_posed[b * v_posed_steps[0] + i * v_posed_steps[1] + j * v_posed_steps[2]];
                }
                v[3] = T(1.0);

                for (size_t j = 0; j < 3; j++)
                {
                    T acc(0.0);
                    for (size_t k = 0; k < 4; k++)
                    {
                        acc += T(blend_transform_mats[b * blend_transform_mats_steps[0] + i * blend_transform_mats_steps[1] + j * 4 + k]) * v[k];
                    }
                    verts[b * v_posed_steps[0] + i * v_posed_steps[1] + j * v_posed_steps[2]] = acc;
                }
            }
        }

        return std::forward_as_tuple(verts, j_transformed);
    }

    template <typename T, typename T1, typename T2, typename T3>
    std::tuple<std::vector<T>, std::vector<T>> draw(const std::vector<T1> &shapes_data,
                                                    const std::vector<uint32_t> &shapes_shape,
                                                    const std::vector<T2> &poses_data,
                                                    const std::vector<uint32_t> &poses_shape,
                                                    const std::vector<T3> &rh,
                                                    const std::vector<T3> &th)
    {
        const auto [verts, joints] = lbs<T, T1, T2>(shapes_data, shapes_shape, poses_data, poses_shape);

        const auto batch_size = poses_shape[0];
        const auto verts_size = j_regressor.rows() + j_regressor_body25.rows();

        std::vector<T> g_verts(batch_size * j_regressor_body25.rows() * 3);

        for (size_t b = 0; b < poses_shape[0]; b++)
        {
            for (size_t i = 0; i < j_regressor_body25.rows(); i++)
            {
                std::array<T, 3> vert;
                for (size_t j = 0; j < 3; j++)
                {
                    vert[j] = T(verts[b * (verts_size * 3) + (j_regressor.rows() + i) * 3 + j]);
                }

                std::array<T, 3> rot_vec;
                for (size_t j = 0; j < 3; j++)
                {
                    rot_vec[j] = T(rh[b * 3 + j]);
                }

                const auto theta2 = dot_product(rot_vec.data(), rot_vec.data());

                std::array<T, 9> rot_mat;
                rodrigues(rot_vec.data(), rot_mat.data());

                for (size_t j = 0; j < 3; j++)
                {
                    T acc(0.0);
                    for (size_t k = 0; k < 3; k++)
                    {
                        acc += rot_mat[j * 3 + k] * vert[k];
                    }
                    acc += T(th[b * 3 + j]);
                    g_verts[b * (j_regressor_body25.rows() * 3) + i * 3 + j] = acc;
                }
            }
        }

        return std::forward_as_tuple(g_verts, joints);
    }
};

struct limb_length_loss
{
    const std::vector<float> keypoints3d;
    const std::vector<uint32_t> keypoints3d_shape;

    limb_length_loss(const std::vector<float> &keypoints3d, const std::vector<uint32_t> &keypoints3d_shape)
        : keypoints3d(keypoints3d), keypoints3d_shape(keypoints3d_shape) {}

    template <typename T>
    inline T operator()(const T *const verts) const
    {
        std::vector<std::pair<int32_t, int32_t>> kintree = {
            {8, 1},
            {2, 5},
            {2, 3},
            {5, 6},
            {3, 4},
            {6, 7},
            {2, 3},
            {5, 6},
            {3, 4},
            {6, 7},
            {2, 3},
            {5, 6},
            {3, 4},
            {6, 7},
            {1, 0},
            {9, 12},
            {9, 10},
            {10, 11},
            {12, 13},
            {13, 14}};

        std::vector<T> pred(keypoints3d_shape[0] * kintree.size());
        for (size_t b = 0; b < keypoints3d_shape[0]; b++)
        {
            for (size_t i = 0; i < kintree.size(); i++)
            {
                const T v1[] = {
                    verts[b * (keypoints3d_shape[1] * 3) + kintree[i].first * 3],
                    verts[b * (keypoints3d_shape[1] * 3) + kintree[i].first * 3 + 1],
                    verts[b * (keypoints3d_shape[1] * 3) + kintree[i].first * 3 + 2]};

                const T v2[] = {
                    verts[b * (keypoints3d_shape[1] * 3) + kintree[i].second * 3],
                    verts[b * (keypoints3d_shape[1] * 3) + kintree[i].second * 3 + 1],
                    verts[b * (keypoints3d_shape[1] * 3) + kintree[i].second * 3 + 2]};

                const T e[] = {
                    v2[0] - v1[0],
                    v2[1] - v1[1],
                    v2[2] - v1[2]};

                pred[b * kintree.size() + i] = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
            }
        }
        std::vector<T> target(keypoints3d_shape[0] * kintree.size());
        for (size_t b = 0; b < keypoints3d_shape[0]; b++)
        {
            for (size_t i = 0; i < kintree.size(); i++)
            {
                const float v1[] = {
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].first * 4],
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].first * 4 + 1],
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].first * 4 + 2],
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].first * 4 + 3]};

                const float v2[] = {
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].second * 4],
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].second * 4 + 1],
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].second * 4 + 2],
                    keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].second * 4 + 3]};

                const T e[] = {
                    T(v2[0] - v1[0]),
                    T(v2[1] - v1[1]),
                    T(v2[2] - v1[2]),
                    T(v2[3] - v1[3])};

                target[b * kintree.size() + i] = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);
            }
        }

        std::vector<float> target_conf(keypoints3d_shape[0] * kintree.size());
        for (size_t b = 0; b < keypoints3d_shape[0]; b++)
        {
            for (size_t i = 0; i < kintree.size(); i++)
            {
                const auto conf1 = keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].first * 4 + 3];
                const auto conf2 = keypoints3d[b * (keypoints3d_shape[1] * 4) + kintree[i].second * 4 + 3];
                target_conf[b * kintree.size() + i] = std::min(conf1, conf2);
            }
        }
        T num(0.0);
        T denom(0.0);
        for (size_t b = 0; b < keypoints3d_shape[0]; b++)
        {
            for (size_t i = 0; i < kintree.size(); i++)
            {
                const auto d = pred[b * kintree.size() + i] - target[b * kintree.size() + i];
                num += d * d * T(target_conf[b * kintree.size() + i]);
                denom += target_conf[b * kintree.size() + i];
            }
        }
        const auto loss = num / (T(1e-5) + denom);
        return loss;
    }
};

struct regression_loss
{
    const std::vector<uint32_t> values_shape;

public:
    regression_loss(const std::vector<uint32_t> &values_shape)
        : values_shape(values_shape)
    {
    }

    template <typename T>
    inline T operator()(const T *const values) const
    {
        T sq_sum(0.0);
        for (size_t b = 0; b < values_shape[0]; b++)
        {
            for (size_t i = 0; i < values_shape[1]; i++)
            {
                sq_sum += values[i] * values[i];
            }
        }
        const auto loss = sq_sum / T(values_shape[0]);
        return loss;
    }
};

struct keypoints3d_loss
{
    std::vector<int32_t> indices;
    const std::vector<float> keypoints3d;
    const std::vector<uint32_t> keypoints3d_shape;

    keypoints3d_loss(const std::vector<float> &keypoints3d, const std::vector<uint32_t> &keypoints3d_shape, const std::vector<int32_t> &indices)
        : keypoints3d(keypoints3d), keypoints3d_shape(keypoints3d_shape), indices(indices)
    {
        if (indices.empty())
        {
            this->indices.resize(keypoints3d_shape[1]);
            std::iota(this->indices.begin(), this->indices.end(), 0);
        }
    }

    template <typename T>
    inline T operator()(const T *const verts) const
    {
        T num(0.0);
        T denom(0.0);
        for (size_t b = 0; b < keypoints3d_shape[0]; b++)
        {
            for (size_t i = 0; i < indices.size(); i++)
            {
                const T v1[] = {
                    verts[b * (keypoints3d_shape[1] * 3) + indices[i] * 3],
                    verts[b * (keypoints3d_shape[1] * 3) + indices[i] * 3 + 1],
                    verts[b * (keypoints3d_shape[1] * 3) + indices[i] * 3 + 2]};

                const T v2[] = {
                    T(keypoints3d[b * (keypoints3d_shape[1] * 4) + indices[i] * 4]),
                    T(keypoints3d[b * (keypoints3d_shape[1] * 4) + indices[i] * 4 + 1]),
                    T(keypoints3d[b * (keypoints3d_shape[1] * 4) + indices[i] * 4 + 2])};

                const auto conf = T(keypoints3d[b * (keypoints3d_shape[1] * 4) + indices[i] * 4 + 3]);

                const T e[] = {
                    v2[0] - v1[0],
                    v2[1] - v1[1],
                    v2[2] - v1[2]};

                num += (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) * conf;
                denom += conf;
            }
        }
        const auto loss = num / (T(1e-5) + denom);
        return loss;
    }
};

enum class smooth_type
{
    LINEAR
};

struct smooth_loss
{
    const std::vector<uint32_t> values_shape;
    const std::vector<float> window_height;
    const bool order2;

public:
    smooth_loss(const std::vector<uint32_t> &values_shape, const std::vector<float> &window_height, bool order2 = true)
        : values_shape(values_shape), window_height(window_height), order2(order2)
    {
    }

    template <typename T>
    inline T operator()(const T *const values) const
    {
        T loss(0.0);
        for (size_t k = 0; k < window_height.size(); k++)
        {
            T sq_sum(0.0);
            if (order2)
            {
                assert(values_shape[0] >= (k + 2));
                for (size_t b = 0; b < values_shape[0] - (k + 2); b++)
                {
                    for (size_t i = 0; i < values_shape[1]; i++)
                    {
                        for (size_t j = 0; j < values_shape[2]; j++)
                        {
                            const auto d1 = values[(b + k + 1) * (values_shape[1] * values_shape[2]) + i * values_shape[2] + j] - values[b * (values_shape[1] * values_shape[2]) + i * values_shape[2] + j];
                            const auto d2 = values[(b + k + 2) * (values_shape[1] * values_shape[2]) + i * values_shape[2] + j] - values[(b + 1) * (values_shape[1] * values_shape[2]) + i * values_shape[2] + j];
                            const auto d = d2 - d1;
                            sq_sum += d * d;
                        }
                    }
                }
                sq_sum /= T((values_shape[0] - (k + 2)) * values_shape[1]);
            }
            else
            {
                assert(values_shape[0] >= (k + 1));
                for (size_t b = 0; b < values_shape[0] - (k + 1); b++)
                {
                    for (size_t i = 0; i < values_shape[1]; i++)
                    {
                        for (size_t j = 0; j < values_shape[2]; j++)
                        {
                            const auto d = values[(b + k + 1) * (values_shape[1] * values_shape[2]) + i * values_shape[2] + j] - values[b * (values_shape[1] * values_shape[2]) + i * values_shape[2] + j];
                            sq_sum += d * d;
                        }
                    }
                }
                sq_sum /= T((values_shape[0] - (k + 1)) * values_shape[1]);
            }
            loss += sq_sum * T(window_height[k]);
        }
        return loss;
    }
};

struct prior_loss
{
    const std::vector<uint32_t> values_shape;
    const size_t num_gaussians;
    const double epsilon;
    const bool use_merged;

    std::vector<double> means;
    std::vector<uint32_t> means_shape;
    std::vector<double> covs;
    std::vector<uint32_t> covs_shape;
    std::vector<double> weights;
    std::vector<uint32_t> weights_shape;

    std::vector<double> precisions;
    std::vector<double> nll_weights;

public:
    prior_loss(const std::vector<uint32_t> &values_shape, size_t num_gaussians = 8, double epsilon = 1e-16, bool use_merged = true)
        : values_shape(values_shape), num_gaussians(num_gaussians), epsilon(epsilon), use_merged(use_merged)
    {
        using path = std::filesystem::path;

        path param_dir("../data/opt/");

        std::tie(means, means_shape) = load_tensor<double>((param_dir / "gmm_means.json").generic_string());
        std::tie(covs, covs_shape) = load_tensor<double>((param_dir / "gmm_covars.json").generic_string());
        std::tie(weights, weights_shape) = load_tensor<double>((param_dir / "gmm_weights.json").generic_string());

        std::vector<double> precisions(calculate_size(covs_shape));
        std::vector<double> sqrdets;
        for (size_t m = 0; m < covs_shape[0]; m++)
        {
            const MatrixXd cov = Eigen::Map<MatrixXd>(covs.data() + (covs_shape[1] * covs_shape[2] * m), covs_shape[1], covs_shape[2]);
            const auto sqrdet = std::sqrt(cov.determinant());

            Eigen::Map<MatrixXd>(precisions.data() + (covs_shape[2] * covs_shape[1] * m), covs_shape[2], covs_shape[1]) = cov.inverse();

            sqrdets.push_back(sqrdet);
        }

        constexpr auto PI = 3.141592653589793;
        const auto c = std::pow((2 * PI), 69.0 / 2);
        const auto min_sqrdets = *std::min_element(sqrdets.begin(), sqrdets.end());

        std::vector<double> nll_weights(calculate_size(weights_shape));
        for (size_t m = 0; m < weights_shape[0]; m++)
        {
            nll_weights[m] = -std::log(weights[m] / (c * (sqrdets[m] / min_sqrdets)));
        }

        this->precisions = precisions;
        this->nll_weights = nll_weights;
    }

    template <typename T>
    inline T operator()(const T *const values) const
    {
        T loss(0.0);
        for (size_t b = 0; b < values_shape[0]; b++)
        {
            std::vector<T> d(means_shape[0] * values_shape[1]);
            for (size_t m = 0; m < means_shape[0]; m++)
            {
                assert(means_shape[1] == values_shape[1]);
                for (size_t j = 0; j < values_shape[1]; j++)
                {
                    d[m * values_shape[1] + j] = values[b * values_shape[1] + j] - means[m * means_shape[1] + j];
                }
            }

            std::vector<T> prec_d(means_shape[0] * values_shape[1]);
            for (size_t m = 0; m < means_shape[0]; m++)
            {
                assert(values_shape[1] == means_shape[1]);
                assert(values_shape[1] == covs_shape[1]);
                assert(values_shape[1] == covs_shape[2]);
                for (size_t i = 0; i < covs_shape[1]; i++)
                {
                    T acc(0.0);
                    for (size_t j = 0; j < covs_shape[2]; j++)
                    {
                        acc += precisions[m * (covs_shape[1] * covs_shape[2]) + i * covs_shape[2] + j] * d[m * values_shape[1] + j];
                    }
                    prec_d[m * values_shape[1] + i] = acc;
                }
            }

            std::vector<T> prec_dd(means_shape[0]);
            for (size_t m = 0; m < means_shape[0]; m++)
            {
                assert(values_shape[1] == means_shape[1]);
                T acc(0.0);
                for (size_t j = 0; j < values_shape[1]; j++)
                {
                    acc += prec_d[m * values_shape[1] + j] * d[m * values_shape[1] + j];
                }
                prec_dd[m] = acc;
            }

            T min_likelihood(std::numeric_limits<double>::max());
            for (size_t m = 0; m < means_shape[0]; m++)
            {
                const T loglikelihood = T(0.5) * prec_dd[m] + T(nll_weights[m]);
                min_likelihood = fmin(min_likelihood, loglikelihood);
            }

            loss += min_likelihood;
        }
        loss /= T(values_shape[0]);
        return loss;
    }
};

struct fit_shape_obj_func
{
    const std::vector<uint32_t> shapes_shape;
    const std::vector<float> poses_data;
    const std::vector<uint32_t> poses_shape;
    const std::vector<float> rh_data;
    const std::vector<float> th_data;

    smpl_model *model;

    const limb_length_loss limb_length;
    const regression_loss reg;

    fit_shape_obj_func(smpl_model *model,
                       const std::vector<float> &keypoints3d, const std::vector<uint32_t> &keypoints3d_shape,
                       const std::vector<uint32_t> &shapes_shape,
                       const std::vector<float> &poses_data, const std::vector<uint32_t> &poses_shape, const std::vector<float> &rh_data, const std::vector<float> &th_data)
        : shapes_shape(shapes_shape), poses_data(poses_data), poses_shape(poses_shape), rh_data(rh_data), th_data(th_data), model(model),
          limb_length(keypoints3d, keypoints3d_shape), reg(shapes_shape) {}

    template <typename T>
    inline T operator()(const T *const shapes) const
    {
        std::vector<T> shapes_data(shapes, shapes + calculate_size(shapes_shape));
        const auto [verts, joints] = model->draw<T, T, float, float>(shapes_data, shapes_shape, poses_data, poses_shape, rh_data, th_data);
        const auto limb_length_loss = limb_length(verts.data());
        const auto limb_length_loss_weight = 100.0;
        const auto reg_loss = reg(shapes);
        const auto reg_loss_weight = 0.1;
        const auto loss = limb_length_loss * limb_length_loss_weight + reg_loss * reg_loss_weight;
        return loss;
    }
};

struct init_rt_obj_func
{
    const std::vector<float> shapes_data;
    const std::vector<uint32_t> shapes_shape;
    const std::vector<float> poses_data;
    const std::vector<uint32_t> poses_shape;

    smpl_model *model;

    const keypoints3d_loss keypoints3d;
    const smooth_loss smooth_keypoints;
    const smooth_loss smooth_th;

    init_rt_obj_func(smpl_model *model,
                     const std::vector<float> &keypoints3d, const std::vector<uint32_t> &keypoints3d_shape,
                     const std::vector<float> &shapes_data, const std::vector<uint32_t> &shapes_shape,
                     const std::vector<float> &poses_data, const std::vector<uint32_t> &poses_shape, const std::vector<float> &rh_data, const std::vector<float> &th_data)
        : shapes_data(shapes_data), shapes_shape(shapes_shape), poses_data(poses_data), poses_shape(poses_shape), model(model),
          keypoints3d(keypoints3d, keypoints3d_shape, {2, 5, 9, 12}), smooth_keypoints({keypoints3d_shape[0], keypoints3d_shape[1], 3}, {0.5, 0.3, 0.1, 0.1}), smooth_th({keypoints3d_shape[0], 1, 3}, {0.5, 0.3, 0.1, 0.1}) {}

    template <typename T>
    inline T operator()(const T *const rh, const T *const th) const
    {
        std::vector<T> rh_data(rh, rh + (poses_shape[0] * 3));
        std::vector<T> th_data(th, th + (poses_shape[0] * 3));

        const auto [verts, joints] = model->draw<T, float, float, T>(shapes_data, shapes_shape, poses_data, poses_shape, rh_data, th_data);

        const auto keypoints3d_loss = keypoints3d(verts.data());
        const auto keypoints3d_loss_weight = 100.0;
        const auto smooth_keypoints_loss = smooth_keypoints(verts.data());
        const auto smooth_th_loss = smooth_th(th_data.data());
        const auto smooth_loss = smooth_keypoints_loss * 10.0 + smooth_th_loss * 100.0;
        const auto smooth_loss_weight = 1.0;
        const auto loss = keypoints3d_loss * keypoints3d_loss_weight + smooth_loss * smooth_loss_weight;
        return loss;
    }
};

struct refine_pose_obj_func
{
    const std::vector<float> shapes_data;
    const std::vector<uint32_t> shapes_shape;
    const std::vector<uint32_t> poses_shape;

    smpl_model *model;

    const keypoints3d_loss keypoints3d;
    const smooth_loss smooth_keypoints;
    const smooth_loss smooth_th;
    const smooth_loss smooth_poses;
    const prior_loss prior;

    refine_pose_obj_func(smpl_model *model,
                         const std::vector<float> &keypoints3d, const std::vector<uint32_t> &keypoints3d_shape,
                         const std::vector<float> &shapes_data, const std::vector<uint32_t> &shapes_shape,
                         const std::vector<uint32_t> &poses_shape, const std::vector<float> &rh_data, const std::vector<float> &th_data)
        : shapes_data(shapes_data), shapes_shape(shapes_shape), poses_shape(poses_shape), model(model),
          keypoints3d(keypoints3d, keypoints3d_shape, {}),
          smooth_keypoints({keypoints3d_shape[0], keypoints3d_shape[1], 3}, {0.5, 0.3, 0.1, 0.1}),
          smooth_th({keypoints3d_shape[0], 1, 3}, {0.5, 0.3, 0.1, 0.1}),
          smooth_poses({poses_shape[0], 1, poses_shape[1]}, {0.5, 0.3, 0.1, 0.1}),
          prior(poses_shape) {}

    template <typename T>
    inline T operator()(const T *const poses, const T *const rh, const T *const th) const
    {
        std::vector<T> poses_data(poses, poses + calculate_size(poses_shape));
        std::vector<T> rh_data(rh, rh + (poses_shape[0] * 3));
        std::vector<T> th_data(th, th + (poses_shape[0] * 3));

        const auto [verts, joints] = model->draw<T, float, T, T>(shapes_data, shapes_shape, poses_data, poses_shape, rh_data, th_data);

        const auto keypoints3d_loss = keypoints3d(verts.data());
        const auto keypoints3d_loss_weight = 1000.0;
        const auto smooth_poses_loss = smooth_poses(poses_data.data());
        const auto smooth_keypoints_loss = smooth_keypoints(verts.data());
        const auto smooth_th_loss = smooth_th(th_data.data());
        const auto smooth_loss = smooth_poses_loss * 100.0 + smooth_keypoints_loss * 10.0 + smooth_th_loss * 10.0;
        const auto smooth_loss_weight = 1.0;
        const auto prior_loss_weight = 0.1;
        const auto prior_loss = prior(poses_data.data());
        const auto loss = keypoints3d_loss * keypoints3d_loss_weight + smooth_loss * smooth_loss_weight + prior_loss * prior_loss_weight;
        return loss;
    }
};

#include "nonlinear_solver.hpp"

static double cubic_interpolate(const double x1, const double f1, const double g1, const double x2, const double f2, const double g2, double xmin_bound, double xmax_bound)
{
    // Compute d1
    const auto d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2);

    // Check that the square root will be real in the
    // expression for d2
    const auto d2_square = d1 * d1 - g1 * g2;
    if (d2_square >= 0)
    {
        // Compute d2
        const auto d2 = std::sqrt(d2_square);

        // Evaluate the new interpolation point
        const auto min_pos = (x1 <= x2) ? (x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))) : (x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2)));

        // If the new point is outside the interval, return
        // the clamped point
        return std::min(std::max(min_pos, xmin_bound), xmax_bound);
    }
    else
    {
        return (xmin_bound + xmax_bound) / 2.0;
    }
}

static double cubic_interpolate(const double x1, const double f1, const double g1, const double x2, const double f2, const double g2)
{
    return cubic_interpolate(x1, f1, g1, x2, f2, g2, std::min(x1, x2), std::max(x1, x2));
}

template <typename Func>
static double zoom(const Func *obj_func,
                   double f_low, double gtd_low, double alpha_low,
                   double f_high, double gtd_high, double alpha_high,
                   const Eigen::VectorXd &params, const double f, const Eigen::VectorXd &g, const Eigen::VectorXd &d, std::size_t max_iteration = 25)
{
    double c1 = 1e-3;
    double c2 = 0.9;

    const auto tolerance_change = 1.0e-9;
    const auto d_norm = d.array().abs().maxCoeff();
    const auto gtd = g.dot(d);

    bool insuf_progress = false;

    auto alpha = 0.0;
    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        // Pick an alpha value using cubic interpolation
        alpha = cubic_interpolate(
            alpha_low,
            f_low,
            gtd_low,
            alpha_high,
            f_high,
            gtd_high);

        // Pick an alpha value by bisecting the interval
        const auto eps = 0.1 * (alpha_high - alpha_low);
        if (std::min(alpha_high - alpha, alpha - alpha_low) < eps)
        {
            if (insuf_progress || alpha >= alpha_high || alpha <= alpha_low)
            {
                if (std::abs(alpha_high - alpha) < std::abs(alpha - alpha_low))
                {
                    alpha = alpha_high - eps;
                }
                else
                {
                    alpha = alpha_low + eps;
                }
                insuf_progress = false;
            }
            else
            {
                insuf_progress = true;
            }
        }
        else
        {
            insuf_progress = false;
        }

        // Evaluate the merit function
        Eigen::VectorXd step_params = params + d * alpha;
        const auto new_f = obj_func->eval(step_params.data(), params.size());

        // Check if the sufficient decrease condition is violated
        if (new_f > (f + c1 * alpha * gtd) || new_f >= f_low)
        {
            alpha_high = alpha;
            f_high = new_f;

            Eigen::VectorXd new_g(params.size());
            obj_func->grad(step_params.data(), params.size(), new_g.data());
            gtd_high = new_g.dot(d);
        }
        else
        {
            // Evaluate the gradient of the function and the
            // derivative of the merit function
            Eigen::VectorXd new_g(params.size());
            obj_func->grad(step_params.data(), params.size(), new_g.data());
            const auto new_gtd = new_g.dot(d);

            // Return alpha, the strong Wolfe conditions are
            // satisfied
            if (std::abs(new_gtd) <= c2 * std::abs(gtd))
            {
                return alpha;
            }

            // Make sure that we have the intervals right
            if (new_gtd * (alpha_high - alpha_low) >= 0.0)
            {
                alpha_high = alpha_low;
                f_high = f_low;
                gtd_high = gtd_low;
            }

            // Swap alpha low/alpha
            alpha_low = alpha;
            gtd_low = new_gtd;
            f_low = new_f;

            if (std::abs(alpha_high - alpha_low) * d_norm < tolerance_change)
            {
                break;
            }
        }
    }

    return alpha;
}

template <typename Func>
static double strong_wolfe(const Func *obj_func,
                           const Eigen::VectorXd &params, const Eigen::VectorXd &g, const Eigen::VectorXd &d, double alpha = 1.0, std::size_t max_iteration = 25)
{
    double c1 = 1e-3;
    double c2 = 0.9;

    // Compute the function and the gradient at alpha = 0
    const auto f = obj_func->eval(params.data(), params.size());
    const auto gtd = g.dot(d);

    auto prev_alpha = 0.0;
    auto prev_f = f;
    auto prev_g = g;
    auto prev_gtd = gtd;

    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        // Evaluate the merit function
        Eigen::VectorXd step_params = params + d * alpha;
        const auto new_f = obj_func->eval(step_params.data(), params.size());

        // Evaluate the gradient at the new point
        Eigen::VectorXd new_g(params.size());
        obj_func->grad(step_params.data(), params.size(), new_g.data());
        const auto new_gtd = new_g.dot(d);

        // Check if either the sufficient decrease condition is
        // violated or the objective increased
        if (new_f > f + c1 * alpha * gtd || (iter > 0 && new_f >= f))
        {
            // Zoom and return
            return zoom(obj_func, prev_f, prev_gtd, prev_alpha, new_f, new_gtd, alpha, params, f, g, d, max_iteration);
        }

        // Check if the strong Wolfe conditions are satisfied
        if (std::abs(new_gtd) <= c2 * std::abs(gtd))
        {
            return alpha;
        }

        // If the line search is vioalted
        if (new_gtd >= 0.0)
        {
            return zoom(obj_func, prev_f, prev_gtd, prev_alpha, new_f, new_gtd, alpha, params, f, g, d, max_iteration);
        }

        // Pick a new value for alpha
        // new_alpha = std::min(2.0 * alpha, 100.0);

        const auto min_step = alpha + 0.01 * (alpha - prev_alpha);
        const auto max_step = alpha * 10;
        const auto new_alpha = cubic_interpolate(
            prev_alpha,
            prev_f,
            prev_gtd,
            alpha,
            new_f,
            new_gtd,
            min_step,
            max_step);

        // Record the old values of alpha and f
        prev_alpha = alpha;
        prev_f = new_f;
        prev_g = new_g;
        prev_gtd = new_gtd;
        alpha = new_alpha;
    }

    std::cout << "aaa" << std::endl;
    return alpha;
}

class lbgfs_optimizer
{
    int n_iter = 0;
    float lr = 1.0f;
    int max_iter = 20;
    float max_eval = max_iter * 1.25f;
    // float tolerance_grad = 1e-5f;
    // float tolerance_change = 1e-9f;
    float tolerance_grad = 1e-7f;
    float tolerance_change = 1e-7f;
    int history_size = 100;
    float alpha = 1.0f;

    Eigen::VectorXd params_v;
    Eigen::VectorXd prev_grad_v;
    Eigen::VectorXd d_v;

    std::vector<Eigen::VectorXd> s;
    std::vector<Eigen::VectorXd> y;

public:
    lbgfs_optimizer()
        : s(history_size), y(history_size)
    {
    }

    void get_params(double *params, size_t num_params) const
    {
        for (std::size_t i = 0; i < num_params; i++)
        {
            params[i] = params_v(i);
        }
    }

    void set_params(const double *params, size_t num_params)
    {
        params_v.resize(num_params);
        for (std::size_t i = 0; i < num_params; i++)
        {
            params_v(i) = params[i];
        }
    }

    template <typename Func>
    double step(const Func *func)
    {
        const auto num_params = params_v.size();

        auto init_loss = func->eval(params_v.data(), num_params);
        auto loss = init_loss;

        Eigen::VectorXd grad_v(num_params);
        func->grad(params_v.data(), num_params, grad_v.data());

        const auto max_m = history_size;

        for (int iter = 0; iter < max_iter; iter++)
        {
            if (n_iter == 0)
            {
                d_v = -grad_v;
            }
            else
            {
                Eigen::VectorXd q = -grad_v;

                const auto m = std::min(n_iter, max_m);
                std::vector<double> a(max_m);

                for (std::size_t j = 0; j < m; j++)
                {
                    const auto i = n_iter - 1 - j;
                    a[i % max_m] = s[i % max_m].dot(q) / y[i % max_m].dot(s[i % max_m]);
                    q = q - a[i % max_m] * y[i % max_m];
                }

                const auto i = n_iter - 1;
                q = s[i % max_m].dot(y[i % max_m]) / y[i % max_m].dot(y[i % max_m]) * q;

                for (std::size_t j = 0; j < m; j++)
                {
                    const auto i = n_iter - m + j;

                    const auto b = y[i % max_m].dot(q) / y[i % max_m].dot(s[i % max_m]);
                    q = q + (a[i % max_m] - b) * s[i % max_m];
                }

                d_v = q;
            }

            prev_grad_v = grad_v;

            if (n_iter == 0)
            {
                alpha = std::min(1.0f, 1.0f / static_cast<float>(grad_v.array().abs().sum())) * lr;
            }
            else
            {
                alpha = lr;
            }

            if (grad_v.dot(d_v) > -tolerance_change)
            {
                break;
            }

            alpha = strong_wolfe(func, params_v, grad_v, d_v, alpha);

            std::cout << "loss = " << loss << ", alpha = " << alpha << std::endl;

            Eigen::VectorXd next_params_v = params_v + alpha * d_v;

            Eigen::VectorXd next_grad_v(num_params);
            func->grad(next_params_v.data(), num_params, next_grad_v.data());

            const auto next_loss = func->eval(next_params_v.data(), num_params);

            Eigen::VectorXd s_v = alpha * d_v;
            Eigen::VectorXd y_v = next_grad_v - grad_v;

            s[n_iter % max_m] = s_v;
            y[n_iter % max_m] = y_v;

            const auto prev_loss = loss;

            params_v = next_params_v;
            grad_v = next_grad_v;
            loss = next_loss;

            n_iter++;

            if (n_iter == max_iter)
            {
                break;
            }

            if (grad_v.array().abs().maxCoeff() <= tolerance_grad)
            {
                break;
            }

            if ((d_v.array() * alpha).maxCoeff() <= tolerance_change)
            {
                break;
            }

            if (std::abs(loss - prev_loss) < tolerance_change)
            {
                break;
            }
        }

        std::cout << "loss = " << loss << ", alpha = " << alpha << std::endl;

        return std::abs(loss - init_loss);
    }
};

int main()
{

    const auto [keypoints3d_data, keypoints3d_shape] = load_tensor<float>("../data/opt/observations_keypoints3d.json");
    auto [poses_data, poses_shape] = load_tensor<float>("../data/opt/params_poses.json");
    auto [shapes_data, shapes_shape] = load_tensor<float>("../data/opt/params_shapes.json");
    auto [rh_data, rh_shape] = load_tensor<float>("../data/opt/params_Rh.json");
    auto [th_data, th_shape] = load_tensor<float>("../data/opt/params_Th.json");

    smpl_model model;

    {
        fit_shape_obj_func loss_func(&model, keypoints3d_data, keypoints3d_shape, shapes_shape, poses_data, poses_shape, rh_data, th_data);

        std::vector<double> shapes_double(shapes_data.begin(), shapes_data.end());
        std::shared_ptr<optimization::parameter_block> shapes_params = std::make_shared<optimization::parameter_block>(shapes_double.size(), shapes_double);
        shapes_params->offset = 0;

        const auto residual = optimization::make_auto_diff_function_term(loss_func, shapes_params);

        lbgfs_optimizer optimizer;
        optimizer.set_params(shapes_double.data(), shapes_double.size());

        for (int i = 0; i < 1000; i++)
        {
            const auto loss_change = optimizer.step(residual.get());
            if (loss_change < 1.0e-7)
            {
                break;
            }
        }

        optimizer.get_params(shapes_double.data(), shapes_double.size());

        std::copy(shapes_double.begin(), shapes_double.end(), shapes_data.begin());
    }

    {
        init_rt_obj_func loss_func(&model, keypoints3d_data, keypoints3d_shape, shapes_data, shapes_shape, poses_data, poses_shape, rh_data, th_data);

        std::vector<double> rh_double(rh_data.begin(), rh_data.end());
        std::vector<double> th_double(th_data.begin(), th_data.end());
        std::shared_ptr<optimization::parameter_block> rh_params = std::make_shared<optimization::parameter_block>(rh_data.size(), rh_double);
        std::shared_ptr<optimization::parameter_block> th_params = std::make_shared<optimization::parameter_block>(th_data.size(), th_double);
        th_params->offset = 0;
        rh_params->offset = th_data.size();

        const auto residual = optimization::make_auto_diff_function_term(loss_func, rh_params, th_params);

        std::vector<double> params;
        std::copy(th_double.begin(), th_double.end(), std::back_inserter(params));
        std::copy(rh_double.begin(), rh_double.end(), std::back_inserter(params));

        lbgfs_optimizer optimizer;
        optimizer.set_params(params.data(), params.size());

        for (int i = 0; i < 1000; i++)
        {
            const auto loss_change = optimizer.step(residual.get());
            if (loss_change < 1.0e-7)
            {
                break;
            }
        }

        optimizer.get_params(params.data(), params.size());

        std::copy_n(params.begin(), th_data.size(), th_data.begin());
        std::copy_n(params.begin() + th_data.size(), rh_data.size(), rh_data.begin());
    }

    {
        refine_pose_obj_func loss_func(&model, keypoints3d_data, keypoints3d_shape, shapes_data, shapes_shape, poses_shape, rh_data, th_data);

        std::vector<double> poses_double(poses_data.begin(), poses_data.end());
        std::vector<double> rh_double(rh_data.begin(), rh_data.end());
        std::vector<double> th_double(th_data.begin(), th_data.end());
        std::shared_ptr<optimization::parameter_block> poses_params = std::make_shared<optimization::parameter_block>(poses_data.size(), poses_double);
        std::shared_ptr<optimization::parameter_block> rh_params = std::make_shared<optimization::parameter_block>(rh_data.size(), rh_double);
        std::shared_ptr<optimization::parameter_block> th_params = std::make_shared<optimization::parameter_block>(th_data.size(), th_double);
        poses_params->offset = 0;
        rh_params->offset = poses_data.size();
        th_params->offset = poses_data.size() + rh_data.size();

        const auto residual = optimization::make_auto_diff_function_term(loss_func, poses_params, rh_params, th_params);

        std::vector<double> params;
        std::copy(poses_double.begin(), poses_double.end(), std::back_inserter(params));
        std::copy(rh_double.begin(), rh_double.end(), std::back_inserter(params));
        std::copy(th_double.begin(), th_double.end(), std::back_inserter(params));

        lbgfs_optimizer optimizer;
        optimizer.set_params(params.data(), params.size());

        for (int i = 0; i < 1000; i++)
        {
            const auto loss_change = optimizer.step(residual.get());
            if (loss_change < 1.0e-7)
            {
                break;
            }
        }

        optimizer.get_params(params.data(), params.size());

        std::copy_n(params.begin(), poses_data.size(), poses_data.begin());
        std::copy_n(params.begin() + poses_data.size(), rh_data.size(), rh_data.begin());
        std::copy_n(params.begin() + poses_data.size() + rh_data.size(), th_data.size(), th_data.begin());
    }
    return 0;
}