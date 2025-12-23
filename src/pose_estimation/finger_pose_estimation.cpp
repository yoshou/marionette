#include "finger_pose_estimation.hpp"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>
#include <iostream>
#include <algorithm>

struct local_angle_error
{
    local_angle_error()
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        const T* const rotation,
        const T* const child_rotation,
        T* residuals) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q3(child_rotation);

        const Eigen::Quaternion<T> q4 = Eigen::Quaternion<T>(q1).slerp(T(0.5), q3);
        const Eigen::Vector4<T> diff = Eigen::Vector4<T>(q4.x(), q4.y(), q4.z(), q4.w()) - q2;

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();
        residuals[3] = diff.w();

        return true;
    }

    static ceres::CostFunction* create()
    {
        return (new ceres::AutoDiffCostFunction<local_angle_error, 4, 4, 4, 4>(
            new local_angle_error()));
    }
};

struct local_angle_error2
{
    const glm::dquat parent_rotation;

    local_angle_error2(const glm::dquat& parent_rotation)
        : parent_rotation(parent_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const rotation,
        const T* const child_rotation,
        T* residuals) const
    {
        const T parent_rotation[] = { T(this->parent_rotation.x), T(this->parent_rotation.y), T(this->parent_rotation.z), T(this->parent_rotation.w)};

        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q3(child_rotation);

        const Eigen::Quaternion<T> q4 = Eigen::Quaternion<T>(q1).slerp(T(0.5), q3);
        const Eigen::Vector4<T> diff = Eigen::Vector4<T>(q4.x(), q4.y(), q4.z(), q4.w()) - q2;

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();
        residuals[3] = diff.w();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& parent_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_angle_error2, 4, 4, 4>(
            new local_angle_error2(parent_rotation)));
    }
};

struct local_angle_error3
{
    const glm::dquat child_rotation;

    local_angle_error3(const glm::dquat& child_rotation)
        : child_rotation(child_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        const T* const rotation,
        T* residuals) const
    {
        const T child_rotation[] = { T(this->child_rotation.x), T(this->child_rotation.y), T(this->child_rotation.z), T(this->child_rotation.w)};

        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Vector4<T>> q2(rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q3(child_rotation);

        const Eigen::Quaternion<T> q4 = Eigen::Quaternion<T>(q1).slerp(T(0.5), q3);
        const Eigen::Vector4<T> diff = Eigen::Vector4<T>(q4.x(), q4.y(), q4.z(), q4.w()) - q2;

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();
        residuals[3] = diff.w();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& child_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_angle_error3, 4, 4, 4>(
            new local_angle_error3(child_rotation)));
    }
};

struct articulation_error
{
    glm::vec3 local_child_position;

    articulation_error(const glm::vec3& local_child_position)
        : local_child_position(local_child_position)
    {}

    template <typename T>
    bool operator()(
        const T* const parent_translation,
        const T* const parent_rotation,
        const T* const child_translation,
        T* residuals) const
    {
        T point[3] = {
            T(static_cast<double>(this->local_child_position.x)),
            T(static_cast<double>(this->local_child_position.y)),
            T(static_cast<double>(this->local_child_position.z)) };

        T p[3];
        ceres::QuaternionRotatePoint(parent_rotation, point, p);
        p[0] += parent_translation[0];
        p[1] += parent_translation[1];
        p[2] += parent_translation[2];

        residuals[0] = p[0] - child_translation[0];
        residuals[1] = p[1] - child_translation[1];
        residuals[2] = p[2] - child_translation[2];

        return true;
    }

    static ceres::CostFunction* create(const glm::vec3& local_child_positionn)
    {
        return (new ceres::AutoDiffCostFunction<articulation_error, 3, 3, 4, 3>(
            new articulation_error(local_child_positionn)));
    }
};

struct articulation_error2
{
    glm::vec3 local_child_position;

    articulation_error2(const glm::vec3& local_child_position)
        : local_child_position(local_child_position)
    {}

    template <typename T>
    bool operator()(
        const T* const child_translation,
        T* residuals) const
    {
        T p[3] = {
            T(static_cast<double>(this->local_child_position.x)),
            T(static_cast<double>(this->local_child_position.y)),
            T(static_cast<double>(this->local_child_position.z)) };

        residuals[0] = p[0] - child_translation[0];
        residuals[1] = p[1] - child_translation[1];
        residuals[2] = p[2] - child_translation[2];

        return true;
    }

    static ceres::CostFunction* create(const glm::vec3& local_child_positionn)
    {
        return (new ceres::AutoDiffCostFunction<articulation_error2, 3, 3>(
            new articulation_error2(local_child_positionn)));
    }
};

struct local_1dof_constraint_error
{
    local_1dof_constraint_error()
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        const T* const child_rotation,
        T* residuals) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q2(child_rotation);

        const Eigen::Vector3<T> axis(T(1.0), T(0.0), T(0.0));

        const Eigen::Vector3<T> v1 = q1 * axis;
        const Eigen::Vector3<T> v2 = q2 * axis;

        const Eigen::Vector3<T> diff = (v1 - v2) * T(100.0);

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();

        return true;
    }

    static ceres::CostFunction* create()
    {
        return (new ceres::AutoDiffCostFunction<local_1dof_constraint_error, 3, 4, 4>(
            new local_1dof_constraint_error()));
    }
};

struct local_1dof_constraint_error2
{
    const glm::dquat parent_rotation;

    local_1dof_constraint_error2(const glm::dquat& parent_rotation)
        : parent_rotation(parent_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const child_rotation,
        T* residuals) const
    {
        const T parent_rotation[] = { T(this->parent_rotation.x), T(this->parent_rotation.y), T(this->parent_rotation.z), T(this->parent_rotation.w) };

        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q2(child_rotation);

        const Eigen::Vector3<T> axis(T(1.0), T(0.0), T(0.0));

        const Eigen::Vector3<T> v1 = q1 * axis;
        const Eigen::Vector3<T> v2 = q2 * axis;

        const Eigen::Vector3<T> diff = v1 - v2;

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& parent_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_1dof_constraint_error2, 3, 4>(
            new local_1dof_constraint_error2(parent_rotation)));
    }
};

struct local_1dof_constraint_error3
{
    const glm::dquat child_rotation;

    local_1dof_constraint_error3(const glm::dquat& child_rotation)
        : child_rotation(child_rotation)
    {}

    template <typename T>
    bool operator()(
        const T* const parent_rotation,
        T* residuals) const
    {
        const T child_rotation[] = { T(this->child_rotation.x), T(this->child_rotation.y), T(this->child_rotation.z), T(this->child_rotation.w) };

        Eigen::Map<const Eigen::Quaternion<T>> q1(parent_rotation);
        Eigen::Map<const Eigen::Quaternion<T>> q2(child_rotation);

        const Eigen::Vector3<T> axis(T(1.0), T(0.0), T(0.0));

        const Eigen::Vector3<T> v1 = q1 * axis;
        const Eigen::Vector3<T> v2 = q2 * axis;

        const Eigen::Vector3<T> diff = (v1 - v2) * T(100.0);

        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();

        return true;
    }

    static ceres::CostFunction* create(const glm::dquat& child_rotation)
    {
        return (new ceres::AutoDiffCostFunction<local_1dof_constraint_error3, 3, 4>(
            new local_1dof_constraint_error3(child_rotation)));
    }
};

void estimate_finger_pose(std::map<std::string, glm::mat4>& poses, const model_data& model)
{
    ceres::Problem problem;

    std::vector<std::tuple<std::string, std::string, std::string>> interpolations = {
        {"Thumb Proximal.R", "hand.R", "Thumb Intermediate.R"},
        {"Index Proximal.R", "hand.R", "Index Intermediate.R"},
        {"Middle Proximal.R", "hand.R", "Middle Intermediate.R"},
        {"Ring Proximal.R", "hand.R", "Ring Intermediate.R"},
        {"Little Proximal.R", "hand.R", "Little Intermediate.R"},
        {"Thumb Intermediate.R", "Thumb Proximal.R", "Thumb Distal.R"},
        {"Index Intermediate.R", "Index Proximal.R", "Index Distal.R"},
        {"Middle Intermediate.R", "Middle Proximal.R", "Middle Distal.R"},
        {"Ring Intermediate.R", "Ring Proximal.R", "Ring Distal.R"},
        {"Little Intermediate.R", "Little Proximal.R", "Little Distal.R"},
    };

    std::vector<std::pair<std::string, std::string>> joints = {
        {"Thumb Proximal.R", "hand.R"},
        {"Index Proximal.R", "hand.R"},
        {"Middle Proximal.R", "hand.R"},
        {"Ring Proximal.R", "hand.R"},
        {"Little Proximal.R", "hand.R"},
        {"Thumb Intermediate.R", "Thumb Proximal.R"},
        {"Index Intermediate.R", "Index Proximal.R"},
        {"Middle Intermediate.R", "Middle Proximal.R"},
        {"Ring Intermediate.R", "Ring Proximal.R"},
        {"Little Intermediate.R", "Little Proximal.R"},
        {"Thumb Distal.R", "Thumb Intermediate.R"},
        {"Index Distal.R", "Index Intermediate.R"},
        {"Middle Distal.R", "Middle Intermediate.R"},
        {"Ring Distal.R", "Ring Intermediate.R"},
        {"Little Distal.R", "Little Intermediate.R"},
    };

    std::vector<std::string> bones = {
        "Little Intermediate.R", "Ring Intermediate.R", "Middle Intermediate.R", "Index Intermediate.R", "Thumb Intermediate.R",
        "Little Proximal.R", "Ring Proximal.R", "Middle Proximal.R", "Index Proximal.R", "Thumb Proximal.R" };

    std::vector<std::string> target_bones = { "Little Distal.R", "Ring Distal.R", "Middle Distal.R", "Index Distal.R", "Thumb Distal.R" };

    std::vector<double> rotation_params(bones.size() * 4);
    std::vector<double> translation_params(bones.size() * 3);

    std::map<std::string, double*> bone_rotations;
    std::map<std::string, double*> bone_translations;
    for (std::size_t i = 0; i < bones.size(); i++)
    {
        const auto& bone = bones[i];
        double* rotation_param = &rotation_params[4 * i];
        const glm::dquat orientation = glm::quat_cast(poses.at(bone));
        const glm::vec3 position = glm::vec3(poses.at(bone)[3]);

        rotation_param[0] = orientation.x;
        rotation_param[1] = orientation.y;
        rotation_param[2] = orientation.z;
        rotation_param[3] = orientation.w;

        bone_rotations.insert(std::make_pair(bone, rotation_param));

        double* translation_param = &translation_params[3 * i];

        translation_param[0] = position.x;
        translation_param[1] = position.y;
        translation_param[2] = position.z;

        bone_translations.insert(std::make_pair(bone, translation_param));
    }

    {
        for (const auto& [bone, parent_bone, child_bone] : interpolations)
        {
            if (std::find(bones.begin(), bones.end(), parent_bone) == bones.end())
            {
                const glm::dquat parent_pose = glm::quat_cast(poses.at(parent_bone));
                double* rotation_param = bone_rotations.at(bone);
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_angle_error2::create(parent_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    rotation_param,
                    child_rotation_param);
            }
            else if (std::find(bones.begin(), bones.end(), child_bone) == bones.end())
            {
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                double* rotation_param = bone_rotations.at(bone);
                const glm::dquat child_pose = glm::quat_cast(poses.at(child_bone));

                ceres::CostFunction* cost_function =
                    local_angle_error3::create(child_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param,
                    rotation_param);
            }
            else
            {
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                double* rotation_param = bone_rotations.at(bone);
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_angle_error::create();

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param,
                    rotation_param,
                    child_rotation_param);
            }
        }
    }

    {
        for (const auto& [child_bone, parent_bone] : joints)
        {
            if (parent_bone == "hand.R")
            {
                continue;
            }
            if (parent_bone.find("Thumb") < parent_bone.size())
            {
                continue;
            }

            if (std::find(bones.begin(), bones.end(), parent_bone) == bones.end())
            {
                const glm::dquat parent_pose = glm::quat_cast(poses.at(parent_bone));
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_1dof_constraint_error2::create(parent_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    child_rotation_param);
            }
            else if (std::find(bones.begin(), bones.end(), child_bone) == bones.end())
            {
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                const glm::dquat child_pose = glm::quat_cast(poses.at(child_bone));

                ceres::CostFunction* cost_function =
                    local_1dof_constraint_error3::create(child_pose);

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param);
            }
            else
            {
                double* parent_rotation_param = bone_rotations.at(parent_bone);
                double* child_rotation_param = bone_rotations.at(child_bone);

                ceres::CostFunction* cost_function =
                    local_1dof_constraint_error::create();

                ceres::LossFunction* loss = nullptr; /* squared loss */

                problem.AddResidualBlock(cost_function,
                    loss,
                    parent_rotation_param,
                    child_rotation_param);
            }
        }
    }

    for (std::size_t i = 0; i < bones.size(); i++)
    {
        const auto& bone = bones[i];
        double* rotation_param = &rotation_params[4 * i];
        double* translation_param = &translation_params[3 * i];

        problem.SetManifold(rotation_param, new ceres::EigenQuaternionManifold());
    }

    {
        const auto start = std::chrono::system_clock::now();

        ceres::Solver::Options options;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Total elapsed for minimizer : " << elapsed / 1000.0 << " [ms]" << std::endl;
    }

    for (std::size_t i = 0; i < bones.size(); i++)
    {
        const auto& bone = bones[i];
        double* rotation_param = &rotation_params[4 * i];
        double* translation_param = &translation_params[3 * i];
        const auto orientation = glm::toMat3(glm::quat(rotation_param[3], rotation_param[0], rotation_param[1], rotation_param[2]));
        poses.at(bone) = glm::mat4(glm::vec4(orientation[0], 0.0), glm::vec4(orientation[1], 0.0), glm::vec4(orientation[2], 0.0), poses.at(bone)[3]);
    }

    for (const auto& [target_bone, parent_bone] : joints)
    {
        if (std::find(bones.begin(), bones.end(), parent_bone) != bones.end())
        {
            const auto bone_pose = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& bone) { return bone.name == target_bone; });
            const auto parent_bone_pose = std::find_if(model.bones.begin(), model.bones.end(), [&](const auto& bone) { return bone.name == parent_bone; });
            const auto bone_position = poses.at(parent_bone) * glm::inverse(parent_bone_pose->pose) * bone_pose->pose[3];

            poses.at(target_bone)[3] = bone_position;
        }
    }
}
