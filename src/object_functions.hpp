#pragma once

#include <glm/glm.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "automatic_differentiation.hpp"

template <typename T>
static void transform_coordinate(const T *p, const glm::mat4 &m, T *result)
{
    result[0] = ((double)m[0][0] * p[0]) + ((double)m[1][0] * p[1]) + ((double)m[2][0] * p[2]) + (double)m[3][0];
    result[1] = ((double)m[0][1] * p[0]) + ((double)m[1][1] * p[1]) + ((double)m[2][1] * p[2]) + (double)m[3][1];
    result[2] = ((double)m[0][2] * p[0]) + ((double)m[1][2] * p[1]) + ((double)m[2][2] * p[2]) + (double)m[3][2];
    result[3] = ((double)m[0][3] * p[0]) + ((double)m[1][3] * p[1]) + ((double)m[2][3] * p[2]) + (double)m[3][3];
}

template <typename T>
static void transform(const T *p, const glm::mat3 &m, T *result)
{
    result[0] = ((double)m[0][0] * p[0]) + ((double)m[1][0] * p[1]) + ((double)m[2][0] * p[2]);
    result[1] = ((double)m[0][1] * p[0]) + ((double)m[1][1] * p[1]) + ((double)m[2][1] * p[2]);
    result[2] = ((double)m[0][2] * p[0]) + ((double)m[1][2] * p[1]) + ((double)m[2][2] * p[2]);
}

template <typename T>
inline void rotate(const T axis[3], const T theta, const T pt[3], T result[3])
{
    if (theta > T(std::numeric_limits<double>::epsilon()))
    {
        const T costheta = cos(theta);
        const T sintheta = sin(theta);

        const T w_cross_pt[3] = {axis[1] * pt[2] - axis[2] * pt[1],
                                 axis[2] * pt[0] - axis[0] * pt[2],
                                 axis[0] * pt[1] - axis[1] * pt[0]};
        const T tmp =
            (axis[0] * pt[0] + axis[1] * pt[1] + axis[2] * pt[2]) * (T(1.0) - costheta);

        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + axis[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + axis[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + axis[2] * tmp;
    }
    else
    {
        const T w_cross_pt[3] = {axis[1] * pt[2] - axis[2] * pt[1],
                                 axis[2] * pt[0] - axis[0] * pt[2],
                                 axis[0] * pt[1] - axis[1] * pt[0]};

        result[0] = pt[0] + w_cross_pt[0] * theta;
        result[1] = pt[1] + w_cross_pt[1] * theta;
        result[2] = pt[2] + w_cross_pt[2] * theta;
    }
}

template <typename T>
inline vec3_t<T> rotate(const vec3_t<T> &axis_v, const T theta, const vec3_t<T> &pt_v)
{
    if (theta > T(std::numeric_limits<double>::epsilon()))
    {
        const T costheta = cos(theta);
        const T sintheta = sin(theta);

        const auto w_cross_pt_v = cross(axis_v, pt_v);
        const auto tmp_v = dot(axis_v, pt_v) * (T(1.0) - costheta);
        const auto result_v = pt_v * costheta + w_cross_pt_v * sintheta + axis_v * tmp_v;
        return result_v;
    }
    else
    {
        const auto w_cross_pt_v = cross(axis_v, pt_v);
        const auto result_v = pt_v + w_cross_pt_v * theta;
        return result_v;
    }
}

struct rt_transform_projection_error
{
    const glm::vec2 observed;
    const glm::vec3 point;
    const glm::mat4 view;
    const glm::mat3 proj;

    rt_transform_projection_error(const glm::vec2 &observed, const glm::vec3 &point, const glm::mat4 &view, const glm::mat3 &proj)
        : observed(observed), point(point), view(view), proj(proj) {}

    template <typename T>
    bool operator()(const T *const rotation,
                    const T *const translation,
                    T *residuals) const
    {
        if (observed.x < 0)
        {
            return false;
        }

        T point[3] = {
            T(static_cast<double>(this->point.x)),
            T(static_cast<double>(this->point.y)),
            T(static_cast<double>(this->point.z))};

        T p[3];
        ceres::AngleAxisRotatePoint(rotation, point, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        T view_p[4];
        transform_coordinate(p, view, view_p);

        view_p[0] = view_p[0] / view_p[3];
        view_p[1] = view_p[1] / view_p[3];
        view_p[2] = view_p[2] / view_p[3];

        T proj_p[3];
        transform(view_p, proj, proj_p);

        T predicted_x = proj_p[0] / proj_p[2];
        T predicted_y = proj_p[1] / proj_p[2];

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - (double)observed.x;
        residuals[1] = predicted_y - (double)observed.y;

        return true;
    }

    static ceres::CostFunction *create(const glm::vec2 &observed, const glm::vec3 &point, const glm::mat4 &view, const glm::mat3 &proj)
    {
        return (new ceres::AutoDiffCostFunction<rt_transform_projection_error, 2, 3, 3>(
            new rt_transform_projection_error(observed, point, view, proj)));
    }
};

struct twisted_rt_transform_projection_error
{
    const glm::vec2 observed;
    const glm::vec3 point;
    const glm::mat4 view_proj;
    float twist_weight;
    const line3 twist_axis;

    twisted_rt_transform_projection_error(const glm::vec2 &observed, const glm::vec3 &point, const glm::mat4 &view, const glm::mat3 &proj,
                                          float twist_weight, const line3 &twist_axis)
        : observed(observed), point(point), view_proj(glm::mat4(proj) * view), twist_weight(twist_weight), twist_axis(twist_axis) {}

    template <typename T>
    bool operator()(const T *const twist_angle,
                    const T *const rotation,
                    const T *const translation,
                    T *residuals) const
    {
        if (observed.x < 0)
        {
            return false;
        }

        T twist_angle_axis[3];
        twist_angle_axis[0] = static_cast<double>(twist_axis.direction.x * twist_weight) * twist_angle[0];
        twist_angle_axis[1] = static_cast<double>(twist_axis.direction.y * twist_weight) * twist_angle[0];
        twist_angle_axis[2] = static_cast<double>(twist_axis.direction.z * twist_weight) * twist_angle[0];

        T point[3] = {
            T(static_cast<double>(this->point.x - this->twist_axis.origin.x)),
            T(static_cast<double>(this->point.y - this->twist_axis.origin.y)),
            T(static_cast<double>(this->point.z - this->twist_axis.origin.z))};

        T twisted_p[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis, point, twisted_p);

        twisted_p[0] += static_cast<double>(this->twist_axis.origin.x);
        twisted_p[1] += static_cast<double>(this->twist_axis.origin.y);
        twisted_p[2] += static_cast<double>(this->twist_axis.origin.z);

        T p[3];
        ceres::AngleAxisRotatePoint(rotation, twisted_p, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        T proj_p[4];
        transform_coordinate(p, view_proj, proj_p);

        T predicted_x = proj_p[0] / proj_p[2];
        T predicted_y = proj_p[1] / proj_p[2];

        residuals[0] = predicted_x - static_cast<double>(observed.x);
        residuals[1] = predicted_y - static_cast<double>(observed.y);

        return true;
    }

    static ceres::CostFunction *create(const glm::vec2 &observed, const glm::vec3 &point, const glm::mat4 &view, const glm::mat3 &proj,
                                       float twist_weight, const line3 &twist_axis)
    {
        return (new ceres::AutoDiffCostFunction<twisted_rt_transform_projection_error, 2, 1, 3, 3>(
            new twisted_rt_transform_projection_error(observed, point, view, proj, twist_weight, twist_axis)));
    }
};

struct articulation_projection_error
{
    glm::vec3 point;
    const glm::mat4 view_proj;

    articulation_projection_error(const glm::vec3 &point, const glm::mat4 &view, const glm::mat3 &proj)
        : point(point), view_proj(glm::mat4(proj) * view) {}

    template <typename T>
    bool operator()(const T *const rotation1,
                    const T *const translation1,
                    const T *const rotation2,
                    const T *const translation2,
                    T *residuals) const
    {
        T point[3] = {
            T(static_cast<double>(this->point.x)),
            T(static_cast<double>(this->point.y)),
            T(static_cast<double>(this->point.z))};

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, point, p1);
        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        T proj_p1[4];
        transform_coordinate(p1, view_proj, proj_p1);

        T predicted1_x = proj_p1[0] / proj_p1[2];
        T predicted1_y = proj_p1[1] / proj_p1[2];

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, point, p2);
        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T proj_p2[4];
        transform_coordinate(p2, view_proj, proj_p2);

        T predicted2_x = proj_p2[0] / proj_p2[2];
        T predicted2_y = proj_p2[1] / proj_p2[2];

        // The error is the difference between the articulation positions.
        residuals[0] = predicted1_x - predicted2_x;
        residuals[1] = predicted1_y - predicted2_y;

        return true;
    }

    static ceres::CostFunction *create(const glm::vec3 &point, const glm::mat4 &view, const glm::mat3 &proj)
    {
        return (new ceres::AutoDiffCostFunction<articulation_projection_error, 2, 3, 3, 3, 3>(
            new articulation_projection_error(point, view, proj)));
    }
};

struct rt_transform_error
{
    const glm::vec3 point;
    const glm::vec3 target;

    rt_transform_error(const glm::vec3 &point, const glm::vec3 &target)
        : point(point), target(target) {}

    template <typename T>
    bool operator()(const T *const rotation,
                    const T *const translation,
                    T *residuals) const
    {
        T point[3] = {
            T(static_cast<double>(this->point.x)),
            T(static_cast<double>(this->point.y)),
            T(static_cast<double>(this->point.z))};

        T p[3];
        ceres::AngleAxisRotatePoint(rotation, point, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        T predicted_x = p[0];
        T predicted_y = p[1];
        T predicted_z = p[2];

        // The error is the difference between the predicted and target position.
        residuals[0] = predicted_x - T(static_cast<double>(this->target.x));
        residuals[1] = predicted_y - T(static_cast<double>(this->target.y));
        residuals[2] = predicted_z - T(static_cast<double>(this->target.z));

        return true;
    }

    static ceres::CostFunction *create(const glm::vec3 &point, const glm::vec3 &target)
    {
        return (new ceres::AutoDiffCostFunction<rt_transform_error, 3, 3, 3>(
            new rt_transform_error(point, target)));
    }
};

struct quat_rt_transform_error
{
    const glm::vec3 point;
    const glm::vec3 target;

    quat_rt_transform_error(const glm::vec3& point, const glm::vec3& target)
        : point(point), target(target) {}

    template <typename T>
    bool operator()(const T* const rotation,
        const T* const translation,
        T* residuals) const
    {
        T point[3] = {
            T(static_cast<double>(this->point.x)),
            T(static_cast<double>(this->point.y)),
            T(static_cast<double>(this->point.z)) };

        T p[3];
        ceres::QuaternionRotatePoint(rotation, point, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        T predicted_x = p[0];
        T predicted_y = p[1];
        T predicted_z = p[2];

        // The error is the difference between the predicted and target position.
        residuals[0] = predicted_x - T(static_cast<double>(this->target.x));
        residuals[1] = predicted_y - T(static_cast<double>(this->target.y));
        residuals[2] = predicted_z - T(static_cast<double>(this->target.z));

        return true;
    }

    static ceres::CostFunction* create(const glm::vec3& point, const glm::vec3& target)
    {
        return (new ceres::AutoDiffCostFunction<quat_rt_transform_error, 3, 4, 3>(
            new quat_rt_transform_error(point, target)));
    }
};

struct twisted_rt_transform_error
{
    const glm::vec3 point;
    const glm::vec3 target;
    const float twist_weight;
    const line3 twist_axis;

    twisted_rt_transform_error(const glm::vec3 &point, const glm::vec3 &target,
                               float twist_weight, const line3 &twist_axis)
        : point(point), target(target), twist_weight(twist_weight), twist_axis(twist_axis) {}

    template <typename T>
    bool operator()(const T *const twist_angle,
                    const T *const rotation,
                    const T *const translation,
                    T *residuals) const
    {
        T twist_angle_axis[3];
        twist_angle_axis[0] = static_cast<double>(twist_axis.direction.x * twist_weight) * twist_angle[0];
        twist_angle_axis[1] = static_cast<double>(twist_axis.direction.y * twist_weight) * twist_angle[0];
        twist_angle_axis[2] = static_cast<double>(twist_axis.direction.z * twist_weight) * twist_angle[0];

        T point[3] = {
            T(static_cast<double>(this->point.x - this->twist_axis.origin.x)),
            T(static_cast<double>(this->point.y - this->twist_axis.origin.y)),
            T(static_cast<double>(this->point.z - this->twist_axis.origin.z))};

        T twisted_p[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis, point, twisted_p);

        twisted_p[0] += static_cast<double>(this->twist_axis.origin.x);
        twisted_p[1] += static_cast<double>(this->twist_axis.origin.y);
        twisted_p[2] += static_cast<double>(this->twist_axis.origin.z);

        T p[3];
        ceres::AngleAxisRotatePoint(rotation, twisted_p, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        T predicted_x = p[0];
        T predicted_y = p[1];
        T predicted_z = p[2];

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(static_cast<double>(this->target.x));
        residuals[1] = predicted_y - T(static_cast<double>(this->target.y));
        residuals[2] = predicted_z - T(static_cast<double>(this->target.z));

        return true;
    }

    static ceres::CostFunction *create(const glm::vec3 &point, const glm::vec3 &target,
                                       float twist_weight, const line3 &twist_axis)
    {
        return (new ceres::AutoDiffCostFunction<twisted_rt_transform_error, 3, 1, 3, 3>(
            new twisted_rt_transform_error(point, target, twist_weight, twist_axis)));
    }
};

struct quat_twisted_rt_transform_error
{
    const glm::vec3 point;
    const glm::vec3 target;
    const float twist_weight;
    const line3 twist_axis;

    quat_twisted_rt_transform_error(const glm::vec3& point, const glm::vec3& target,
        float twist_weight, const line3& twist_axis)
        : point(point), target(target), twist_weight(twist_weight), twist_axis(twist_axis) {}

    template <typename T>
    bool operator()(const T* const twist_angle,
        const T* const rotation,
        const T* const translation,
        T* residuals) const
    {
        T twist_angle_axis[3];
        twist_angle_axis[0] = static_cast<double>(twist_axis.direction.x * twist_weight) * twist_angle[0];
        twist_angle_axis[1] = static_cast<double>(twist_axis.direction.y * twist_weight) * twist_angle[0];
        twist_angle_axis[2] = static_cast<double>(twist_axis.direction.z * twist_weight) * twist_angle[0];

        T point[3] = {
            T(static_cast<double>(this->point.x - this->twist_axis.origin.x)),
            T(static_cast<double>(this->point.y - this->twist_axis.origin.y)),
            T(static_cast<double>(this->point.z - this->twist_axis.origin.z)) };

        T twisted_p[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis, point, twisted_p);

        twisted_p[0] += static_cast<double>(this->twist_axis.origin.x);
        twisted_p[1] += static_cast<double>(this->twist_axis.origin.y);
        twisted_p[2] += static_cast<double>(this->twist_axis.origin.z);

        T p[3];
        ceres::QuaternionRotatePoint(rotation, twisted_p, p);
        p[0] += translation[0];
        p[1] += translation[1];
        p[2] += translation[2];

        T predicted_x = p[0];
        T predicted_y = p[1];
        T predicted_z = p[2];

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(static_cast<double>(this->target.x));
        residuals[1] = predicted_y - T(static_cast<double>(this->target.y));
        residuals[2] = predicted_z - T(static_cast<double>(this->target.z));

        return true;
    }

    static ceres::CostFunction* create(const glm::vec3& point, const glm::vec3& target,
        float twist_weight, const line3& twist_axis)
    {
        return (new ceres::AutoDiffCostFunction<quat_twisted_rt_transform_error, 3, 1, 4, 3>(
            new quat_twisted_rt_transform_error(point, target, twist_weight, twist_axis)));
    }
};

struct articulation_point_error
{
    const glm::vec3 point1;
    const glm::vec3 point2;

    articulation_point_error(const glm::vec3 &point1, const glm::vec3 &point2)
        : point1(point1), point2(point2) {}

    template <typename T>
    bool operator()(const T *const rotation1,
                    const T *const translation1,
                    const T *const rotation2,
                    const T *const translation2,
                    T *residuals) const
    {
        T point1[3] = {
            T(static_cast<double>(this->point1.x)),
            T(static_cast<double>(this->point1.y)),
            T(static_cast<double>(this->point1.z))};
        T point2[3] = {
            T(static_cast<double>(this->point2.x)),
            T(static_cast<double>(this->point2.y)),
            T(static_cast<double>(this->point2.z))};

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, point1, p1);
        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        T predicted1_x = p1[0];
        T predicted1_y = p1[1];
        T predicted1_z = p1[2];

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, point2, p2);
        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T predicted2_x = p2[0];
        T predicted2_y = p2[1];
        T predicted2_z = p2[2];

        // The error is the difference between the articulation positions.
        residuals[0] = predicted1_x - predicted2_x;
        residuals[1] = predicted1_y - predicted2_y;
        residuals[2] = predicted1_z - predicted2_z;

        return true;
    }

    static ceres::CostFunction *create(const glm::vec3 &point1, const glm::vec3 &point2)
    {
        return (new ceres::AutoDiffCostFunction<articulation_point_error, 3, 3, 3, 3, 3>(
            new articulation_point_error(point1, point2)));
    }
};

struct quat_articulation_point_error
{
    const glm::vec3 point1;
    const glm::vec3 point2;

    quat_articulation_point_error(const glm::vec3& point1, const glm::vec3& point2)
        : point1(point1), point2(point2) {}

    template <typename T>
    bool operator()(const T* const rotation1,
        const T* const translation1,
        const T* const rotation2,
        const T* const translation2,
        T* residuals) const
    {
        T point1[3] = {
            T(static_cast<double>(this->point1.x)),
            T(static_cast<double>(this->point1.y)),
            T(static_cast<double>(this->point1.z)) };
        T point2[3] = {
            T(static_cast<double>(this->point2.x)),
            T(static_cast<double>(this->point2.y)),
            T(static_cast<double>(this->point2.z)) };

        // Transform articulation by parent
        T p1[3];
        ceres::QuaternionRotatePoint(rotation1, point1, p1);
        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        T predicted1_x = p1[0];
        T predicted1_y = p1[1];
        T predicted1_z = p1[2];

        // Transform articulation by child
        T p2[3];
        ceres::QuaternionRotatePoint(rotation2, point2, p2);
        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T predicted2_x = p2[0];
        T predicted2_y = p2[1];
        T predicted2_z = p2[2];

        // The error is the difference between the articulation positions.
        residuals[0] = predicted1_x - predicted2_x;
        residuals[1] = predicted1_y - predicted2_y;
        residuals[2] = predicted1_z - predicted2_z;

        return true;
    }

    static ceres::CostFunction* create(const glm::vec3& point1, const glm::vec3& point2)
    {
        return (new ceres::AutoDiffCostFunction<quat_articulation_point_error, 3, 4, 3, 4, 3>(
            new quat_articulation_point_error(point1, point2)));
    }
};

struct twist_articulation_point_error
{
    const glm::vec3 point1;
    const glm::vec3 point2;
    const float twist_weight1;
    const line3 twist_axis1;
    const float twist_weight2;
    const line3 twist_axis2;

    twist_articulation_point_error(glm::vec3 point1, glm::vec3 point2,
                                   float twist_weight1, const line3 &twist_axis1, float twist_weight2, const line3 &twist_axis2)
        : point1(point1), point2(point2), twist_weight1(twist_weight1), twist_axis1(twist_axis1),
          twist_weight2(twist_weight2), twist_axis2(twist_axis2) {}

    template <typename T>
    bool operator()(const T *const twist_angle1,
                    const T *const rotation1,
                    const T *const translation1,
                    const T *const twist_angle2,
                    const T *const rotation2,
                    const T *const translation2,
                    T *residuals) const
    {
        T twist_angle_axis1[3];
        twist_angle_axis1[0] = static_cast<double>(twist_axis1.direction.x * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[1] = static_cast<double>(twist_axis1.direction.y * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[2] = static_cast<double>(twist_axis1.direction.z * twist_weight1) * twist_angle1[0];

        T twist_angle_axis2[3];
        twist_angle_axis2[0] = static_cast<double>(twist_axis2.direction.x * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[1] = static_cast<double>(twist_axis2.direction.y * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[2] = static_cast<double>(twist_axis2.direction.z * twist_weight2) * twist_angle2[0];

        T point1[3] = {
            T(static_cast<double>(this->point1.x - this->twist_axis1.origin.x)),
            T(static_cast<double>(this->point1.y - this->twist_axis1.origin.y)),
            T(static_cast<double>(this->point1.z - this->twist_axis1.origin.z))};

        T point2[3] = {
            T(static_cast<double>(this->point2.x - this->twist_axis2.origin.x)),
            T(static_cast<double>(this->point2.y - this->twist_axis2.origin.y)),
            T(static_cast<double>(this->point2.z - this->twist_axis2.origin.z))};

        T twisted_p1[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis1, point1, twisted_p1);

        twisted_p1[0] += static_cast<double>(this->twist_axis1.origin.x);
        twisted_p1[1] += static_cast<double>(this->twist_axis1.origin.y);
        twisted_p1[2] += static_cast<double>(this->twist_axis1.origin.z);

        T twisted_p2[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis2, point2, twisted_p2);

        twisted_p2[0] += static_cast<double>(this->twist_axis2.origin.x);
        twisted_p2[1] += static_cast<double>(this->twist_axis2.origin.y);
        twisted_p2[2] += static_cast<double>(this->twist_axis2.origin.z);

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, twisted_p1, p1);
        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        T predicted1_x = p1[0];
        T predicted1_y = p1[1];
        T predicted1_z = p1[2];

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, twisted_p2, p2);
        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T predicted2_x = p2[0];
        T predicted2_y = p2[1];
        T predicted2_z = p2[2];

        // The error is the difference between the articulation positions.
        residuals[0] = predicted1_x - predicted2_x;
        residuals[1] = predicted1_y - predicted2_y;
        residuals[2] = predicted1_z - predicted2_z;

        return true;
    }

    static ceres::CostFunction *create(glm::vec3 point1, glm::vec3 point2,
                                       float twist_weight1, const line3 &twist_axis1, float twist_weight2, const line3 &twist_axis2)
    {
        return (new ceres::AutoDiffCostFunction<twist_articulation_point_error, 3, 1, 3, 3, 1, 3, 3>(
            new twist_articulation_point_error(point1, point2, twist_weight1, twist_axis1, twist_weight2, twist_axis2)));
    }
};

struct quat_twist_articulation_point_error
{
    const glm::vec3 point1;
    const glm::vec3 point2;
    const float twist_weight1;
    const line3 twist_axis1;
    const float twist_weight2;
    const line3 twist_axis2;

    quat_twist_articulation_point_error(glm::vec3 point1, glm::vec3 point2,
        float twist_weight1, const line3& twist_axis1, float twist_weight2, const line3& twist_axis2)
        : point1(point1), point2(point2), twist_weight1(twist_weight1), twist_axis1(twist_axis1),
        twist_weight2(twist_weight2), twist_axis2(twist_axis2) {}

    template <typename T>
    bool operator()(const T* const twist_angle1,
        const T* const rotation1,
        const T* const translation1,
        const T* const twist_angle2,
        const T* const rotation2,
        const T* const translation2,
        T* residuals) const
    {
        T twist_angle_axis1[3];
        twist_angle_axis1[0] = static_cast<double>(twist_axis1.direction.x * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[1] = static_cast<double>(twist_axis1.direction.y * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[2] = static_cast<double>(twist_axis1.direction.z * twist_weight1) * twist_angle1[0];

        T twist_angle_axis2[3];
        twist_angle_axis2[0] = static_cast<double>(twist_axis2.direction.x * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[1] = static_cast<double>(twist_axis2.direction.y * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[2] = static_cast<double>(twist_axis2.direction.z * twist_weight2) * twist_angle2[0];

        T point1[3] = {
            T(static_cast<double>(this->point1.x - this->twist_axis1.origin.x)),
            T(static_cast<double>(this->point1.y - this->twist_axis1.origin.y)),
            T(static_cast<double>(this->point1.z - this->twist_axis1.origin.z)) };

        T point2[3] = {
            T(static_cast<double>(this->point2.x - this->twist_axis2.origin.x)),
            T(static_cast<double>(this->point2.y - this->twist_axis2.origin.y)),
            T(static_cast<double>(this->point2.z - this->twist_axis2.origin.z)) };

        T twisted_p1[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis1, point1, twisted_p1);

        twisted_p1[0] += static_cast<double>(this->twist_axis1.origin.x);
        twisted_p1[1] += static_cast<double>(this->twist_axis1.origin.y);
        twisted_p1[2] += static_cast<double>(this->twist_axis1.origin.z);

        T twisted_p2[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis2, point2, twisted_p2);

        twisted_p2[0] += static_cast<double>(this->twist_axis2.origin.x);
        twisted_p2[1] += static_cast<double>(this->twist_axis2.origin.y);
        twisted_p2[2] += static_cast<double>(this->twist_axis2.origin.z);

        // Transform articulation by parent
        T p1[3];
        ceres::QuaternionRotatePoint(rotation1, twisted_p1, p1);
        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        T predicted1_x = p1[0];
        T predicted1_y = p1[1];
        T predicted1_z = p1[2];

        // Transform articulation by child
        T p2[3];
        ceres::QuaternionRotatePoint(rotation2, twisted_p2, p2);
        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T predicted2_x = p2[0];
        T predicted2_y = p2[1];
        T predicted2_z = p2[2];

        // The error is the difference between the articulation positions.
        residuals[0] = predicted1_x - predicted2_x;
        residuals[1] = predicted1_y - predicted2_y;
        residuals[2] = predicted1_z - predicted2_z;

        return true;
    }

    static ceres::CostFunction* create(glm::vec3 point1, glm::vec3 point2,
        float twist_weight1, const line3& twist_axis1, float twist_weight2, const line3& twist_axis2)
    {
        return (new ceres::AutoDiffCostFunction<quat_twist_articulation_point_error, 3, 1, 4, 3, 1, 4, 3>(
            new quat_twist_articulation_point_error(point1, point2, twist_weight1, twist_axis1, twist_weight2, twist_axis2)));
    }
};

struct srt_least_square_functor
{
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> target_points;
    glm::vec3 origin;

    srt_least_square_functor(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &target_points, const glm::vec3 &origin)
        : points(points), target_points(target_points), origin(origin)
    {
    }
    template <typename T>
    bool operator()(const T *const scale,
                    const T *const rotation,
                    const T *const translation,
                    T *residuals) const
    {
        T residual(0);

        for (std::size_t i = 0; i < points.size(); i++)
        {
            const auto &pt = this->points[i];
            const auto &target = this->target_points[i];

            T point[3] = {
                T(static_cast<double>(pt.x - origin.x)),
                T(static_cast<double>(pt.y - origin.y)),
                T(static_cast<double>(pt.z - origin.z))};

            point[0] *= scale[0];
            point[1] *= scale[1];
            point[2] *= scale[2];

            T p[3];
            ceres::AngleAxisRotatePoint(rotation, point, p);

            p[0] += static_cast<double>(origin.x);
            p[1] += static_cast<double>(origin.y);
            p[2] += static_cast<double>(origin.z);

            p[0] += translation[0];
            p[1] += translation[1];
            p[2] += translation[2];

            T residuals[3];
            residuals[0] = p[0] - static_cast<double>(target.x);
            residuals[1] = p[1] - static_cast<double>(target.y);
            residuals[2] = p[2] - static_cast<double>(target.z);

            residual += (residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]) * 0.5;
        }

        residuals[0] = residual;

        return true;
    }
};

struct srt_articulation_least_square_constraint
{
    glm::vec3 point1;
    glm::vec3 point2;
    glm::vec3 parent_origin;
    glm::vec3 child_origin;

    srt_articulation_least_square_constraint(glm::vec3 point1, glm::vec3 point2, glm::vec3 parent_origin, glm::vec3 child_origin)
        : point1(point1), point2(point2), parent_origin(parent_origin), child_origin(child_origin)
    {
    }

    template <typename T>
    bool operator()(const T *const scale1,
                    const T *const rotation1,
                    const T *const translation1,
                    const T *const scale2,
                    const T *const rotation2,
                    const T *const translation2,
                    T *residuals) const
    {
        T point1[3] = {
            T(static_cast<double>(this->point1.x - this->parent_origin.x)),
            T(static_cast<double>(this->point1.y - this->parent_origin.y)),
            T(static_cast<double>(this->point1.z - this->parent_origin.z))};
        T point2[3] = {
            T(static_cast<double>(this->point2.x - this->child_origin.x)),
            T(static_cast<double>(this->point2.y - this->child_origin.y)),
            T(static_cast<double>(this->point2.z - this->child_origin.z))};

        point1[0] *= scale1[0];
        point1[1] *= scale1[1];
        point1[2] *= scale1[2];

        point2[0] *= scale2[0];
        point2[1] *= scale2[1];
        point2[2] *= scale2[2];

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, point1, p1);

        p1[0] += static_cast<double>(this->parent_origin.x);
        p1[1] += static_cast<double>(this->parent_origin.y);
        p1[2] += static_cast<double>(this->parent_origin.z);

        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, point2, p2);

        p2[0] += static_cast<double>(this->child_origin.x);
        p2[1] += static_cast<double>(this->child_origin.y);
        p2[2] += static_cast<double>(this->child_origin.z);

        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T constraints[3];
        constraints[0] = (p1[0] - p2[0]);
        constraints[1] = (p1[1] - p2[1]);
        constraints[2] = (p1[2] - p2[2]);
        residuals[0] = (constraints[0] * constraints[0] + constraints[1] * constraints[1] + constraints[2] * constraints[2]) * 10.0;

        return true;
    }
};

struct sr_least_square_functor
{
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> target_points;
    glm::vec3 origin;

    sr_least_square_functor(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &target_points, const glm::vec3 &origin)
        : points(points), target_points(target_points), origin(origin)
    {
    }
    template <typename T>
    bool operator()(const T *const scale,
                    const T *const rotation,
                    T *residuals) const
    {
        T residual(0);

        for (std::size_t i = 0; i < points.size(); i++)
        {
            const auto &pt = this->points[i];
            const auto &target = this->target_points[i];

            T point[3] = {
                T(static_cast<double>(pt.x - origin.x)),
                T(static_cast<double>(pt.y - origin.y)),
                T(static_cast<double>(pt.z - origin.z))};

            point[0] *= scale[0];
            point[1] *= scale[1];
            point[2] *= scale[2];

            T p[3];
            ceres::AngleAxisRotatePoint(rotation, point, p);

            p[0] += static_cast<double>(origin.x);
            p[1] += static_cast<double>(origin.y);
            p[2] += static_cast<double>(origin.z);

            T residuals[3];
            residuals[0] = p[0] - static_cast<double>(target.x);
            residuals[1] = p[1] - static_cast<double>(target.y);
            residuals[2] = p[2] - static_cast<double>(target.z);

            residual += (residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]) * 10.0;
        }

        residuals[0] = residual;

        return true;
    }
};

struct sr_articulation_least_square_constraint
{
    glm::vec3 point1;
    glm::vec3 point2;
    glm::vec3 parent_origin;
    glm::vec3 child_origin;

    sr_articulation_least_square_constraint(glm::vec3 point1, glm::vec3 point2, glm::vec3 parent_origin, glm::vec3 child_origin)
        : point1(point1), point2(point2), parent_origin(parent_origin), child_origin(child_origin)
    {
    }
    template <typename T>
    bool operator()(const T *const scale1,
                    const T *const rotation1,
                    const T *const scale2,
                    const T *const rotation2,
                    T *residuals) const
    {
        T point1[3] = {
            T(static_cast<double>(this->point1.x - this->parent_origin.x)),
            T(static_cast<double>(this->point1.y - this->parent_origin.y)),
            T(static_cast<double>(this->point1.z - this->parent_origin.z))};
        T point2[3] = {
            T(static_cast<double>(this->point2.x - this->child_origin.x)),
            T(static_cast<double>(this->point2.y - this->child_origin.y)),
            T(static_cast<double>(this->point2.z - this->child_origin.z))};

        point1[0] *= scale1[0];
        point1[1] *= scale1[1];
        point1[2] *= scale1[2];

        point2[0] *= scale2[0];
        point2[1] *= scale2[1];
        point2[2] *= scale2[2];

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, point1, p1);

        p1[0] += static_cast<double>(this->parent_origin.x);
        p1[1] += static_cast<double>(this->parent_origin.y);
        p1[2] += static_cast<double>(this->parent_origin.z);

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, point2, p2);

        p2[0] += static_cast<double>(this->child_origin.x);
        p2[1] += static_cast<double>(this->child_origin.y);
        p2[2] += static_cast<double>(this->child_origin.z);

        T constraints[3];
        constraints[0] = p1[0] - p2[0];
        constraints[1] = p1[1] - p2[1];
        constraints[2] = p1[2] - p2[2];
        residuals[0] = (constraints[0] * constraints[0] + constraints[1] * constraints[1] + constraints[2] * constraints[2]) * 10.0;
    }
};

struct rt_least_square_functor
{
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> target_points;
    std::vector<double> weights;

    rt_least_square_functor(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &target_points, const std::vector<double> &weights)
        : points(points), target_points(target_points), weights(weights)
    {
    }
    template <typename T>
    bool operator()(const T *const rotation,
                    const T *const translation,
                    T *residuals) const
    {
        T residual(0);

        for (std::size_t i = 0; i < points.size(); i++)
        {
            const auto &pt = this->points[i];
            const auto &target = this->target_points[i];

            T point[3] = {
                T(static_cast<double>(pt.x)),
                T(static_cast<double>(pt.y)),
                T(static_cast<double>(pt.z))};

            T p[3];
            ceres::AngleAxisRotatePoint(rotation, point, p);

            p[0] += translation[0];
            p[1] += translation[1];
            p[2] += translation[2];

            T residuals[3];
            residuals[0] = p[0] - static_cast<double>(target.x);
            residuals[1] = p[1] - static_cast<double>(target.y);
            residuals[2] = p[2] - static_cast<double>(target.z);

            residual += weights[i] * (residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]);
        }

        residuals[0] = residual;

        return true;
    }
    template <typename T>
    inline T operator()(const T *const rotation,
                        const T *const translation) const
    {
        T residual(0);

        vec3_t<T> r_vec(rotation[0], rotation[1], rotation[2]);
        vec3_t<T> t_vec(translation[0], translation[1], translation[2]);

        const T theta2 = dot(r_vec, r_vec);
        const T theta = sqrt(theta2);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = T(1.0) / theta;
        const vec3_t<T> w = r_vec * theta_inverse;

        for (std::size_t i = 0; i < points.size(); i++)
        {
            const auto &pt = this->points[i];
            const auto &target = this->target_points[i];

#if 0
            T point[3] = {
                T(static_cast<double>(pt.x)),
                T(static_cast<double>(pt.y)),
                T(static_cast<double>(pt.z))};

            T p[3];
            ceres::AngleAxisRotatePoint(rotation, point, p);

            p[0] += translation[0];
            p[1] += translation[1];
            p[2] += translation[2];

            T residuals[3];
            residuals[0] = p[0] - static_cast<double>(target.x);
            residuals[1] = p[1] - static_cast<double>(target.y);
            residuals[2] = p[2] - static_cast<double>(target.z);

            residual += weights[i] * (residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]);
#else
            vec3_t<T> point_vec(T(pt.x), T(pt.y), T(pt.z));
            vec3_t<T> target_vec(T(target.x), T(target.y), T(target.z));

#if 1
            const auto p_vec = rotate_angle_axis(r_vec, point_vec);
#else
            vec3_t<T> p_vec;
            {
                const T theta2 = dot(r_vec, r_vec);
                if (theta2 > T(std::numeric_limits<double>::epsilon()))
                {
                    const vec3_t<T> w_cross_pt = cross(w, point_vec);
                    const T tmp = dot(w, point_vec) * (T(1.0) - costheta);

                    p_vec = point_vec * costheta + w_cross_pt * sintheta + w * tmp;
                }
                else
                {
                    const vec3_t<T> w_cross_pt = cross(r_vec, point_vec);
                    p_vec = point_vec + w_cross_pt;
                }
            }
#endif

            const auto res_vec = p_vec + t_vec - target_vec;

            residual += weights[i] * dot(res_vec, res_vec);
#endif
        }

        return residual;
    }

    // std::vector<std::shared_ptr<parameter>> parameters;

    // template <typename T>
    // T operator()(const T *params) const
    // {
    //     assert(parameters.size() == 2);
    //     const auto rotation = params + parameters[0]->offset;
    //     const auto translation = params + parameters[1]->offset;

    //     T residual(0);
    //     this->operator()(rotation, translation, &residual);
    //     return residual;
    // }
};

struct twisted_rt_least_square_functor
{
    const std::vector<glm::vec3> points;
    const std::vector<glm::vec3> target_points;
    const std::vector<double> weights;
    const std::vector<double> twist_weights;
    const line3 twist_axis;

    twisted_rt_least_square_functor(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &target_points,
                                    const std::vector<double> &twist_weights, const line3 &twist_axis, const std::vector<double> &weights)
        : points(points), target_points(target_points), twist_weights(twist_weights), twist_axis(twist_axis), weights(weights) {}

    template <typename T>
    inline T operator()(const T *const rotation,
                        const T *const translation,
                        const T *const twist_angle) const
    {
        T residual(0);

        for (std::size_t i = 0; i < points.size(); i++)
        {
            const auto &source = this->points[i];
            const auto &target = this->target_points[i];
            const auto &twist_weight = this->twist_weights[i];

            T twist_angle_axis[3];
            twist_angle_axis[0] = static_cast<double>(twist_axis.direction.x * twist_weight) * twist_angle[0];
            twist_angle_axis[1] = static_cast<double>(twist_axis.direction.y * twist_weight) * twist_angle[0];
            twist_angle_axis[2] = static_cast<double>(twist_axis.direction.z * twist_weight) * twist_angle[0];

            T point[3] = {
                T(static_cast<double>(source.x - twist_axis.origin.x)),
                T(static_cast<double>(source.y - twist_axis.origin.y)),
                T(static_cast<double>(source.z - twist_axis.origin.z))};

            T twisted_p[3];
            ceres::AngleAxisRotatePoint(twist_angle_axis, point, twisted_p);

            twisted_p[0] += static_cast<double>(twist_axis.origin.x);
            twisted_p[1] += static_cast<double>(twist_axis.origin.y);
            twisted_p[2] += static_cast<double>(twist_axis.origin.z);

            T p[3];
            ceres::AngleAxisRotatePoint(rotation, twisted_p, p);
            p[0] += translation[0];
            p[1] += translation[1];
            p[2] += translation[2];

            T predicted_x = p[0];
            T predicted_y = p[1];
            T predicted_z = p[2];

            // The error is the difference between the predicted and observed position.
            T residuals[3];
            residuals[0] = predicted_x - T(static_cast<double>(target.x));
            residuals[1] = predicted_y - T(static_cast<double>(target.y));
            residuals[2] = predicted_z - T(static_cast<double>(target.z));

            residual += 1.0 * (residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]);
        }

        return residual;
    }

    static ceres::CostFunction *create(const glm::vec3 point, const glm::vec3 target,
                                       float twist_weight, const line3 &twist_axis)
    {
        return (new ceres::AutoDiffCostFunction<twisted_rt_transform_error, 3, 1, 3, 3>(
            new twisted_rt_transform_error(point, target, twist_weight, twist_axis)));
    }
};

struct quat_rt_least_square_functor
{
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> target_points;
    std::vector<double> weights;

    quat_rt_least_square_functor(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &target_points, const std::vector<double> &weights)
        : points(points), target_points(target_points), weights(weights)
    {
    }

    template <typename T>
    inline T operator()(const T *const rotation,
                        const T *const translation) const
    {
        T residual(0);

        quat_t<T> r_quat(rotation[0], rotation[1], rotation[2], rotation[3]);
        vec3_t<T> t_vec(translation[0], translation[1], translation[2]);

        for (std::size_t i = 0; i < points.size(); i++)
        {
            const auto &pt = this->points[i];
            const auto &target = this->target_points[i];

            vec3_t<T> point_vec(T(pt.x), T(pt.y), T(pt.z));
            vec3_t<T> target_vec(T(target.x), T(target.y), T(target.z));

            const auto p_vec = rotate(r_quat, point_vec);

            const auto res_vec = p_vec + t_vec - target_vec;

            residual += weights[i] * dot(res_vec, res_vec);
        }

        return residual;
    }
};

struct quat_twisted_rt_least_square_functor
{
    const std::vector<glm::vec3> points;
    const std::vector<glm::vec3> target_points;
    const std::vector<double> weights;
    const std::vector<double> twist_weights;
    const line3 twist_axis;

    quat_twisted_rt_least_square_functor(const std::vector<glm::vec3> &points, const std::vector<glm::vec3> &target_points,
                                         const std::vector<double> &twist_weights, const line3 &twist_axis, const std::vector<double> &weights)
        : points(points), target_points(target_points), twist_weights(twist_weights), twist_axis(twist_axis), weights(weights) {}

    template <typename T>
    inline T operator()(const T *const rotation,
                        const T *const translation,
                        const T *const twist_angle) const
    {
        T residual(0);

        const quat_t<T> r_quat(rotation[0], rotation[1], rotation[2], rotation[3]);
        const vec3_t<T> t_vec(translation[0], translation[1], translation[2]);
        const vec3_t<T> origin_vec(T(twist_axis.origin.x), T(twist_axis.origin.y), T(twist_axis.origin.z));
        const vec3_t<T> axis_vec(T(twist_axis.direction.x), T(twist_axis.direction.y), T(twist_axis.direction.z));

        T twist_angle_axis[3];
        twist_angle_axis[0] = T(static_cast<double>(twist_axis.direction.x));
        twist_angle_axis[1] = T(static_cast<double>(twist_axis.direction.y));
        twist_angle_axis[2] = T(static_cast<double>(twist_axis.direction.z));

        for (std::size_t i = 0; i < points.size(); i++)
        {
            const auto &pt = this->points[i];
            const auto &target = this->target_points[i];
            const auto &twist_weight = this->twist_weights[i];

            vec3_t<T> pt_vec(T(pt.x), T(pt.y), T(pt.z));
#if 0
            const T sintheta = sin(0.5 * twist_angle[0]);
            const T costheta = cos(0.5 * twist_angle[0]);
            quat_t<T> twist_quat(costheta, sintheta * twist_angle_axis[0], sintheta * twist_angle_axis[1], sintheta * twist_angle_axis[2]);

            const auto point_vec = rotate(twist_quat, pt_vec - origin_vec) + origin_vec;
#else
            const auto point_vec = rotate(axis_vec, T(twist_weight) * twist_angle[0], pt_vec - origin_vec) + origin_vec;
#endif
            const vec3_t<T> target_vec(T(target.x), T(target.y), T(target.z));

            const auto p_vec = rotate(r_quat, point_vec);

            const auto res_vec = p_vec + t_vec - target_vec;

            residual += weights[i] * l2norm(res_vec);
        }

        return residual;
    }

    static ceres::CostFunction *create(const glm::vec3 point, const glm::vec3 target,
                                       float twist_weight, const line3 &twist_axis)
    {
        return (new ceres::AutoDiffCostFunction<quat_twisted_rt_transform_error, 3, 1, 3, 3>(
            new quat_twisted_rt_transform_error(point, target, twist_weight, twist_axis)));
    }
};

struct rt_articulation_least_square_constraint
{
    glm::vec3 point1;
    glm::vec3 point2;

    rt_articulation_least_square_constraint(glm::vec3 point1, glm::vec3 point2)
        : point1(point1), point2(point2)
    {
    }

    template <typename T>
    inline T operator()(const T *const rotation1,
                        const T *const translation1,
                        const T *const rotation2,
                        const T *const translation2) const
    {

        T point1[3] = {
            T(static_cast<double>(this->point1.x)),
            T(static_cast<double>(this->point1.y)),
            T(static_cast<double>(this->point1.z))};
        T point2[3] = {
            T(static_cast<double>(this->point2.x)),
            T(static_cast<double>(this->point2.y)),
            T(static_cast<double>(this->point2.z))};

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, point1, p1);

        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, point2, p2);

        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T constraints[3];
        constraints[0] = (p1[0] - p2[0]);
        constraints[1] = (p1[1] - p2[1]);
        constraints[2] = (p1[2] - p2[2]);

        return 5.0 * (constraints[0] * constraints[0] + constraints[1] * constraints[1] + constraints[2] * constraints[2]);
    }

    template <typename T>
    bool operator()(const T *const rotation1,
                    const T *const translation1,
                    const T *const rotation2,
                    const T *const translation2,
                    T *residuals) const
    {

        T point1[3] = {
            T(static_cast<double>(this->point1.x)),
            T(static_cast<double>(this->point1.y)),
            T(static_cast<double>(this->point1.z))};
        T point2[3] = {
            T(static_cast<double>(this->point2.x)),
            T(static_cast<double>(this->point2.y)),
            T(static_cast<double>(this->point2.z))};

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, point1, p1);

        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, point2, p2);

        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T constraints[3];
        constraints[0] = (p1[0] - p2[0]);
        constraints[1] = (p1[1] - p2[1]);
        constraints[2] = (p1[2] - p2[2]);

        residuals[0] = 5.0 * (constraints[0] * constraints[0] + constraints[1] * constraints[1] + constraints[2] * constraints[2]);

        return true;
    }

    // std::vector<std::shared_ptr<parameter>> parameters;

    // template <typename T>
    // T operator()(const T *params) const
    // {
    //     assert(parameters.size() == 4);
    //     const auto rotation1 = params + parameters[0]->offset;
    //     const auto translation1 = params + parameters[1]->offset;
    //     const auto rotation2 = params + parameters[2]->offset;
    //     const auto translation2 = params + parameters[3]->offset;

    //     T residual(0);
    //     this->operator()(rotation1, translation1, rotation2, translation2, &residual);
    //     return residual;
    // }
};

struct quat_rt_articulation_least_square_constraint
{
    glm::vec3 point1;
    glm::vec3 point2;
    double weight;

    quat_rt_articulation_least_square_constraint(glm::vec3 point1, glm::vec3 point2, double weight = 5.0)
        : point1(point1), point2(point2), weight(weight)
    {
    }

    template <typename T>
    inline T operator()(const T *const rotation1,
                        const T *const translation1,
                        const T *const rotation2,
                        const T *const translation2) const
    {
        quat_t<T> r_quat1(rotation1[0], rotation1[1], rotation1[2], rotation1[3]);
        quat_t<T> r_quat2(rotation2[0], rotation2[1], rotation2[2], rotation2[3]);
        vec3_t<T> t_vec1(translation1[0], translation1[1], translation1[2]);
        vec3_t<T> t_vec2(translation2[0], translation2[1], translation2[2]);

        vec3_t<T> point1_vec(T(point1.x), T(point1.y), T(point1.z));
        vec3_t<T> point2_vec(T(point2.x), T(point2.y), T(point2.z));

        const auto p1 = rotate(r_quat1, point1_vec) + t_vec1;
        const auto p2 = rotate(r_quat2, point2_vec) + t_vec2;

        const auto res_vec = p1 - p2;

        return weight * l2norm(res_vec);
    }
};

struct twisted_rt_articulation_least_square_constraint
{
    glm::vec3 point1;
    glm::vec3 point2;
    double twist_weight1;
    double twist_weight2;
    line3 twist_axis1;
    line3 twist_axis2;

    twisted_rt_articulation_least_square_constraint(glm::vec3 point1, glm::vec3 point2, double twist_weight1, double twist_weight2, const line3 &twist_axis1, const line3 &twist_axis2)
        : point1(point1), point2(point2), twist_weight1(twist_weight1), twist_weight2(twist_weight2), twist_axis1(twist_axis1), twist_axis2(twist_axis2)
    {
    }

    template <typename T>
    inline T operator()(const T *const rotation1,
                        const T *const translation1,
                        const T *const twist_angle1,
                        const T *const rotation2,
                        const T *const translation2,
                        const T *const twist_angle2) const
    {
        T point1[3] = {
            T(static_cast<double>(this->point1.x - twist_axis1.origin.x)),
            T(static_cast<double>(this->point1.y - twist_axis1.origin.y)),
            T(static_cast<double>(this->point1.z - twist_axis1.origin.z))};
        T point2[3] = {
            T(static_cast<double>(this->point2.x - twist_axis2.origin.x)),
            T(static_cast<double>(this->point2.y - twist_axis2.origin.y)),
            T(static_cast<double>(this->point2.z - twist_axis2.origin.z))};

        T twist_angle_axis1[3];
        twist_angle_axis1[0] = static_cast<double>(twist_axis1.direction.x * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[1] = static_cast<double>(twist_axis1.direction.y * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[2] = static_cast<double>(twist_axis1.direction.z * twist_weight1) * twist_angle1[0];

        T twist_angle_axis2[3];
        twist_angle_axis2[0] = static_cast<double>(twist_axis2.direction.x * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[1] = static_cast<double>(twist_axis2.direction.y * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[2] = static_cast<double>(twist_axis2.direction.z * twist_weight2) * twist_angle2[0];

        T twisted_p1[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis1, point1, twisted_p1);

        twisted_p1[0] += static_cast<double>(twist_axis1.origin.x);
        twisted_p1[1] += static_cast<double>(twist_axis1.origin.y);
        twisted_p1[2] += static_cast<double>(twist_axis1.origin.z);

        T twisted_p2[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis2, point2, twisted_p2);

        twisted_p2[0] += static_cast<double>(twist_axis2.origin.x);
        twisted_p2[1] += static_cast<double>(twist_axis2.origin.y);
        twisted_p2[2] += static_cast<double>(twist_axis2.origin.z);

        // Transform articulation by parent
        T p1[3];
        ceres::AngleAxisRotatePoint(rotation1, twisted_p1, p1);

        p1[0] += translation1[0];
        p1[1] += translation1[1];
        p1[2] += translation1[2];

        // Transform articulation by child
        T p2[3];
        ceres::AngleAxisRotatePoint(rotation2, twisted_p2, p2);

        p2[0] += translation2[0];
        p2[1] += translation2[1];
        p2[2] += translation2[2];

        T constraints[3];
        constraints[0] = (p1[0] - p2[0]);
        constraints[1] = (p1[1] - p2[1]);
        constraints[2] = (p1[2] - p2[2]);

        return 5.0 * (constraints[0] * constraints[0] + constraints[1] * constraints[1] + constraints[2] * constraints[2]);
    }
};

struct quat_twisted_rt_articulation_least_square_constraint
{
    glm::vec3 point1;
    glm::vec3 point2;
    double twist_weight1;
    double twist_weight2;
    line3 twist_axis1;
    line3 twist_axis2;

    quat_twisted_rt_articulation_least_square_constraint(glm::vec3 point1, glm::vec3 point2, double twist_weight1, double twist_weight2, const line3 &twist_axis1, const line3 &twist_axis2)
        : point1(point1), point2(point2), twist_weight1(twist_weight1), twist_weight2(twist_weight2), twist_axis1(twist_axis1), twist_axis2(twist_axis2)
    {
    }

    template <typename T>
    inline T operator()(const T *const rotation1,
                        const T *const translation1,
                        const T *const twist_angle1,
                        const T *const rotation2,
                        const T *const translation2,
                        const T *const twist_angle2) const
    {
        T point1[3] = {
            T(static_cast<double>(this->point1.x - twist_axis1.origin.x)),
            T(static_cast<double>(this->point1.y - twist_axis1.origin.y)),
            T(static_cast<double>(this->point1.z - twist_axis1.origin.z))};
        T point2[3] = {
            T(static_cast<double>(this->point2.x - twist_axis2.origin.x)),
            T(static_cast<double>(this->point2.y - twist_axis2.origin.y)),
            T(static_cast<double>(this->point2.z - twist_axis2.origin.z))};

        quat_t<T> r_quat1(rotation1[0], rotation1[1], rotation1[2], rotation1[3]);
        quat_t<T> r_quat2(rotation2[0], rotation2[1], rotation2[2], rotation2[3]);
        vec3_t<T> t_vec1(translation1[0], translation1[1], translation1[2]);
        vec3_t<T> t_vec2(translation2[0], translation2[1], translation2[2]);

        T twist_angle_axis1[3];
        twist_angle_axis1[0] = static_cast<double>(twist_axis1.direction.x * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[1] = static_cast<double>(twist_axis1.direction.y * twist_weight1) * twist_angle1[0];
        twist_angle_axis1[2] = static_cast<double>(twist_axis1.direction.z * twist_weight1) * twist_angle1[0];

        T twist_angle_axis2[3];
        twist_angle_axis2[0] = static_cast<double>(twist_axis2.direction.x * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[1] = static_cast<double>(twist_axis2.direction.y * twist_weight2) * twist_angle2[0];
        twist_angle_axis2[2] = static_cast<double>(twist_axis2.direction.z * twist_weight2) * twist_angle2[0];

        T twisted_p1[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis1, point1, twisted_p1);

        twisted_p1[0] += static_cast<double>(twist_axis1.origin.x);
        twisted_p1[1] += static_cast<double>(twist_axis1.origin.y);
        twisted_p1[2] += static_cast<double>(twist_axis1.origin.z);

        T twisted_p2[3];
        ceres::AngleAxisRotatePoint(twist_angle_axis2, point2, twisted_p2);

        twisted_p2[0] += static_cast<double>(twist_axis2.origin.x);
        twisted_p2[1] += static_cast<double>(twist_axis2.origin.y);
        twisted_p2[2] += static_cast<double>(twist_axis2.origin.z);

        vec3_t<T> point1_vec(twisted_p1[0], twisted_p1[1], twisted_p1[2]);
        vec3_t<T> point2_vec(twisted_p2[0], twisted_p2[1], twisted_p2[2]);

        const auto p1 = rotate(r_quat1, point1_vec) + t_vec1;
        const auto p2 = rotate(r_quat2, point2_vec) + t_vec2;

        const auto res_vec = p1 - p2;

        // return 5.0 * dot(res_vec, res_vec);
        return 5.0 * l2norm(res_vec);
    }
};
