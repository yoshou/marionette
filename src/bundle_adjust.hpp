#pragma once 

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <cstdio>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

// Read a Bundle Adjustment in the Large dataset.
class buneld_adjust_probrem
{
public:
    ~buneld_adjust_probrem()
    {
    }
    int num_observations() const { return num_observations_; }
    int num_cameras() const { return num_cameras_; }
    int num_points() const { return num_points_; }
    const double *observations() const { return observations_.data(); }
    double *mutable_cameras() { return parameters_.data(); }
    double *mutable_points() { return parameters_.data() + 9 * num_cameras_; }
    double *mutable_point(int i) { return parameters_.data() + 9 * num_cameras_ + 3 * i; }
    double *mutable_camera(int i) { return parameters_.data() + 9 * i; }
    double *mutable_camera_for_observation(int i)
    {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double *mutable_point_for_observation(int i)
    {
        return mutable_points() + point_index_[i] * 3;
    }
    int *point_index() { return point_index_.data(); }
    int *camera_index() { return camera_index_.data(); };

    bool load_file(std::istream &ifs)
    {
        if (!ifs)
        {
            return false;
        }

        ifs >> num_cameras_;
        ifs >> num_points_;
        ifs >> num_observations_;
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;

        point_index_.resize(num_observations_);
        camera_index_.resize(num_observations_);
        observations_.resize(2 * num_observations_);
        parameters_.resize(num_parameters_);

        for (int i = 0; i < num_observations_; i++)
        {
            ifs >> camera_index_[i];
            ifs >> point_index_[i];
            for (int j = 0; j < 2; j++)
            {
                ifs >> observations_[2 * i + j];
            }
        }
        for (int i = 0; i < num_parameters_; i++)
        {
            ifs >> parameters_[i];
        }
        return true;
    }

    bool load_file(const char *filename)
    {
        std::ifstream ifs;
        ifs.open(filename, std::ios::in);

        return load_file(ifs);
    }

    glm::mat4 get_camera_extrinsic(std::size_t i)
    {
        double quat[4];
        ceres::AngleAxisToQuaternion(mutable_camera(i), quat);
        double rot[9];
        ceres::QuaternionToRotation(quat, rot);

        double *trans = &mutable_camera(i)[3];

        glm::mat4 mat(1.0);
        for (size_t j = 0; j < 3; j++)
        {
            for (size_t k = 0; k < 3; k++)
            {
                mat[j][k] = rot[k * 3 + j];
            }
        }
        for (size_t k = 0; k < 3; k++)
        {
            mat[3][k] = trans[k];
        }

        return mat;
    }

private:
    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;

    std::vector<int> point_index_;
    std::vector<int> camera_index_;
    std::vector<double> observations_;
    std::vector<double> parameters_;
};
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct snavely_reprojection_error
{
    snavely_reprojection_error(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const
    {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        // Apply second and fourth order radial distortion.
        const T &l1 = camera[7];
        const T &l2 = camera[8];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + r2 * (l1 + l2 * r2);
        // Compute final projected point position.
        const T &focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        residuals[0] = xp - observed_x;
        residuals[1] = yp - observed_y;

        if (observed_x == -1)
        {
            return false;
        }
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction *create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<snavely_reprojection_error, 2, 9, 3>(
            new snavely_reprojection_error(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

static void add_noise(buneld_adjust_probrem &bal_problem, double mu = 0.0, double sigma = 0.1)
{
    std::mt19937 rand_src(12345);
    for (int i = 0; i < bal_problem.num_points(); i++)
    {
        std::normal_distribution<double> rand_dist(mu, sigma);
        bal_problem.mutable_point(i)[0] += rand_dist(rand_src);
        bal_problem.mutable_point(i)[1] += rand_dist(rand_src);
        bal_problem.mutable_point(i)[2] += rand_dist(rand_src);
    }
}
