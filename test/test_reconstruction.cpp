#include <gtest/gtest.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/hash.hpp>

#include <pcl/io/pcd_io.h>

#include <random>
#include <sstream>
#include <iostream>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "triangulation.hpp"
#include "bundle_adjust.hpp"
#include "correspondance.hpp"
#include "camera_info.hpp"

#if 0

struct scene_t {
    std::vector<camera_t> cameras;
};

static void get_plane_ring_camera_extrins(uint32_t num_cameras, glm::vec3 center, float radius, glm::vec3 lookat, std::vector<camera_extrin_t> &extrins)
{
    extrins.clear();

    glm::vec3 up_vec = glm::normalize(center);
    glm::vec3 radius_vec = glm::cross(up_vec, glm::vec3(1, 0, 0));

    for (uint32_t i = 0; i < num_cameras; i++)
    {
        glm::quat rot = glm::angleAxis(2.0f * glm::pi<float>() * i / num_cameras, up_vec);
        glm::vec3 pos = center + glm::rotate(rot, radius_vec) * radius;

        glm::mat4 view = glm::lookAt(pos, lookat, up_vec);
        extrins.push_back({pos, view});
    }
}

static camera_intrin_t get_typical_intrin()
{
    constexpr float fx = 380;
    constexpr float fy = 380;
    constexpr float cx = 640 / 2;
    constexpr float cy = 480 / 2;
    return {
        fx, fy, cx, cy
    };
}

static void generate_test_scene1(scene_t &scene)
{
    const auto ring_heights = {1.7, 2.1};

    for (auto ring_height : ring_heights)
    {
        std::vector<camera_extrin_t> extrins;

        get_plane_ring_camera_extrins(5, glm::vec3(0, ring_height, 0), 1.5, glm::vec3(0, 0, 0), extrins);

        for (size_t i = 0; i < extrins.size(); i++) {
            camera_t camera;
            camera.extrin = extrins[i];
            camera.intrin = get_typical_intrin();
            camera.width = 640;
            camera.height = 480;
            
            scene.cameras.push_back(camera);
        }
    }
}

template <typename Rand>
static void gen_random_cylinder_marker(Rand rand, std::size_t count, glm::vec3 scale, std::vector<glm::vec3>& markers)
{
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (std::size_t i = 0; i < count; i++) {
        const auto theta = 2.0 * M_PI * dist(rand);
        const auto r = sqrt(dist(rand));
        const auto h = dist(rand);

        markers.push_back(glm::vec3(r * cos(theta), r * sin(theta), h) * scale);
    }
}

template <typename Rand>
static void gen_random_cylinder_marker(Rand rand, std::size_t count, glm::vec3 scale, glm::vec3 voxel_grid_size, std::vector<glm::vec3> &markers)
{
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::unordered_set<glm::vec3> points;

    for (std::size_t i = 0; i < count; i++)
    {
        glm::vec3 voxel_pt;

        do
        {
            const auto theta = 2.0 * M_PI * dist(rand);
            const auto r = sqrt(dist(rand));
            const auto h = dist(rand);

            const auto pt = glm::vec3(r * cos(theta), r * sin(theta), h) * scale;

            voxel_pt = glm::round(pt / voxel_grid_size) * voxel_grid_size;
        } while (!points.insert(voxel_pt).second);

        markers.push_back(voxel_pt);
    }
}

static glm::mat4 estimate_extrinsic(const std::vector<glm::vec2> &pts1, const std::vector<glm::vec2> &pts2, const camera_intrin_t &i1, const camera_intrin_t &i2)
{
    std::vector<cv::Point2d> avail_pts1, avail_pts2;

    for (std::size_t i = 0; i < pts1.size(); i++)
    {
        const auto pt1 = pts1[i];
        const auto pt2 = pts2[i];

        if (pt1.x < 0 || pt1.y < 0 || pt2.x < 0 || pt2.y < 0)
        {
            continue;
        }

        avail_pts1.push_back(cv::Point2d(pt1.x, pt1.y));
        avail_pts2.push_back(cv::Point2d(pt2.x, pt2.y));
    }

    cv::Mat camera_mat1, camera_mat2;
    cv::Mat dist_coeffs1, dist_coeffs2;
    get_cv_intrinsic(i1, camera_mat1, dist_coeffs1);
    get_cv_intrinsic(i2, camera_mat2, dist_coeffs2);

    std::vector<cv::Point2d> undistort_pts1, undistort_pts2;
    cv::undistortPoints(avail_pts1, undistort_pts1, camera_mat1, dist_coeffs1);
    cv::undistortPoints(avail_pts2, undistort_pts2, camera_mat2, dist_coeffs2);

    const auto focal = 1.0;
    const cv::Point2d pp(0.0, 0.0);

    cv::Mat E, R, t, mask;
    E = cv::findEssentialMat(undistort_pts1, undistort_pts2, focal, pp, cv::RANSAC, 0.999, 0.003, mask);
    cv::recoverPose(E, undistort_pts1, undistort_pts2, R, t, focal, pp, mask);

    cv::Mat F = cv::findFundamentalMat(avail_pts1, avail_pts2, cv::RANSAC);
    cv::Mat F2 = camera_mat2.t().inv() * E * camera_mat1.inv();

    std::cout << F << std::endl;

    glm::mat4 result(1.0f);

    for (std::size_t i = 0; i < 3; i++)
    {
        for (std::size_t j = 0; j < 3; j++)
        {
            result[i][j] = static_cast<float>(R.at<double>(j, i));
        }
    }
    result[3][0] = static_cast<float>(t.at<double>(0));
    result[3][1] = static_cast<float>(t.at<double>(1));
    result[3][2] = static_cast<float>(t.at<double>(2));

    return result;
}

static void triangulate_markers(const std::vector<std::vector<glm::vec2>> &pts, const std::vector<camera_t> &cameras, const std::vector<glm::mat4> &poses, std::vector<glm::vec3> &result, std::vector<uint8_t> &mask)
{
    if (pts.size() == 0)
    {
        return;
    }

    const auto num_points = pts[0].size();
    const auto num_cameras = cameras.size();

    for (size_t i = 0; i < num_points; i++)
    {
        bool triangulated = false;
        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto c1 = c;
            const auto c2 = (c + 1) % num_cameras;

            const auto pt1 = pts[c1][i];
            const auto pt2 = pts[c2][i];

            const auto &camera1 = cameras[c1];
            const auto &camera2 = cameras[c2];

            if (pt1.x >= 0 && pt2.x >= 0)
            {
                cv::Mat proj1 = glm_to_cv_mat3x4(poses[c1]);
                cv::Mat proj2 = glm_to_cv_mat3x4(poses[c2]);

                std::vector<cv::Point2d> pts1, pts2;
                pts1.push_back(cv::Point2d(pt1.x, pt1.y));
                pts2.push_back(cv::Point2d(pt2.x, pt2.y));

                cv::Mat camera_mat1, camera_mat2;
                cv::Mat dist_coeffs1, dist_coeffs2;
                get_cv_intrinsic(camera1.intrin, camera_mat1, dist_coeffs1);
                get_cv_intrinsic(camera2.intrin, camera_mat2, dist_coeffs2);

                std::vector<cv::Point2d> undistort_pts1, undistort_pts2;
                cv::undistortPoints(pts1, undistort_pts1, camera_mat1, dist_coeffs1);
                cv::undistortPoints(pts2, undistort_pts2, camera_mat2, dist_coeffs2);

                cv::Mat output;
                cv::triangulatePoints(proj1, proj2, undistort_pts1, undistort_pts2, output);
                const auto w = output.at<double>(0, 3);

                result.push_back(glm::vec3(output.at<double>(0, 0) / w, output.at<double>(0, 1) / w, output.at<double>(0, 2) / w));

                triangulated = true;
                break;
            }
        }

        mask.push_back(triangulated);
        if (!triangulated)
        {
            result.push_back(glm::vec3(0.0f));
        }
    }
}

static glm::mat4 estimate_extrinsic(const std::vector<glm::vec2> &pts1, const std::vector<glm::vec2> &pts2, const std::vector<glm::vec2> &pts3, const camera_intrin_t &i1, const camera_intrin_t &i2, const camera_intrin_t &i3, const glm::mat4 &extrin1, const glm::mat4 &extrin2)
{
    std::vector<cv::Point2d> avail_pts1, avail_pts2, avail_pts3;

    for (std::size_t i = 0; i < pts1.size(); i++)
    {
        const auto pt1 = pts1[i];
        const auto pt2 = pts2[i];
        const auto pt3 = pts3[i];

        if (pt1.x < 0 || pt1.y < 0 || pt2.x < 0 || pt2.y < 0 || pt3.x < 0 || pt3.y < 0)
        {
            continue;
        }

        avail_pts1.push_back(cv::Point2d(pt1.x, pt1.y));
        avail_pts2.push_back(cv::Point2d(pt2.x, pt2.y));
        avail_pts3.push_back(cv::Point2d(pt3.x, pt3.y));
    }

    cv::Mat camera_mat1, camera_mat2, camera_mat3;
    cv::Mat dist_coeffs1, dist_coeffs2, dist_coeffs3;
    get_cv_intrinsic(i1, camera_mat1, dist_coeffs1);
    get_cv_intrinsic(i2, camera_mat2, dist_coeffs2);
    get_cv_intrinsic(i3, camera_mat3, dist_coeffs3);

    std::vector<cv::Point2d> undistort_pts1, undistort_pts2, undistort_pts3;
    cv::undistortPoints(avail_pts1, undistort_pts1, camera_mat1, dist_coeffs1);
    cv::undistortPoints(avail_pts2, undistort_pts2, camera_mat2, dist_coeffs2);
    cv::undistortPoints(avail_pts3, undistort_pts3, camera_mat3, dist_coeffs3);

    cv::Mat proj1 = glm_to_cv_mat3x4(extrin1);
    cv::Mat proj2 = glm_to_cv_mat3x4(extrin2);

    cv::Mat output;
    cv::triangulatePoints(proj1, proj2, undistort_pts1, undistort_pts2, output);

    std::vector<cv::Point3d> points;
    for (size_t i = 0; i < static_cast<size_t>(output.cols); i++)
    {
        points.push_back(cv::Point3d(
            output.at<double>(0, i) / output.at<double>(3, i),
            output.at<double>(1, i) / output.at<double>(3, i),
            output.at<double>(2, i) / output.at<double>(3, i)));
    }

    cv::Mat R, r, t;
    cv::solvePnP(points, undistort_pts3, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(5, 1, CV_64F), r, t);
    cv::Rodrigues(r, R);

    glm::mat4 result(1.0f);

    for (std::size_t i = 0; i < 3; i++)
    {
        for (std::size_t j = 0; j < 3; j++)
        {
            result[i][j] = static_cast<float>(R.at<double>(j, i));
        }
    }
    result[3][0] = static_cast<float>(t.at<double>(0));
    result[3][1] = static_cast<float>(t.at<double>(1));
    result[3][2] = static_cast<float>(t.at<double>(2));

    return result;
}

static void estimate_camera_poses(const std::vector<std::vector<glm::vec2>> &pts, const std::vector<camera_t> &cameras,
                                  std::vector<glm::mat4> &poses)
{
    const auto num_cameras = cameras.size();

    const auto estimated_base_pose = estimate_extrinsic(pts[0], pts[1], cameras[0].intrin, cameras[1].intrin);

    std::vector<triple<size_t>> camera_pairs;
    for (size_t i = 0; i < num_cameras; i++)
    {
        camera_pairs.push_back(std::make_tuple(i, (i + 1) % num_cameras, (i + 2) % num_cameras));
    }

    poses.resize(num_cameras);

    poses[0] = glm::mat4(1.0f);
    poses[1] = estimated_base_pose;

    for (size_t i = 0; i < camera_pairs.size() - 2; i++)
    {
        size_t c0, c1, c2;
        std::tie(c0, c1, c2) = camera_pairs[i];

        const auto estimated_pose = estimate_extrinsic(pts[c0], pts[c1], pts[c2], cameras[c0].intrin, cameras[c1].intrin, cameras[c2].intrin, poses[c0], poses[c1]);

        poses[c2] = estimated_pose;
    }
}

static void adjust_pose(const std::vector<glm::mat4> &poses, const std::vector<std::vector<glm::vec2>> &pts, const std::vector<glm::vec3> &markers, const std::vector<uint8_t> &mask, const std::vector<camera_t> &cameras, std::vector<glm::mat4> &adjusted_poses)
{
    const auto num_cameras = cameras.size();

    std::vector<glm::vec3> points;
    std::vector<glm::vec2> observations;
    std::vector<int> point_index;
    std::vector<int> camera_index;

    for (size_t m = 0; m < markers.size(); m++)
    {
        if (!mask[m])
        {
            continue;
        }

        const auto marker = markers[m];

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto camera = cameras[c];

            const auto observed_pt = pts[c][m];

            if (observed_pt.x < 0)
            {
                continue;
            }

            std::vector<cv::Point2d> pts;
            pts.push_back(cv::Point2d(observed_pt.x, observed_pt.y));

            cv::Mat camera_mat;
            cv::Mat dist_coeffs;
            get_cv_intrinsic(camera.intrin, camera_mat, dist_coeffs);

            std::vector<cv::Point2d> undistort_pts;
            cv::undistortPoints(pts, undistort_pts, camera_mat, dist_coeffs);

            observations.push_back(glm::vec2(undistort_pts[0].x, undistort_pts[0].y));
            camera_index.push_back(static_cast<int>(c));
            point_index.push_back(static_cast<int>(points.size()));
        }

        points.push_back(marker);
    }

    std::stringstream ss;
    ss << num_cameras << " " << points.size() << " " << observations.size() << std::endl;

    for (size_t i = 0; i < observations.size(); i++)
    {
        ss << camera_index[i] << " " << point_index[i] << " " << observations[i].x << " " << observations[i].y << std::endl;
    }

    for (size_t i = 0; i < cameras.size(); i++)
    {
        glm::quat rotation = glm::quat_cast(poses[i]);
        glm::vec3 axis_angle = glm::normalize(glm::axis(rotation)) * glm::angle(rotation);
        ss << axis_angle.x << std::endl;
        ss << axis_angle.y << std::endl;
        ss << axis_angle.z << std::endl;

        glm::vec3 translation(poses[i][3]);
        ss << translation.x << std::endl;
        ss << translation.y << std::endl;
        ss << translation.z << std::endl;

        glm::vec3 coeffs(1, 0, 0);
        ss << coeffs.x << std::endl;
        ss << coeffs.y << std::endl;
        ss << coeffs.z << std::endl;
    }

    for (size_t i = 0; i < points.size(); i++)
    {
        ss << points[i].x << std::endl;
        ss << points[i].y << std::endl;
        ss << points[i].z << std::endl;
    }

    adjusted_poses.clear();
    {
        buneld_adjust_probrem bal_problem;
        ASSERT_TRUE(bal_problem.load_file(ss));

        const double *observations = bal_problem.observations();
        ceres::Problem problem;
        for (int i = 0; i < bal_problem.num_observations(); ++i)
        {
            ceres::CostFunction *cost_function =
                snavely_reprojection_error::create(observations[2 * i + 0],
                                                   observations[2 * i + 1]);
            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     bal_problem.mutable_camera_for_observation(i),
                                     bal_problem.mutable_point_for_observation(i));
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 1000;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        for (size_t i = 0; i < num_cameras; i++)
        {
            auto mat = bal_problem.get_camera_extrinsic(i);
            adjusted_poses.push_back(mat);
        }
    }
}

static void find_correspondance(const std::vector<std::vector<glm::vec2>> &pts, const std::vector<camera_t> &cameras, const std::vector<glm::mat4> &poses, const std::vector<std::vector<size_t>>& index, double thresh)
{
    std::size_t c1 = 0;

    size_t positive = 0;
    size_t negative = 0;

    for (std::size_t i = 0; i < pts[c1].size(); i++)
    {
        const auto pt1 = pts[c1][i];

        if (pt1.x < 0 || pt1.y < 0)
        {
            continue;
        }

        for (std::size_t c2 = c1 + 1; c2 < cameras.size(); c2++)
        {
            auto f_m = glm_to_cv_mat3x3(
                calculate_fundametal_matrix(cameras[c1], cameras[c2]));

            std::vector<cv::Point2f> points;
            points.push_back(cv::Point2f(pt1.x, pt1.y));

            std::vector<cv::Vec3f> lines;
            cv::computeCorrespondEpilines(points, 1, f_m, lines);
            auto line = lines[0];

            auto min_dist = std::numeric_limits<float>::max();
            size_t min_idx = 0;

            std::stringstream ss;
            ss << "-------------" << std::endl;

            for (std::size_t j = 0; j < pts[c2].size(); j++)
            {
                const auto pt2 = pts[c2][j];

                if (pt2.x < 0 || pt2.y < 0)
                {
                    continue;
                }

                const auto dist = distance_line_point(glm::vec3(line[0], line[1], line[2]), pt2);
                if (dist < thresh)
                {
                    ss << "----" << std::endl;
                    const auto marker = triangulate(pt1, pt2, cameras[c1], cameras[c2]);

                    float nearest_pt_dist_acc = 0;
                    size_t nearest_pt_dist_count = 0;
                    for (size_t c3 = 0; c3 < cameras.size(); c3++)
                    {
                        if (c1 == c3 || c2 == c3)
                        {
                            continue;
                        }

                        const auto pt = project(cameras[c3], marker);

                        float nearest_pt_dist;
                        const auto nearest_pt_idx = find_nearest_point(pt, pts[c3], thresh, nearest_pt_dist);

                        if (nearest_pt_idx == pts[c3].size())
                        {
                            continue;
                        }

                        ss << nearest_pt_dist << std::endl;

                        nearest_pt_dist_acc += nearest_pt_dist;
                        nearest_pt_dist_count++;
                    }

                    nearest_pt_dist_acc += dist;
                    nearest_pt_dist_count += 1;

                    nearest_pt_dist_acc /= (nearest_pt_dist_count);
                    if (nearest_pt_dist_count <= 2)
                    {
                        nearest_pt_dist_acc = std::numeric_limits<float>::max();
                    }

                    ss << "nearest_pt_dist_acc: " << nearest_pt_dist_acc << std::endl;
                    ss << "nearest_pt_dist_count: " << nearest_pt_dist_count << std::endl;
                    ss << "idx: " << j << std::endl;

                    if (nearest_pt_dist_acc < min_dist)
                    {
                        min_dist = nearest_pt_dist_acc;
                        min_idx = j;
                    }
                }
            }

            if (min_dist > thresh)
            {
                continue;
            }

            ss << "min dist: " << min_dist << std::endl;
            ss << "min idx: " << min_idx << std::endl;

            ss << "answer idx: " << index[c2][i] << std::endl;

            if (min_idx == index[c2][i])
            {
                positive++;
            }
            else
            {
                //std::cout << ss.str();
                negative++;
            }
        }
    }

    std::cout << "accuracy rate: " << (double)positive / (positive + negative) << std::endl;
}

TEST(IrReconstruction, OpenCVTriangulationTest)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    for (size_t m = 0; m < markers.size(); m++)
    {
        for (size_t c = 0; c < scene.cameras.size(); c++)
        {
            const auto camera1 = scene.cameras[c];
            const auto camera2 = scene.cameras[(c + 1) % scene.cameras.size()];
            const auto marker = markers[m];

            const auto pt1 = project(camera1, marker);
            const auto pt2 = project(camera2, marker);

            if (pt1.x < 0 || pt2.x < 0)
            {
                continue;
            }

            cv::Mat proj1 = glm_to_cv_mat3x4(camera1.extrin.rotation);
            cv::Mat proj2 = glm_to_cv_mat3x4(camera2.extrin.rotation);

            std::vector<cv::Point2d> pts1, pts2;
            pts1.push_back(cv::Point2d(pt1.x, pt1.y));
            pts2.push_back(cv::Point2d(pt2.x, pt2.y));

            cv::Mat camera_mat1, camera_mat2;
            cv::Mat dist_coeffs1, dist_coeffs2;
            get_cv_intrinsic(camera1.intrin, camera_mat1, dist_coeffs1);
            get_cv_intrinsic(camera1.intrin, camera_mat2, dist_coeffs2);

            std::vector<cv::Point2d> undistort_pts1, undistort_pts2;
            cv::undistortPoints(pts1, undistort_pts1, camera_mat1, dist_coeffs1);
            cv::undistortPoints(pts2, undistort_pts2, camera_mat2, dist_coeffs2);

            cv::Mat output;
            cv::triangulatePoints(proj1, proj2, undistort_pts1, undistort_pts2, output);
            const auto w = output.at<double>(0, 3);

            glm::vec3 result(output.at<double>(0, 0) / w, output.at<double>(0, 1) / w, output.at<double>(0, 2) / w);
            glm::vec3 expect = marker;

            ASSERT_TRUE(glm::distance(result, expect) < 1e-3f);
        }
    }
}

TEST(IrReconstruction, TriangulationTest)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    for (size_t m = 0; m < markers.size(); m++)
    {
        for (size_t c = 0; c < scene.cameras.size(); c++)
        {
            const auto camera1 = scene.cameras[c];
            const auto camera2 = scene.cameras[(c + 1) % scene.cameras.size()];
            const auto marker = markers[m];

            const auto pt1 = project(camera1, marker);
            const auto pt2 = project(camera2, marker);

            if (pt1.x < 0 || pt2.x < 0)
            {
                continue;
            }

            glm::mat4 camera_mat1(camera1.intrin.get_matrix());
            glm::mat4 camera_mat2(camera2.intrin.get_matrix());

            glm::vec2 s;
            glm::vec3 result = triangulation(camera_mat1 * camera1.extrin.rotation, camera_mat2 * camera2.extrin.rotation,
                                             pt1,
                                             pt2, s);

            glm::vec3 expect = marker;

            ASSERT_TRUE(glm::distance(result, expect) < 1e-3f);
        }
    }
}

TEST(IrReconstruction, BACalibrationTest)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    const auto num_cameras = scene.cameras.size();

    std::vector<glm::vec3> points;
    std::vector<glm::vec2> observations;
    std::vector<int> point_index;
    std::vector<int> camera_index;

    for (size_t m = 0; m < markers.size(); m++)
    {
        const auto marker = markers[m];

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto camera = scene.cameras[c];

            const auto pt = project(camera, marker);

            if (pt.x < 0)
            {
                continue;
            }

            auto pixel_err = 2.0f;

            std::uniform_real_distribution<> x_dist(-pixel_err, pixel_err);
            std::uniform_real_distribution<> y_dist(-pixel_err, pixel_err);

            glm::vec2 observed_pt(pt.x + x_dist(rand), pt.y + y_dist(rand));

            std::vector<cv::Point2d> pts;
            pts.push_back(cv::Point2d(observed_pt.x, observed_pt.y));

            cv::Mat camera_mat;
            cv::Mat dist_coeffs;
            get_cv_intrinsic(camera.intrin, camera_mat, dist_coeffs);

            std::vector<cv::Point2d> undistort_pts;
            cv::undistortPoints(pts, undistort_pts, camera_mat, dist_coeffs);

            observations.push_back(glm::vec2(undistort_pts[0].x, undistort_pts[0].y));
            camera_index.push_back(static_cast<int>(c));
            point_index.push_back(static_cast<int>(m));
        }

        points.push_back(marker);
    }

    std::stringstream ss;
    ss << num_cameras << " " << points.size() << " " << observations.size() << std::endl;

    for (size_t i = 0; i < observations.size(); i++)
    {
        ss << camera_index[i] << " " << point_index[i] << " " << observations[i].x << " " << observations[i].y << std::endl;
    }

    for (size_t i = 0; i < scene.cameras.size(); i++) {
        glm::quat rotation = glm::quat_cast(scene.cameras[i].extrin.rotation);
        glm::vec3 axis_angle = glm::normalize(glm::axis(rotation)) * glm::angle(rotation);
        ss << axis_angle.x << std::endl;
        ss << axis_angle.y << std::endl;
        ss << axis_angle.z << std::endl;

        glm::vec3 translation(scene.cameras[i].extrin.rotation[3]);
        ss << translation.x << std::endl;
        ss << translation.y << std::endl;
        ss << translation.z << std::endl;

        glm::vec3 coeffs(1, 0, 0);
        ss << coeffs.x << std::endl;
        ss << coeffs.y << std::endl;
        ss << coeffs.z << std::endl;
    }

    for (size_t i = 0; i < points.size(); i++)
    {
        ss << points[i].x << std::endl;
        ss << points[i].y << std::endl;
        ss << points[i].z << std::endl;
    }

    {
        buneld_adjust_probrem bal_problem;
        ASSERT_TRUE(bal_problem.load_file(ss));

        const double *observations = bal_problem.observations();
        ceres::Problem problem;
        for (int i = 0; i < bal_problem.num_observations(); ++i)
        {
            ceres::CostFunction *cost_function =
                snavely_reprojection_error::create(observations[2 * i + 0],
                                                   observations[2 * i + 1]);
            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     bal_problem.mutable_camera_for_observation(i),
                                     bal_problem.mutable_point_for_observation(i));
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        ASSERT_EQ(summary.termination_type, ceres::CONVERGENCE);
    }
}

TEST(IrReconstruction, EpipolarLineTest)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    const auto num_cameras = scene.cameras.size();
    
    std::vector<std::vector<glm::vec2>> observed_pts(num_cameras);

    for (size_t m = 0; m < markers.size(); m++)
    {
        const auto marker = markers[m];

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto camera = scene.cameras[c];

            const auto pt = project(camera, marker);

            auto pixel_err = (pt.x < 0 || pt.y < 0) ? 0.0f : 2.0f;

            std::uniform_real_distribution<> x_dist(-pixel_err, pixel_err);
            std::uniform_real_distribution<> y_dist(-pixel_err, pixel_err);

            glm::vec2 observed_pt(pt.x + x_dist(rand), pt.y + y_dist(rand));

            observed_pts[c].push_back(observed_pt);
        }
    }

    ASSERT_GT(observed_pts.size(), 0u);

    size_t rows = observed_pts[0].size();
    size_t cols = observed_pts.size() * 2;

    cv::Mat m;
    m = cv::Mat::zeros(rows, cols, CV_64F);
    for (size_t row = 0; row < rows; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            m.at<double>(row, col) = observed_pts[col / 2][row][col % 2];
        }
    }

    std::vector<triple<size_t>> camera_pairs;
    for (size_t i = 0; i < num_cameras; i++)
    {
        camera_pairs.push_back(std::make_tuple(i, (i + 1) % num_cameras, (i + 2) % num_cameras));
    }
    
    const auto fund_mats = estimate_triple_fundamental_mat(m, camera_pairs);

    for (auto p : camera_pairs)
    {
        auto f_mats = fund_mats.at(p);

        size_t camera1 = std::get<0>(p);
        size_t camera2 = std::get<1>(p);
        size_t camera3 = std::get<2>(p);

        auto f_m1 = std::get<0>(f_mats);
        auto f_m2 = std::get<1>(f_mats);
        auto f_m3 = std::get<2>(f_mats);

        std::vector<cv::Point2f> points1, points2, points3;
        for (auto &pt : observed_pts[camera1])
        {
            points1.push_back(cv::Point2f(pt.x, pt.y));
        }
        for (auto &pt : observed_pts[camera2])
        {
            points2.push_back(cv::Point2f(pt.x, pt.y));
        }
        for (auto &pt : observed_pts[camera3])
        {
            points3.push_back(cv::Point2f(pt.x, pt.y));
        }

        std::vector<cv::Vec3f> lines1, lines2, lines3, lines4, lines5, lines6;
        if (points1.size() > 0)
        {
            cv::computeCorrespondEpilines(points1, 1, f_m1, lines1);
            cv::computeCorrespondEpilines(points1, 2, f_m3, lines6);
        }
        if (points2.size() > 0)
        {
            cv::computeCorrespondEpilines(points2, 1, f_m2, lines2);
            cv::computeCorrespondEpilines(points2, 2, f_m1, lines4);
        }
        if (points3.size() > 0)
        {
            cv::computeCorrespondEpilines(points3, 1, f_m3, lines3);
            cv::computeCorrespondEpilines(points3, 2, f_m2, lines5);
        }

        ASSERT_EQ(lines1.size(), points2.size());

        auto var = 0.0;
        auto n = 0;
        for (size_t i = 0; i < lines1.size(); i++)
        {
            auto line = lines1[i];
            auto point = points2[i];

            if (points1[i].x < 0 || points1[i].y < 0)
            {
                continue;
            }
            if (point.x < 0 || point.y < 0)
            {
                continue;
            }

            const auto dist = distance_line_point(glm::vec3(line[0], line[1], line[2]), glm::vec2(point.x, point.y));
            var += dist;
            n++;
            ASSERT_LT(dist, 10.0);
        }
        std::cout << var / n << std::endl;
    }
}

TEST(IrReconstruction, FundamentalMatrixTest)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    const auto num_cameras = scene.cameras.size();

    std::vector<std::vector<glm::vec2>> observed_pts(num_cameras);

    for (size_t m = 0; m < markers.size(); m++)
    {
        const auto marker = markers[m];

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto camera = scene.cameras[c];

            const auto observed_pt = project(camera, marker);

            observed_pts[c].push_back(observed_pt);
        }
    }

    ASSERT_GT(observed_pts.size(), 0u);

    std::vector<std::pair<size_t, size_t>> camera_pairs;
    for (size_t i = 0; i < num_cameras; i++)
    {
        camera_pairs.push_back(std::make_pair(i, (i + 1) % num_cameras));
    }

    for (auto p : camera_pairs)
    {
        size_t camera1 = std::get<0>(p);
        size_t camera2 = std::get<1>(p);

        auto f_m1 = glm_to_cv_mat3x3(
            calculate_fundametal_matrix(scene.cameras[camera1], scene.cameras[camera2]));

        std::vector<cv::Point2f> points1, points2;
        for (auto &pt : observed_pts[camera1])
        {
            points1.push_back(cv::Point2f(pt.x, pt.y));
        }
        for (auto &pt : observed_pts[camera2])
        {
            points2.push_back(cv::Point2f(pt.x, pt.y));
        }

        std::vector<cv::Vec3f> lines1;
        if (points1.size() > 0)
        {
            cv::computeCorrespondEpilines(points1, 1, f_m1, lines1);
        }

        ASSERT_EQ(lines1.size(), points2.size());

        for (size_t i = 0; i < lines1.size(); i++)
        {
            auto line = lines1[i];
            auto point = points2[i];

            if (points1[i].x < 0 || points1[i].y < 0)
            {
                continue;
            }
            if (point.x < 0 || point.y < 0)
            {
                continue;
            }

            const auto dist = distance_line_point(glm::vec3(line[0], line[1], line[2]), glm::vec2(point.x, point.y));
            ASSERT_LT(dist, 1e-3);
        }
    }
}

TEST(IrReconstruction, EpipolarLineTest2)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    const auto num_cameras = scene.cameras.size();

    std::vector<std::vector<glm::vec2>> pts(num_cameras);
    for (size_t m = 0; m < markers.size(); m++)
    {
        const auto marker = markers[m];

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto camera = scene.cameras[c];

            const auto pt = project(camera, marker);

            auto pixel_err = (pt.x < 0) ? 0.0f : 2.0f;

            std::uniform_real_distribution<> x_dist(-pixel_err, pixel_err);
            std::uniform_real_distribution<> y_dist(-pixel_err, pixel_err);

            glm::vec2 observed_pt(pt.x + x_dist(rand), pt.y + y_dist(rand));

            pts[c].push_back(observed_pt);
        }
    }

    std::vector<glm::mat4> poses(num_cameras);
    estimate_camera_poses(pts, scene.cameras, poses);

    for (size_t i = 1; i < num_cameras; i++)
    {
        const auto estimated_pose = poses[i];
        const auto pose = scene.cameras[i].extrin.rotation * glm::inverse(scene.cameras[0].extrin.rotation);

        ASSERT_LT(
            1.0f - std::abs(glm::dot(glm::normalize(glm::vec3(pose[3])), glm::normalize(glm::vec3(estimated_pose[3])))),
            1e-2f);
    }

    std::vector<glm::vec3> estimated_markers;
    std::vector<uint8_t> mask;
    triangulate_markers(pts, scene.cameras, poses, estimated_markers, mask);

    std::vector<glm::mat4> adjusted_poses;
    adjust_pose(poses, pts, estimated_markers, mask, scene.cameras, adjusted_poses);

    for (size_t i = 1; i < num_cameras; i++)
    {
        const auto adjusted_pose = adjusted_poses[i];
        const auto pose = scene.cameras[i].extrin.rotation * glm::inverse(scene.cameras[0].extrin.rotation);

        ASSERT_LT(
            1.0f - std::abs(glm::dot(glm::normalize(glm::vec3(pose[3])), glm::normalize(glm::vec3(adjusted_pose[3])))),
            1e-2f);
    }

    {
        std::vector<std::pair<size_t, size_t>> camera_pairs;
        for (size_t i = 0; i < num_cameras; i++)
        {
            camera_pairs.push_back(std::make_pair(i, (i + 1) % num_cameras));
        }

        for (auto p : camera_pairs)
        {
            size_t camera1 = std::get<0>(p);
            size_t camera2 = std::get<1>(p);

            auto f_m1 = glm_to_cv_mat3x3(
                calculate_fundametal_matrix(scene.cameras[camera1].intrin.get_matrix(), scene.cameras[camera2].intrin.get_matrix(),
                                            adjusted_poses[camera1], adjusted_poses[camera2]));

            std::vector<cv::Point2f> points1, points2;
            for (auto &pt : pts[camera1])
            {
                points1.push_back(cv::Point2f(pt.x, pt.y));
            }
            for (auto &pt : pts[camera2])
            {
                points2.push_back(cv::Point2f(pt.x, pt.y));
            }

            std::vector<cv::Vec3f> lines1;
            if (points1.size() > 0)
            {
                cv::computeCorrespondEpilines(points1, 1, f_m1, lines1);
            }

            ASSERT_EQ(lines1.size(), points2.size());

            auto var = 0.0;
            auto n = 0;
            for (size_t i = 0; i < lines1.size(); i++)
            {
                auto line = lines1[i];
                auto point = points2[i];

                if (points1[i].x < 0 || points1[i].y < 0)
                {
                    continue;
                }
                if (point.x < 0 || point.y < 0)
                {
                    continue;
                }

                const auto dist = distance_line_point(glm::vec3(line[0], line[1], line[2]), glm::vec2(point.x, point.y));
                var += dist;
                n++;
                ASSERT_LT(dist, 10);
            }
            std::cout << var / n << std::endl;
        }
    }
}

TEST(IrReconstruction, EpipolarLineTest3)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    const auto num_cameras = scene.cameras.size();

    std::vector<std::vector<glm::vec2>> pts(num_cameras);
    for (size_t m = 0; m < markers.size(); m++)
    {
        const auto marker = markers[m];

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto camera = scene.cameras[c];

            const auto pt = project(camera, marker);

            auto pixel_err = (pt.x < 0) ? 0.0f : 2.0f;

            std::uniform_real_distribution<> x_dist(-pixel_err, pixel_err);
            std::uniform_real_distribution<> y_dist(-pixel_err, pixel_err);

            glm::vec2 observed_pt(pt.x + x_dist(rand), pt.y + y_dist(rand));

            pts[c].push_back(observed_pt);
        }
    }

    std::vector<glm::mat4> poses(num_cameras);
    estimate_camera_poses(pts, scene.cameras, poses);

    for (size_t i = 1; i < num_cameras; i++)
    {
        const auto estimated_pose = poses[i];
        const auto pose = scene.cameras[i].extrin.rotation * glm::inverse(scene.cameras[0].extrin.rotation);

        ASSERT_LT(
            1.0f - std::abs(glm::dot(glm::normalize(glm::vec3(pose[3])), glm::normalize(glm::vec3(estimated_pose[3])))),
            1e-2f);
    }

    std::vector<glm::vec3> estimated_markers;
    std::vector<uint8_t> mask;
    triangulate_markers(pts, scene.cameras, poses, estimated_markers, mask);

    std::vector<glm::mat4> adjusted_poses;
    adjust_pose(poses, pts, estimated_markers, mask, scene.cameras, adjusted_poses);

    for (size_t i = 1; i < num_cameras; i++)
    {
        const auto adjusted_pose = adjusted_poses[i];
        const auto pose = scene.cameras[i].extrin.rotation * glm::inverse(scene.cameras[0].extrin.rotation);

        ASSERT_LT(
            1.0f - std::abs(glm::dot(glm::normalize(glm::vec3(pose[3])), glm::normalize(glm::vec3(adjusted_pose[3])))),
            1e-2f);
    }

    {
        std::vector<std::pair<size_t, size_t>> camera_pairs;
        for (size_t i = 0; i < num_cameras; i++)
        {
            camera_pairs.push_back(std::make_pair(i, (i + 1) % num_cameras));
        }

        for (auto p : camera_pairs)
        {
            size_t camera1 = std::get<0>(p);
            size_t camera2 = std::get<1>(p);

            auto f_m1 = calculate_fundametal_matrix(scene.cameras[camera1].intrin.get_matrix(), scene.cameras[camera2].intrin.get_matrix(),
                                            adjusted_poses[camera1], adjusted_poses[camera2]);

            const auto &pts1 = pts[camera1];
            const auto &pts2 = pts[camera2];

            std::vector<glm::vec3> lines1;
            for (const auto &pt1 : pts1)
            {
                lines1.push_back(compute_correspond_epiline(f_m1, pt1));
            }

            ASSERT_EQ(lines1.size(), pts2.size());

            auto var = 0.0;
            auto n = 0;
            for (size_t i = 0; i < lines1.size(); i++)
            {
                auto line = lines1[i];
                const auto pt1 = pts1[i];
                const auto pt2 = pts2[i];

                if (pt1.x < 0 || pt1.y < 0)
                {
                    continue;
                }
                if (pt2.x < 0 || pt2.y < 0)
                {
                    continue;
                }

                const auto dist = distance_line_point(line, pt2);
                var += dist;
                n++;
                ASSERT_LT(dist, 10);
            }
            std::cout << var / n << std::endl;
        }
    }
}

TEST(IrReconstruction, FindCorrespondanceTest)
{
    scene_t scene;
    generate_test_scene1(scene);

    std::mt19937 rand;

    size_t num_markers = 500;

    std::vector<glm::vec3> markers;
    gen_random_cylinder_marker(rand, num_markers, glm::vec3(1.5f), markers);

    const auto num_cameras = scene.cameras.size();

    std::vector<std::vector<glm::vec2>> pts(num_cameras);
    for (size_t m = 0; m < markers.size(); m++)
    {
        const auto marker = markers[m];

        for (size_t c = 0; c < num_cameras; c++)
        {
            const auto camera = scene.cameras[c];

            const auto pt = project(camera, marker);

            auto pixel_err = (pt.x < 0) ? 0.0f : 2.0f;

            std::uniform_real_distribution<> x_dist(-pixel_err, pixel_err);
            std::uniform_real_distribution<> y_dist(-pixel_err, pixel_err);

            glm::vec2 observed_pt(pt.x + x_dist(rand), pt.y + y_dist(rand));

            pts[c].push_back(observed_pt);
        }
    }

    std::vector<glm::mat4> poses(num_cameras);
    estimate_camera_poses(pts, scene.cameras, poses);

    for (size_t i = 1; i < num_cameras; i++)
    {
        const auto estimated_pose = poses[i];
        const auto pose = scene.cameras[i].extrin.rotation * glm::inverse(scene.cameras[0].extrin.rotation);

        ASSERT_LT(
            1.0f - std::abs(glm::dot(glm::normalize(glm::vec3(pose[3])), glm::normalize(glm::vec3(estimated_pose[3])))),
            1e-2f);
    }

    std::vector<glm::vec3> estimated_markers;
    std::vector<uint8_t> mask;
    triangulate_markers(pts, scene.cameras, poses, estimated_markers, mask);

    std::vector<glm::mat4> adjusted_poses;
    adjust_pose(poses, pts, estimated_markers, mask, scene.cameras, adjusted_poses);

    for (size_t i = 1; i < num_cameras; i++)
    {
        const auto adjusted_pose = adjusted_poses[i];
        const auto pose = scene.cameras[i].extrin.rotation * glm::inverse(scene.cameras[0].extrin.rotation);

        ASSERT_LT(
            1.0f - std::abs(glm::dot(glm::normalize(glm::vec3(pose[3])), glm::normalize(glm::vec3(adjusted_pose[3])))),
            1e-2f);
    }

    {
        std::vector<std::pair<size_t, size_t>> camera_pairs;
        for (size_t i = 0; i < num_cameras; i++)
        {
            camera_pairs.push_back(std::make_pair(i, (i + 1) % num_cameras));
        }

        for (auto p : camera_pairs)
        {
            size_t camera1 = std::get<0>(p);
            size_t camera2 = std::get<1>(p);

            auto f_m1 = glm_to_cv_mat3x3(
                calculate_fundametal_matrix(scene.cameras[camera1].intrin.get_matrix(), scene.cameras[camera2].intrin.get_matrix(),
                                            adjusted_poses[camera1], adjusted_poses[camera2]));

            std::vector<cv::Point2f> points1, points2;
            for (auto &pt : pts[camera1])
            {
                points1.push_back(cv::Point2f(pt.x, pt.y));
            }
            for (auto &pt : pts[camera2])
            {
                points2.push_back(cv::Point2f(pt.x, pt.y));
            }

            std::vector<cv::Vec3f> lines1;
            if (points1.size() > 0)
            {
                cv::computeCorrespondEpilines(points1, 1, f_m1, lines1);
            }

            ASSERT_EQ(lines1.size(), points2.size());

            auto var = 0.0;
            auto n = 0;
            for (size_t i = 0; i < lines1.size(); i++)
            {
                auto line = lines1[i];
                auto point = points2[i];

                if (points1[i].x < 0 || points1[i].y < 0)
                {
                    continue;
                }
                if (point.x < 0 || point.y < 0)
                {
                    continue;
                }

                const auto dist = distance_line_point(glm::vec3(line[0], line[1], line[2]), glm::vec2(point.x, point.y));
                var += dist;
                n++;
                ASSERT_LT(dist, 10);
            }
            std::cout << var / n << std::endl;
        }
    }

    {
        size_t num_test_markers = 50;

        glm::vec3 voxel_grid_size(0.02f);

        std::vector<glm::vec3> test_markers;
        gen_random_cylinder_marker(rand, num_test_markers, glm::vec3(1.5f), voxel_grid_size, test_markers);

        const auto thresh = 2.0;

        std::vector<std::vector<glm::vec2>> pts(num_cameras);
        for (size_t m = 0; m < test_markers.size(); m++)
        {
            const auto marker = test_markers[m];

            for (size_t c = 0; c < num_cameras; c++)
            {
                const auto camera = scene.cameras[c];

                const auto pt = project(camera, marker);

                auto pixel_err = (pt.x < 0) ? 0.0f : 4.0f;
                // auto pixel_err = 0.0f;

                std::uniform_real_distribution<> x_dist(-pixel_err, pixel_err);
                std::uniform_real_distribution<> y_dist(-pixel_err, pixel_err);

                glm::vec2 observed_pt(pt.x + x_dist(rand), pt.y + y_dist(rand));

                pts[c].push_back(observed_pt);
            }
        }

        std::vector<std::vector<size_t>> index(num_cameras);
        std::vector<std::vector<size_t>> rev_index(num_cameras);
        std::vector<std::vector<glm::vec2>> shuffle_pts(num_cameras);

        for (size_t c = 0; c < num_cameras; c++)
        {
            index[c].resize(pts[c].size());
            std::iota(index[c].begin(), index[c].end(), 0);

            if (c != 0)
            {
                std::shuffle(index[c].begin(), index[c].end(), rand);
            }

            shuffle_pts[c].resize(index[c].size());
            rev_index[c].resize(index[c].size());

            for (size_t i = 0; i < index[c].size(); i++)
            {
                shuffle_pts[c][index[c][i]] = pts[c][i];
                rev_index[c][index[c][i]] = i;
            }
        }

        find_correspondance(shuffle_pts, scene.cameras, adjusted_poses, index, thresh);
    }
}
#endif

namespace glm
{
    static void to_json(nlohmann::json &j, const glm::vec2 &v)
    {
        j = {v.x, v.y};
    }
    static void from_json(const nlohmann::json &j, glm::vec2 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::vec3 &v)
    {
        j = {v.x, v.y, v.z};
    }
    static void from_json(const nlohmann::json &j, glm::vec3 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
        v.z = j[2].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::vec4 &v)
    {
        j = {v.x, v.y, v.z, v.w};
    }
    static void from_json(const nlohmann::json &j, glm::vec4 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
        v.z = j[2].get<float>();
        v.w = j[3].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::mat4 &m)
    {
        j = {m[0], m[1], m[2], m[3]};
    }
    static void from_json(const nlohmann::json &j, glm::mat4 &m)
    {
        m[0] = j[0].get<glm::vec4>();
        m[1] = j[1].get<glm::vec4>();
        m[2] = j[2].get<glm::vec4>();
        m[3] = j[3].get<glm::vec4>();
    }
}

static void to_json(nlohmann::json &j, const camera_intrin_t &intrin)
{
    j = {
        {"fx", intrin.fx},
        {"fy", intrin.fy},
        {"cx", intrin.cx},
        {"cy", intrin.cy},
        {"coeffs", intrin.coeffs},
    };
}

static void from_json(const nlohmann::json &j, camera_intrin_t &intrin)
{
    intrin.fx = j["fx"].get<float>();
    intrin.fy = j["fy"].get<float>();
    intrin.cx = j["cx"].get<float>();
    intrin.cy = j["cy"].get<float>();
    intrin.coeffs = j["coeffs"].get<std::array<float, 5>>();
}

static void to_json(nlohmann::json &j, const camera_extrin_t &extrin)
{
    j = {
        {"rotation", extrin.rotation},
        {"translation", extrin.translation},
    };
}

static void from_json(const nlohmann::json &j, camera_extrin_t &extrin)
{
    extrin.rotation = j["rotation"].get<glm::mat4>();
    extrin.translation = j["translation"].get<glm::vec3>();
}

static void to_json(nlohmann::json &j, const camera_t &camera)
{
    j = {
        {"intrin", camera.intrin},
        {"extrin", camera.extrin},
        {"width", camera.width},
        {"height", camera.height},
    };
}

static void from_json(const nlohmann::json &j, camera_t &camera)
{
    camera.intrin = j["intrin"].get<camera_intrin_t>();
    camera.extrin = j["extrin"].get<camera_extrin_t>();
    camera.width = j["width"].get<uint32_t>();
    camera.height = j["height"].get<uint32_t>();
}

TEST(IrReconstruction, TestReconstruction)
{
    std::vector<reconstruction::node_t> nodes;
    reconstruction::adj_list_t adj;

    nlohmann::json j;
    std::ifstream ifs("../test/dump.json");
    ifs >> j;

    const auto pts = j["pts"].get<std::vector<std::vector<glm::vec2>>>();
    const auto cameras = j["cameras"].get<std::vector<camera_t>>();
    constexpr auto thresh = 1.0;

    reconstruction::find_correspondance(pts, cameras, nodes, adj, thresh, 20);

    reconstruction::save_graphs(nodes, adj, "before_cut");
    std::vector<std::vector<std::size_t>> connected_components;
    reconstruction::compute_observations_with_filter(nodes, adj, cameras, connected_components, 0.02);

    // reconstruction::save_graphs(nodes, adj, "after_cut");

    std::vector<glm::vec3> markers;
    for (auto &g : connected_components)
    {
        if (g.size() < 4)
        {
            continue;
        }

        std::vector<glm::vec2> pts;
        std::vector<camera_t> cams;

        for (std::size_t i = 0; i < g.size(); i++)
        {
            pts.push_back(nodes[g[i]].pt);
            cams.push_back(cameras[nodes[g[i]].camera_idx]);
        }
        const auto marker = triangulate(pts, cams);

        markers.push_back(marker);
    }

    {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
        p_cloud->width = markers.size();
        p_cloud->height = 1;
        p_cloud->points.resize(p_cloud->width * p_cloud->height);

        for (std::size_t i = 0; i < markers.size(); i++)
        {
            pcl::PointXYZRGBA &point = p_cloud->points[i];

            const auto &marker = markers[i];

            point.x = marker.x;
            point.y = marker.y;
            point.z = marker.z;
            point.r = 255;
            point.g = 0;
            point.b = 0;
            point.a = 0;
        }

        if (markers.size() > 0)
        {
            pcl::io::savePCDFileASCII("dump.pcd", *p_cloud);
        }
    }
}
