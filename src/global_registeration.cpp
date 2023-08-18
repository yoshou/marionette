#include "global_registeration.hpp"

#include <unordered_set>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/constants.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>

#include "model.hpp"
#include "nanoflann.hpp"
#include "transform.hpp"
#include "features.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#endif

void find_nearest_pair(
    const std::vector<weighted_point> &source_points, const glm::mat4 &source_pose, const glm::mat4 &inv_pose, const point_cloud &target_points,
    glm::vec3 translation, glm::mat3 rotation, std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle)
{
    std::unordered_set<std::size_t> mask;
    std::vector<point_cloud::index_type> knn_index(source_points.size());
    std::vector<float> knn_distsq(source_points.size());

    pairs.clear();
    for (std::size_t i = 0; i < source_points.size(); i++)
    {
        std::size_t min_idx = 0;

        const auto source_point = rotation * twist(source_points[i], source_pose, inv_pose, twist_angle) + translation;

        std::size_t num_points = target_points.knn_search(source_point, source_points.size(), knn_index.data(), knn_distsq.data());
        for (std::size_t k = 0; k < num_points; k++)
        {
            const auto j = knn_index[k];

            if (mask.find(j) == mask.end())
            {
                min_idx = j;
                break;
            }
        }

        mask.insert(min_idx);
        pairs.push_back(std::make_pair(i, min_idx));
    }
}

void find_nearest_pair(
    const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
    const glm::vec3 &translation, const glm::mat3 &rotation, std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle)
{
    std::array<std::size_t, 100> mask = {};
    std::array<point_cloud::index_type, 16> knn_index;
    std::array<float, 16> knn_distsq;

    pairs.resize(source_points.size());
    for (std::size_t i = 0; i < source_points.size(); i++)
    {
        const auto source_point = rotation * twist(source_points[i], twist_axis, twist_angle) + translation;
        std::size_t num_points = target_points.knn_search(source_point, source_points.size(), knn_index.data(), knn_distsq.data());

        std::size_t min_idx = target_points.size();
        for (std::size_t k = 0; k < num_points; k++)
        {
            const auto j = knn_index[k];
            if (!mask[j])
            {
                min_idx = j;
                break;
            }
        }

        if (min_idx < target_points.size())
        {
            mask[min_idx] = 1;
            pairs[i] = std::make_pair(i, min_idx);
        }
    }
}

static inline float mse(
    const std::vector<weighted_point> &source_points, const glm::mat4 &source_pose, const glm::mat4 &inv_pose, const point_cloud &target_points,
    const glm::vec3 &translation, const glm::mat3 &rotation, const std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle)
{
    if (pairs.size() == 0)
    {
        return 0.f;
    }

    float error = 0.f;
    for (const auto &[i, j] : pairs)
    {
        const auto source_point = twist(source_points[i], source_pose, inv_pose, twist_angle);
        const auto target_point = target_points[j];
        const auto dist_sq = glm::distance2(rotation * source_point + translation, target_point);
        error += dist_sq;
    }

    return std::sqrt(error) / pairs.size();
}

static inline float mse(
    const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
    const glm::vec3 &translation, const glm::mat3 &rotation, const std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle)
{
    if (pairs.size() == 0)
    {
        return 0.f;
    }

    float error = 0.f;
    for (const auto &[i, j] : pairs)
    {
        const auto source_point = twist(source_points[i], twist_axis, twist_angle);
        const auto target_point = target_points[j];
        const auto dist_sq = glm::distance2(rotation * source_point + translation, target_point);
        error += dist_sq;
    }

    return std::sqrt(error) / pairs.size();
}

static inline float mse(
    const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
    const glm::vec3 &translation, const glm::mat3 &rotation, const std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle, const float *max_rotation_dists)
{
    if (pairs.size() == 0)
    {
        return 0.f;
    }

    float error = 0.f;
    for (const auto &[i, j] : pairs)
    {
        const auto source_point = twist(source_points[i], twist_axis, twist_angle);
        const auto target_point = target_points[j];
        auto dist_sq = glm::distance(rotation * source_point + translation, target_point);

        if (max_rotation_dists)
        {
            dist_sq -= max_rotation_dists[i];
            dist_sq = std::max(dist_sq, 0.f);
        }
        error += dist_sq;
    }

    return error / pairs.size();
}

static std::pair<glm::vec3, glm::vec3> centroid(
    const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
    glm::vec3 translation, glm::mat3 rotation, const std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle)
{
    glm::vec3 centroid_a(0.f);
    glm::vec3 centroid_b(0.f);

    if (pairs.size() == 0)
    {
        return std::make_pair(centroid_a, centroid_b);
    }

    for (const auto &[i, j] : pairs)
    {
        const auto source_point = rotation * twist(source_points[i], twist_axis, twist_angle) + translation;
        const auto target_point = target_points[j];

        centroid_a += source_point;
        centroid_b += target_point;
    }

    const auto num = (float)pairs.size();

    return std::make_pair(centroid_a / num, centroid_b / num);
}

static inline void solve_svd(const float *a, float *u, float *v, float *s, std::size_t m, std::size_t n)
{
    Eigen::Map<const Eigen::MatrixXf> a_m(a, m, n);
    Eigen::Map<Eigen::MatrixXf> u_m(u, m, m);
    Eigen::Map<Eigen::MatrixXf> v_m(v, n, n);
    Eigen::Map<Eigen::VectorXf> s_v(s, n);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(a_m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    u_m = svd.matrixU();
    v_m = svd.matrixV();
    s_v = svd.singularValues();
}

#ifdef USE_CUDA
static void solve_svd_batched(
    const thrust::host_vector<float> &a,
    thrust::host_vector<float> &u,
    thrust::host_vector<float> &v,
    thrust::host_vector<float> &s,
    std::size_t m, std::size_t n, std::size_t batch_size)
{
#if 0
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int rank = n;
    const long long int strideA = (long long int)lda * n;
    const long long int strideS = n;
    const long long int strideU = (long long int)ldu * m;
    const long long int strideV = (long long int)ldv * n;

    float *d_A = NULL;     /* lda-by-n-by-batch_size */
    float *d_U = NULL;     /* ldu-by-m-by-batch_size */
    float *d_V = NULL;     /* ldv-by-n-by-batch_size */
    float *d_S = NULL;     /* minmn-by-batch_sizee */
    int *d_info = NULL;    /* batch_size */
    int lwork = 0;         /* size of workspace */
    float *d_work = NULL;  /* device workspace for gesvdjBatched */
    const auto RnrmF = std::make_unique<double[]>(batch_size); /* residual norm */
    const auto info = std::make_unique<int[]>(batch_size); /* residual norm */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.

    /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);
    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 2: copy A to device */
    cudaStat1 = cudaMalloc((void **)&d_A, sizeof(float) * strideA * batch_size);
    cudaStat2 = cudaMalloc((void **)&d_U, sizeof(float) * strideU * batch_size);
    cudaStat3 = cudaMalloc((void **)&d_V, sizeof(float) * strideV * batch_size);
    cudaStat4 = cudaMalloc((void **)&d_S, sizeof(float) * strideS * batch_size);
    cudaStat5 = cudaMalloc((void **)&d_info, sizeof(int) * batch_size);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    cudaStat1 = cudaMemcpy(d_A, a.data(), sizeof(float) * strideA * batch_size, cudaMemcpyHostToDevice);
    cudaStat2 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query workspace of SVD */
    status = cusolverDnSgesvdaStridedBatched_bufferSize(
        cusolverH,
        jobz,    /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                 /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank,    /* number of singular values */
        m,       /* nubmer of rows of Aj, 0 <= m */
        n,       /* number of columns of Aj, 0 <= n  */
        d_A,     /* Aj is m-by-n */
        lda,     /* leading dimension of Aj */
        strideA, /* >= lda*n */
        d_S,     /* Sj is rank-by-1, singular values in descending order */
        strideS, /* >= rank */
        d_U,     /* Uj is m-by-rank */
        ldu,     /* leading dimension of Uj, ldu >= max(1,m) */
        strideU, /* >= ldu*rank */
        d_V,     /* Vj is n-by-rank */
        ldv,     /* leading dimension of Vj, ldv >= max(1,n) */
        strideV, /* >= ldv*rank */
        &lwork,
        batch_size /* number of matrices */
    );

    cudaStat1 = cudaMalloc((void **)&d_work, sizeof(float) * lwork);
    assert(cudaSuccess == cudaStat1);

    std::chrono::system_clock::time_point start, end; // 型は auto で可
    start = std::chrono::system_clock::now();         // 計測開始時間
    /* step 4: compute SVD */
    status = cusolverDnSgesvdaStridedBatched(
        cusolverH,
        jobz,    /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                 /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank,    /* number of singular values */
        m,       /* nubmer of rows of Aj, 0 <= m */
        n,       /* number of columns of Aj, 0 <= n  */
        d_A,     /* Aj is m-by-n */
        lda,     /* leading dimension of Aj */
        strideA, /* >= lda*n */
        d_S,     /* Sj is rank-by-1 */
                 /* the singular values in descending order */
        strideS, /* >= rank */
        d_U,     /* Uj is m-by-rank */
        ldu,     /* leading dimension of Uj, ldu >= max(1,m) */
        strideU, /* >= ldu*rank */
        d_V,     /* Vj is n-by-rank */
        ldv,     /* leading dimension of Vj, ldv >= max(1,n) */
        strideV, /* >= ldv*rank */
        d_work,
        lwork,
        d_info,
        RnrmF.get(),
        batch_size /* number of matrices */
    );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);
    end = std::chrono::system_clock::now(); // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << elapsed << std::endl;
    std::cout << "batch size : " << batch_size << std::endl;

    cudaStat1 = cudaMemcpy(u.data(), d_U, sizeof(float) * strideU * batch_size, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(v.data(), d_V, sizeof(float) * strideV * batch_size, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(s.data(), d_S, sizeof(float) * strideS * batch_size, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaMemcpy(info.get(), d_info, sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    if (d_A)
        cudaFree(d_A);
    if (d_U)
        cudaFree(d_U);
    if (d_V)
        cudaFree(d_V);
    if (d_S)
        cudaFree(d_S);
    if (d_info)
        cudaFree(d_info);
    if (d_work)
        cudaFree(d_work);

    if (cusolverH)
        cusolverDnDestroy(cusolverH);
    if (stream)
        cudaStreamDestroy(stream);
#else
    for (std::size_t i = 0; i < batch_size; i++)
    {
        solve_svd(a.data() + i * m * n, u.data() + i * m * m, v.data() + i * n * n, s.data() + i * n, m, n);
    }
#endif
}
#endif

struct icp_minimizer
{
    std::vector<std::pair<std::size_t, std::size_t>> pairs;

    void update_rotation(
        const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
        const glm::vec3 &translation, glm::mat3 &rotation, float twist_angle)
    {
        Eigen::MatrixXf m1(pairs.size(), 3);
        Eigen::MatrixXf m2(pairs.size(), 3);

        for (std::size_t k = 0; k < pairs.size(); k++)
        {
            const auto [i, j] = pairs[k];
            const auto source_point = rotation * twist(source_points[i], twist_axis, twist_angle);
            const auto target_point = target_points[j] - translation;

            m1(k, 0) = source_point.x;
            m1(k, 1) = source_point.y;
            m1(k, 2) = source_point.z;

            m2(k, 0) = target_point.x;
            m2(k, 1) = target_point.y;
            m2(k, 2) = target_point.z;
        }

        const Eigen::MatrixXf a = m1.transpose() * m2;

        const std::size_t m = a.rows();
        const std::size_t n = a.cols();

        Eigen::MatrixXf u(m, m);
        Eigen::MatrixXf v(n, n);
        Eigen::VectorXf s(n);

        solve_svd(a.data(), u.data(), v.data(), s.data(), m, n);

        Eigen::MatrixXf u_t = u.transpose();

        Eigen::MatrixXf r = v * u_t;
        const float det = r.determinant();
        if (det < 0.f)
        {
            v(0, 2) = -v(0, 2);
            v(1, 2) = -v(1, 2);
            v(2, 2) = -v(2, 2);
            r = v * u_t;
        }

        glm::mat3 rot(1.f);
        for (std::size_t i = 0; i < 3; i++)
        {
            for (std::size_t j = 0; j < 3; j++)
            {
                rot[j][i] = r(i, j);
            }
        }

        rotation = rot * rotation;
    }

    void update_translation(
        const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
        glm::vec3 &translation, const glm::mat3 &rotation, float twist_angle)
    {
        const auto [centroid_a, centroid_b] = centroid(source_points, twist_axis, target_points, translation, rotation, pairs, twist_angle);

        translation += (centroid_b - centroid_a);
    }

    void update_twist_angle(
        const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
        const glm::vec3 &translation, const glm::mat3 &rotation, float &twist_angle, float min_twist_angle, float max_twist_angle, float d_angle = 10.f)
    {
        const auto error1 = mse(source_points, twist_axis, target_points, translation, rotation, pairs, twist_angle);
        const auto error2 = mse(source_points, twist_axis, target_points, translation, rotation, pairs, twist_angle + d_angle);

        twist_angle = std::max(min_twist_angle, std::min(max_twist_angle, twist_angle - d_angle * (error2 - error1) / error1));
    }

    float compute(
        const std::vector<weighted_point> &source_points, const glm::mat4 &source_pose, const point_cloud &target_points,
        glm::vec3 &translation, glm::mat3 &rotation, float &twist_angle, float min_twist_angle, float max_twist_angle, std::uint32_t max_iter = 10, float threshold = 1e-4f)
    {
        float last_error = std::numeric_limits<float>::max();

        const auto twist_axis = to_line(source_pose);

        for (std::uint32_t iter = 0; iter < max_iter; iter++)
        {
            find_nearest_pair(source_points, twist_axis, target_points, translation, rotation, pairs, twist_angle);

            update_rotation(source_points, twist_axis, target_points, translation, rotation, twist_angle);

            update_twist_angle(source_points, twist_axis, target_points, translation, rotation, twist_angle, min_twist_angle, max_twist_angle);

            //update_translation(source_points, twist_axis, target_points, translation, rotation, twist_angle);

            const auto error = mse(source_points, twist_axis, target_points, translation, rotation, pairs, twist_angle);
            if (iter > 0 && (std::abs(last_error - error) / last_error) < threshold)
            {
                last_error = error;
                break;
            }

            last_error = error;
        }

        return last_error;
    }
};

struct icp_context
{
    std::size_t i, j;
    glm::vec3 translation;
    glm::mat3 rotation;
    float twist_angle;
    float error;
    float done;
    icp_minimizer icp;
};

struct initial_rotations
{
    std::vector<glm::mat3> rots;

    initial_rotations()
    {
        const auto x_rots = {
            glm::rotate(glm::radians(-90.0f), glm::vec3(1, 0, 0)),
            // glm::rotate(glm::radians(-45.0f), glm::vec3(1, 0, 0)),
            glm::rotate(glm::radians(0.0f), glm::vec3(1, 0, 0)),
            // glm::rotate(glm::radians(45.0f), glm::vec3(1, 0, 0)),
            glm::rotate(glm::radians(90.0f), glm::vec3(1, 0, 0)),
            // glm::rotate(glm::radians(135.0f), glm::vec3(1, 0, 0)),
            glm::rotate(glm::radians(180.0f), glm::vec3(1, 0, 0)),
            // glm::rotate(glm::radians(225.0f), glm::vec3(1, 0, 0)),
        };

        const auto y_rots = {
            glm::rotate(glm::radians(-90.0f), glm::vec3(0, 1, 0)),
            // glm::rotate(glm::radians(-45.0f), glm::vec3(0, 1, 0)),
            glm::rotate(glm::radians(0.0f), glm::vec3(0, 1, 0)),
            // glm::rotate(glm::radians(45.0f), glm::vec3(0, 1, 0)),
            glm::rotate(glm::radians(90.0f), glm::vec3(0, 1, 0)),
            // glm::rotate(glm::radians(135.0f), glm::vec3(0, 1, 0)),
            glm::rotate(glm::radians(180.0f), glm::vec3(0, 1, 0)),
            // glm::rotate(glm::radians(225.0f), glm::vec3(0, 1, 0)),
        };

        const auto z_rots = {
            glm::rotate(glm::radians(-90.0f), glm::vec3(0, 0, 1)),
            // glm::rotate(glm::radians(-45.0f), glm::vec3(0, 0, 1)),
            glm::rotate(glm::radians(0.0f), glm::vec3(0, 0, 1)),
            // glm::rotate(glm::radians(45.0f), glm::vec3(0, 0, 1)),
            glm::rotate(glm::radians(90.0f), glm::vec3(0, 0, 1)),
            // glm::rotate(glm::radians(135.0f), glm::vec3(0, 0, 1)),
            glm::rotate(glm::radians(180.0f), glm::vec3(0, 0, 1)),
            // glm::rotate(glm::radians(225.0f), glm::vec3(0, 0, 1)),
        };

        for (const auto &x_rot : x_rots)
        {
            for (const auto &y_rot : y_rots)
            {
                for (const auto &z_rot : z_rots)
                {
                    const auto rotation = glm::mat3(z_rot * y_rot * x_rot);
                    rots.push_back(rotation);
                }
            }
        }
    }
};

static void compute_max_rotation_dists(const std::vector<weighted_point> &source_points, std::vector<std::vector<float>> &max_rotation_dists)
{
    std::vector<float> source_point_norms;
    for (std::size_t i = 0; i < source_points.size(); i++)
    {
        const auto norm = glm::length(source_points[i].position);
        source_point_norms.push_back(norm);
    }

    const auto max_rotation_level = max_rotation_dists.size();
    for (std::size_t l = 0; l < max_rotation_level; l++)
    {
        const auto sigma = 2.0 * glm::pi<double>() / std::pow(2.0, l) / 2.0; // Half-side length of each level of rotation subcube
        auto max_angle = std::sqrt(3.0) * sigma;
        max_angle = std::min(max_angle, glm::pi<double>());
        for (std::size_t j = 0; j < source_point_norms.size(); j++)
        {
            const auto dist = 2.0 * std::sin(max_angle / 2.0) * source_point_norms[j];
            max_rotation_dists[l].push_back(dist);
        }
    }
}

static initial_rotations initial_rots;

static std::tuple<float, glm::mat3, glm::vec3, float> estimate_rotation(
    const std::vector<weighted_point> &source_points, const glm::mat4 source_pose, const point_cloud &target_points,
    glm::vec3 initial_translation, float initial_twist_angle, float min_twist_angle, float max_twist_angle)
{
    const auto &rots = initial_rots.rots;

    float min_error = std::numeric_limits<float>::max();
    glm::mat3 best_rotation(1.f);
    glm::vec3 best_translation(0.f);
    float best_twist_angle = 0.f;

    // constexpr auto max_rotation_level = 10;
    // std::vector<std::vector<float>> max_rotation_dists(max_rotation_level);
    // compute_max_rotation_dists(source_points, max_rotation_dists);

    for (std::size_t i = 0; i < rots.size(); i++)
    {
        auto rotation = rots[i];
        auto translation = initial_translation;
        auto twist_angle = initial_twist_angle;

        icp_minimizer icp;
        const auto error = icp.compute(source_points, source_pose, target_points, translation, rotation,
                                       twist_angle, min_twist_angle, max_twist_angle);

        if (error < min_error)
        {
            min_error = error;
            best_rotation = rotation;
            best_translation = translation;
            best_twist_angle = twist_angle;
        }
    }

    return std::make_tuple(min_error, best_rotation, best_translation, best_twist_angle);
}

static std::pair<triangle_feature, float> compute_feature(const std::vector<weighted_point> &source_points)
{
    triangle_feature source_feature;
    float feature_radius;
    {
        std::vector<glm::vec3> points(source_points.size());
        for (std::size_t i = 0; i < points.size(); i++)
        {
            points[i] = source_points[i].position;
        }

        feature_radius = compute_max_distance(points);

        const auto anchor_pos = points[0];
        source_feature = extract_triangle_feature(anchor_pos, points, std::numeric_limits<float>::max());
    }
    return std::make_pair(source_feature, feature_radius);
}

static void find_fit(
    const std::vector<weighted_point> &source_points, const glm::mat4 &source_pose, const point_cloud &target_points,
    float initial_twist_angle, float min_twist_angle, float max_twist_angle,
    std::vector<find_fit_result> &results)
{
    if (source_points.size() == 0 || target_points.size() == 0)
    {
        return;
    }

    const auto anchor_pos = source_points[0].position;

    std::vector<weighted_point> rel_source_points;
    for (const auto &point : source_points)
    {
        rel_source_points.push_back(weighted_point{
            point.position - anchor_pos, point.weight, point.id});
    }

    const auto rel_source_pose = glm::translate(glm::mat4(1.f), -anchor_pos) * source_pose;

    const auto [source_feature, feature_radius] = compute_feature(source_points);

    for (std::size_t i = 0; i < target_points.size(); i++)
    {
        const auto feature = extract_triangle_feature(target_points[i], target_points.points, feature_radius * 1.2f);
        const auto feature_dist = compute_feature_distance(source_feature, feature);

        if (feature_dist > 0.05f)
        {
            continue;
        }

        const auto initial_translation = target_points[i];
        const auto [error, rotation, translation, twist_angle] = estimate_rotation(rel_source_points, rel_source_pose, target_points, initial_translation,
                                                                                   initial_twist_angle, min_twist_angle, max_twist_angle);

        auto twist_axis = to_line(rel_source_pose);
        twist_axis.direction *= twist_angle;

        results.push_back(find_fit_result(0, rotation, translation, twist_angle, error));
    }
}

static void find_fit_batch(
    const std::vector<weighted_point> &source_points, const glm::mat4 &source_pose, const point_cloud &target_points,
    float initial_twist_angle, float min_twist_angle, float max_twist_angle,
    std::vector<find_fit_result> &results)
{
    if (source_points.size() == 0 || target_points.size() == 0)
    {
        return;
    }

    const auto anchor_pos = source_points[0].position;

    std::vector<weighted_point> rel_source_points;
    for (const auto &point : source_points)
    {
        rel_source_points.push_back(weighted_point{
            point.position - anchor_pos, point.weight, point.id});
    }

    const auto rel_source_pose = glm::translate(glm::mat4(1.f), -anchor_pos) * source_pose;

    const auto &rots = initial_rots.rots;
    std::uint32_t max_iter = 10;
    float threshold = 1e-4f;

    std::vector<icp_context> ctxs;
    for (std::size_t i = 0; i < target_points.size(); i++)
    {
        const auto initial_translation = target_points[i];

        {
            float min_error = std::numeric_limits<float>::max();
            glm::mat3 best_rotation(1.f);
            glm::vec3 best_translation(0.f);
            float best_twist_angle = 0.f;

            for (std::size_t j = 0; j < rots.size(); j++)
            {
                icp_context ctx;
                ctx.i = i;
                ctx.j = j;
                ctx.rotation = rots[j];
                ctx.translation = initial_translation;
                ctx.twist_angle = initial_twist_angle;
                ctx.error = std::numeric_limits<float>::max();
                ctx.done = false;

                ctxs.push_back(ctx);
            }
        }
    }

    for (std::uint32_t iter = 0; iter < max_iter; iter++)
    {
        for (auto &ctx : ctxs)
        {
            if (ctx.done)
            {
                continue;
            }

            auto &icp = ctx.icp;
            auto &translation = ctx.translation;
            auto &rotation = ctx.rotation;
            auto &twist_angle = ctx.twist_angle;

            const auto twist_axis = to_line(rel_source_pose);

            find_nearest_pair(rel_source_points, twist_axis, target_points, translation, rotation, icp.pairs, twist_angle);

            //icp.update_translation(source_points, twist_axis, target_points, translation, rotation, twist_angle);
        }

#if 1
        for (auto &ctx : ctxs)
        {
            if (ctx.done)
            {
                continue;
            }

            auto &icp = ctx.icp;
            const auto &translation = ctx.translation;
            auto &rotation = ctx.rotation;
            const auto twist_angle = ctx.twist_angle;
            const auto twist_axis = to_line(rel_source_pose);

            icp.update_rotation(rel_source_points, twist_axis, target_points, translation, rotation, twist_angle);
        }
#else

        thrust::host_vector<float> a_batch;
        for (auto &ctx : ctxs)
        {
            if (ctx.done)
            {
                continue;
            }

            auto &icp = ctx.icp;
            const auto &translation = ctx.translation;
            auto &rotation = ctx.rotation;
            const auto twist_angle = ctx.twist_angle;
            const auto twist_axis = to_line(rel_source_pose);

            Eigen::MatrixXf m1(icp.pairs.size(), 3);
            Eigen::MatrixXf m2(icp.pairs.size(), 3);

            for (std::size_t k = 0; k < icp.pairs.size(); k++)
            {
                const auto [p, q] = icp.pairs[k];
                const auto source_point = rotation * twist(rel_source_points[p], twist_axis, twist_angle);
                const auto target_point = target_points[q] - translation;

                m1(k, 0) = source_point.x;
                m1(k, 1) = source_point.y;
                m1(k, 2) = source_point.z;

                m2(k, 0) = target_point.x;
                m2(k, 1) = target_point.y;
                m2(k, 2) = target_point.z;
            }

            const Eigen::MatrixXf a = m1.transpose() * m2;
            assert(a.size() == 9);

            std::copy(a.data(), a.data() + a.size(), std::back_inserter(a_batch));
        }
        std::size_t m = 3;
        std::size_t n = 3;
        std::size_t batch_size = a_batch.size() / m / n;
        thrust::host_vector<float> u_batch(m * m * batch_size);
        thrust::host_vector<float> v_batch(n * n * batch_size);
        thrust::host_vector<float> s_batch(n * batch_size);

        if (batch_size > 0)
        {
            solve_svd_batched(a_batch, u_batch, v_batch, s_batch, m, n, batch_size);
        }

        std::size_t idx = 0;
        for (auto &ctx : ctxs)
        {
            if (ctx.done)
            {
                continue;
            }

            auto &icp = ctx.icp;
            const auto &translation = ctx.translation;
            auto &rotation = ctx.rotation;
            const auto twist_angle = ctx.twist_angle;
            const auto twist_axis = to_line(rel_source_pose);

            Eigen::Map<Eigen::MatrixXf> u(u_batch.data() + idx * m * m, m, m);
            Eigen::Map<Eigen::MatrixXf> v(v_batch.data() + idx * n * n, n, n);
            idx++;

            Eigen::MatrixXf r = v * u.transpose();
            const float det = r.determinant();
            if (det < 0.f)
            {
                v(0, 2) = -v(0, 2);
                v(1, 2) = -v(1, 2);
                v(2, 2) = -v(2, 2);
                r = v * u.transpose();
            }

            glm::mat3 rot(1.f);
            for (std::size_t i = 0; i < 3; i++)
            {
                for (std::size_t j = 0; j < 3; j++)
                {
                    rot[j][i] = r(i, j);
                }
            }

            rotation = rot * rotation;
        }
#endif

        for (auto &ctx : ctxs)
        {
            if (ctx.done)
            {
                continue;
            }

            auto &icp = ctx.icp;
            const auto &translation = ctx.translation;
            const auto &rotation = ctx.rotation;
            auto &twist_angle = ctx.twist_angle;
            const auto twist_axis = to_line(rel_source_pose);

            icp.update_twist_angle(rel_source_points, twist_axis, target_points, translation, rotation, twist_angle, min_twist_angle, max_twist_angle);
        }

        for (auto &ctx : ctxs)
        {
            if (ctx.done)
            {
                continue;
            }

            const auto &icp = ctx.icp;
            const auto &translation = ctx.translation;
            const auto &rotation = ctx.rotation;
            const auto twist_angle = ctx.twist_angle;
            const auto twist_axis = to_line(rel_source_pose);

            const float last_error = ctx.error;
            ctx.error = mse(rel_source_points, twist_axis, target_points, translation, rotation, icp.pairs, twist_angle);

            if (iter > 0 && (std::abs(last_error - ctx.error) / last_error) < threshold)
            {
                ctx.done = true;
            }
        }
    }

    for (std::size_t i = 0; i < target_points.size(); i++)
    {
        find_fit_result result;
        result.anchor_index = 0;
        results.push_back(result);
    }
    
    for (auto &ctx : ctxs)
    {
        if (ctx.error < results[ctx.i].error)
        {
            const auto error = ctx.error;
            const auto rotation = ctx.rotation;
            const auto translation = ctx.translation;
            const auto twist_angle = ctx.twist_angle;

            auto twist_axis = to_line(rel_source_pose);
            twist_axis.direction *= twist_angle;

            results[ctx.i] = find_fit_result(0, rotation, translation, twist_angle, error);
        }
    }
}

void global_registration::find_fits(const rigid_cluster &cluster, const point_cloud &cloud,
                                    float initial_twist_angle, float min_twist_angle, float max_twist_angle, std::vector<find_fit_result> &results)
{
    if (cluster.points.size() == 0)
    {
        return;
    }

    //find_fit_batch(cluster.points, cluster.pose, cloud, initial_twist_angle, min_twist_angle, max_twist_angle, results);
    find_fit(cluster.points, cluster.pose, cloud, initial_twist_angle, min_twist_angle, max_twist_angle, results);

    std::sort(results.begin(), results.end(), [](const find_fit_result &a, const find_fit_result &b)
              { return a.error < b.error; });
}

void global_registration::find_fits(const rigid_cluster &cluster, const std::vector<glm::vec3> &points,
                                    float initial_twist_angle, float min_twist_angle, float max_twist_angle, std::vector<find_fit_result> &results)
{
    if (cluster.points.size() == 0)
    {
        return;
    }

    point_cloud cloud(points);
    cloud.build_index();

    find_fits(cluster, cloud, initial_twist_angle, min_twist_angle, max_twist_angle, results);
}
