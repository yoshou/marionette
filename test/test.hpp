#pragma once

void test_registeration()
{
#if 0
    {
        auto &cluster = model.clusters.at("upper_arm.R");
        const auto source_pose = cluster.pose;

        std::vector<weighted_point> source_points;
        for (const auto point : cluster.points)
        {
            source_points.push_back(point);
        }

        const auto anchor_pos = source_points[0].position;

        std::vector<weighted_point> rel_source_points;
        for (const auto point : source_points)
        {
            rel_source_points.push_back(weighted_point{
                point.position - anchor_pos, point.weight, point.id});
        }

        const auto rel_source_pose = glm::translate(glm::mat4(1.f), -anchor_pos) * source_pose;
        std::vector<glm::vec3> points1;
        for (const auto point : rel_source_points)
        {
            points1.push_back(glm::vec3(glm::inverse(rel_source_pose) * glm::vec4(point.position, 1.f)));
        }
        std::vector<glm::vec3> points2;
        for (const auto point : rel_source_points)
        {
            points2.push_back(glm::vec3(glm::inverse(rel_source_pose) * glm::vec4(point.position, 1.f)));
        }

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

        {
            const auto &target_points = frame.markers;

            for (std::size_t i = 0; i < points1.size(); i++)
            {
                pcl::PointXYZRGBA point;

                point.x = points1[i].x;
                point.y = points1[i].y;
                point.z = points1[i].z;
                point.r = 255;
                point.g = 255;
                point.b = 255;
                point.a = 0;

                p_cloud->points.push_back(point);
            }
            glm::mat3 rotation = glm::rotate(45.0f * (float)M_PI / 180.0f, glm::vec3(0, 1, 0));
            for (std::size_t i = 0; i < points2.size(); i++)
            {
                pcl::PointXYZRGBA point;

                const auto weight = rel_source_points[i].weight;
                const auto weight_mat = glm::mat3(glm::scale(glm::vec3(1.f, 0.f, 1.f)));
                const auto pt = ((rotation * weight_mat) + (glm::mat3(1.f) - weight_mat)) * points2[i];

                std::cout << glm::to_string(rotation) << std::endl;
                std::cout << glm::to_string(weight_mat) << std::endl;
                std::cout << glm::to_string(((weight_mat * rotation) + (glm::mat3(1.f) - weight_mat))) << std::endl;
                std::cout << "----------" << std::endl;

                point.x = pt.x;
                point.y = pt.y;
                point.z = pt.z;
                point.r = 255;
                point.g = 0;
                point.b = 0;
                point.a = 0;

                p_cloud->points.push_back(point);
            }
            // for (std::size_t i = 0; i < target_points.size(); i++)
            // {
            //     pcl::PointXYZRGBA point;

            //     const auto pt = target_points[i] - target_points[40];

            //     point.x = pt.x;
            //     point.y = pt.y;
            //     point.z = pt.z;
            //     point.r = 255;
            //     point.g = 0;
            //     point.b = 0;
            //     point.a = 0;

            //     p_cloud->points.push_back(point);
            // }
            {
                pcl::PointXYZRGBA point;

                point.x = 0;
                point.y = 0;
                point.z = 0;
                point.r = 0;
                point.g = 0;
                point.b = 0;
                point.a = 0;

                p_cloud->points.push_back(point);
            }
            {
                pcl::PointXYZRGBA point;

                point.x = 1;
                point.y = 0;
                point.z = 0;
                point.r = 255;
                point.g = 0;
                point.b = 0;
                point.a = 0;

                p_cloud->points.push_back(point);
            }
            {
                pcl::PointXYZRGBA point;

                point.x = 0;
                point.y = 1;
                point.z = 0;
                point.r = 0;
                point.g = 255;
                point.b = 0;
                point.a = 0;

                p_cloud->points.push_back(point);
            }
            {
                pcl::PointXYZRGBA point;

                point.x = 0;
                point.y = 0;
                point.z = 1;
                point.r = 0;
                point.g = 0;
                point.b = 255;
                point.a = 0;

                p_cloud->points.push_back(point);
            }
        }

        p_cloud->width = p_cloud->points.size();
        p_cloud->height = 1;

        pcl::io::savePCDFileASCII("registered.pcd", *p_cloud);
    }

    exit(0);
#endif
}

void test_feature_extraction(const model_data &model, const frame_data_t &frame)
{
#if 0
    std::vector<glm::vec3> source_points;
    for (const auto &point : model.clusters.at("lower_leg.R").points)
    {
        source_points.push_back(point.position);
    }

    float max_radius = compute_max_distance(source_points);

    std::vector<triangle_feature> features1;
    for (const auto point : source_points)
    {
        features1.push_back(extract_triangle_feature(point, source_points, max_radius));
    }

    std::vector<triangle_feature> features2;
    for (const auto point : frame.markers)
    {
        features2.push_back(extract_triangle_feature(point, frame.markers, max_radius * 1.5f));
    }

    std::cout << features1.size() << std::endl;
    std::cout << features2.size() << std::endl;

    std::vector<float> dists;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> pairs;
    for (std::size_t i = 0; i < features1.size(); i++)
    {
        for (std::size_t j = 0; j < features2.size(); j++)
        {
            const auto &feature1 = features1[i];
            const auto &feature2 = features2[j];
            dists.push_back(compute_triangle_feature_distance(feature1, feature2));
            pairs.push_back(std::make_tuple(i, j, pairs.size()));
        }
    }

    std::sort(pairs.begin(), pairs.end(), [&](const auto &a, const auto &b)
              { return dists[std::get<2>(a)] < dists[std::get<2>(b)]; });
    for (const auto &[i, j, idx] : pairs)
    {
        std::cout << i << ", " << j << ", " << dists[idx] << std::endl;
    }
#endif
}

static void check_twist(const rigid_cluster &cluster, const find_fit_result &result, const point_cloud &target_points)
{
#if 0
    const auto translation = result.translation;
    const auto rotation = result.rotation;
    auto twist_angle = result.twist_angle;

    const auto anchor_pos = cluster.points[0].position;

    std::vector<weighted_point> rel_source_points;
    for (const auto point : cluster.points)
    {
        rel_source_points.push_back(weighted_point{
            point.position - anchor_pos, point.weight, point.id});
    }

    const auto rel_source_pose = glm::translate(glm::mat4(1.f), -anchor_pos) * cluster.pose;

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    find_nearest_pair(rel_source_points, rel_source_pose, glm::inverse(rel_source_pose), target_points, translation, rotation, pairs, twist_angle);

    for (auto [i, j] : pairs)
    {
        std::cout << i << ", " << j << std::endl;
    }

    const auto error = mse(rel_source_points, rel_source_pose, glm::inverse(rel_source_pose), target_points, translation, rotation, pairs, twist_angle);

    std::cout << error << std::endl;

    const auto error2 = mse(rel_source_points, rel_source_pose, glm::inverse(rel_source_pose), target_points, translation, rotation, pairs, twist_angle + 10.f);
    twist_angle = std::max(cluster.min_twist_angle, std::min(cluster.max_twist_angle, twist_angle - 10.f * (error2 - error) / error));

    const auto error3 = mse(rel_source_points, rel_source_pose, glm::inverse(rel_source_pose), target_points, translation, rotation, pairs, twist_angle);

    std::cout << error2 << std::endl;
    std::cout << error3 << std::endl;
#endif
}