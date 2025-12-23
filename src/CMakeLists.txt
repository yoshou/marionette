cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

if (WIN32)
    add_compile_options(/utf-8)
else ()
set(CMAKE_C_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif ()

# find_package(CUDA 11 REQUIRED)
add_library(marionette
    ${CMAKE_CURRENT_LIST_DIR}/optimization/global_registeration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/articulation_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/icp_3d_2d_optimizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/icp_3d_3d_optimizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/model_detector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/motion_tracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/base64.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/serial_port.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/stream_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/playback_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/imu_playback_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/remote_sensor_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pose_estimation/retarget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pose_estimation/pose_computation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/model_detection_worker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/model_tracking_worker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/imu_capture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pose_estimation/finger_pose_estimation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/finger_tracker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/preprocess/fbx_loader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/preprocess/model_converter.cpp
)

target_link_libraries(marionette
    ${CUDA_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    PRIVATE nanoflann::nanoflann
    PRIVATE nlohmann_json::nlohmann_json
    PUBLIC spdlog::spdlog

    # ${CUDA_CUFFT_LIBRARIES}
    # ${CUDA_CUBLAS_LIBRARIES}
    # ${CUDA_cupti_LIBRARY}
    # ${CUDA_curand_LIBRARY}
    # ${CUDA_cusparse_LIBRARY}
    # ${CUDA_npp_LIBRARY}
    # ${CUDA_nppc_LIBRARY}
    # ${CUDA_nppi_LIBRARY}
    # ${CUDA_npps_LIBRARY}
    # ${CUDA_nvcuvenc_LIBRARY}
    # ${CUDA_nvcuvid_LIBRARY}
    # ${PCL_LIBRARIES}
    # ${OpenCV_LIBS}
)

target_compile_definitions(marionette
    PUBLIC
    GLOG_NO_ABBREVIATED_SEVERITIES
    GOOGLE_GLOG_DLL_DECL=
    _MBCS
    GLM_ENABLE_EXPERIMENTAL
    GLOG_USE_GLOG_EXPORT
)

target_include_directories(marionette
    PRIVATE
    SYSTEM ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${CERES_INCLUDE_DIRS}
    SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/core
    ${CMAKE_CURRENT_LIST_DIR}/data_source
    ${CMAKE_CURRENT_LIST_DIR}/tracking
    ${CMAKE_CURRENT_LIST_DIR}/optimization
    ${CMAKE_CURRENT_LIST_DIR}/pose_estimation
    ${CMAKE_CURRENT_LIST_DIR}/utils
    ${CMAKE_CURRENT_LIST_DIR}/preprocess
)

target_sources(marionette
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/optimization/alignment.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/articulation_solver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/automatic_differentiation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/features.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/frame.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/global_registeration.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/icp_3d_2d_optimizer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/icp_3d_3d_optimizer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/model_detector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/model.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/motion_estimation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/nearest_neighbor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/nonlinear_least_square_solver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/nonlinear_solver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/object_functions.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/point_cloud.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/registration.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/srt_transform.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/transform.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/tuple_hash.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/utils.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/motion_tracker.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/base64.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/serial_port.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/stream_server.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/imu_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/playback_stream.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/imu_playback_stream.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/remote_sensor_stream.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pose_estimation/retarget.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/frame_queue.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pose_estimation/pose_computation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/model_detection_worker.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/model_tracking_worker.hpp
    ${CMAKE_CURRENT_LIST_DIR}/data_source/imu_capture.hpp
    ${CMAKE_CURRENT_LIST_DIR}/pose_estimation/finger_pose_estimation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tracking/finger_tracker.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/camera_info.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/glm_json_ext.hpp
    ${CMAKE_CURRENT_LIST_DIR}/optimization/model_optimizer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/preprocess/fbx_model.hpp
    ${CMAKE_CURRENT_LIST_DIR}/preprocess/fbx_loader.hpp
    ${CMAKE_CURRENT_LIST_DIR}/preprocess/model_converter.hpp
)
