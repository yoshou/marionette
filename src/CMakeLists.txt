cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)

if (WIN32)
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
    ${CMAKE_CURRENT_LIST_DIR}/global_registeration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/articulation_solver.cpp
    ${CMAKE_CURRENT_LIST_DIR}/correspondance.cpp
    ${CMAKE_CURRENT_LIST_DIR}/triangulation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/icp_3d_2d_optimizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/icp_3d_3d_optimizer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model_detector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.cpp
    ${CMAKE_CURRENT_LIST_DIR}/motion_tracker.cpp
)

target_link_libraries(marionette
    ${CUDA_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    PRIVATE nanoflann::nanoflann
    PRIVATE nlohmann_json::nlohmann_json

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
)

target_sources(marionette
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/alignment.hpp
    ${CMAKE_CURRENT_LIST_DIR}/articulation_solver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/automatic_differentiation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/correspondance.hpp
    ${CMAKE_CURRENT_LIST_DIR}/debug.hpp
    ${CMAKE_CURRENT_LIST_DIR}/features.hpp
    ${CMAKE_CURRENT_LIST_DIR}/frame.hpp
    ${CMAKE_CURRENT_LIST_DIR}/global_registeration.hpp
    ${CMAKE_CURRENT_LIST_DIR}/icp_3d_2d_optimizer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/icp_3d_3d_optimizer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/model_detector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/model.hpp
    ${CMAKE_CURRENT_LIST_DIR}/motion_estimation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/nearest_neighbor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/nonlinear_least_square_solver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/nonlinear_solver.hpp
    ${CMAKE_CURRENT_LIST_DIR}/object_functions.hpp
    ${CMAKE_CURRENT_LIST_DIR}/point_cloud.hpp
    ${CMAKE_CURRENT_LIST_DIR}/registration.hpp
    ${CMAKE_CURRENT_LIST_DIR}/srt_transform.hpp
    ${CMAKE_CURRENT_LIST_DIR}/transform.hpp
    ${CMAKE_CURRENT_LIST_DIR}/triangulation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/tuple_hash.hpp
    ${CMAKE_CURRENT_LIST_DIR}/utils.hpp
    ${CMAKE_CURRENT_LIST_DIR}/motion_tracker.hpp
)
