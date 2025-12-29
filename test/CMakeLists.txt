# Find packages
find_package(GTest CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Torch REQUIRED)

enable_testing()

# Test: Solver
add_executable(test_solver
    ${CMAKE_CURRENT_LIST_DIR}/test_solver.cpp
)

target_link_libraries(test_solver
    PRIVATE
    GTest::gtest_main
    marionette
)

add_test(NAME test_solver COMMAND test_solver)

# Test: Autodiff
add_executable(test_autodiff
    ${CMAKE_CURRENT_LIST_DIR}/test_autodiff.cpp
)

target_compile_features(test_autodiff PRIVATE cxx_std_20)

target_link_libraries(test_autodiff
    PRIVATE
    GTest::gtest_main
    marionette
)

add_test(NAME test_autodiff COMMAND test_autodiff)

# Test: Fitting
add_executable(test_fitting
    ${CMAKE_CURRENT_LIST_DIR}/test_fitting.cpp
)

target_link_libraries(test_fitting
    PRIVATE
    marionette
    nlohmann_json::nlohmann_json
)

# Test: Fitting Backward
add_executable(test_fitting_backward
    ${CMAKE_CURRENT_LIST_DIR}/test_fitting_backward.cpp
)

target_link_libraries(test_fitting_backward
    PRIVATE
    marionette
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Detect and Triangulate App
find_package(OpenCV REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

add_executable(test_detect_triangulate
    ${CMAKE_CURRENT_LIST_DIR}/test_detect_triangulate.cpp
)

target_link_libraries(test_detect_triangulate
    PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    onnxruntime::onnxruntime
)
