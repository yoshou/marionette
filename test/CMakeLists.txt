find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(reconstruction_test
    ${CMAKE_CURRENT_LIST_DIR}/test_reconstruction.cpp
)

add_dependencies(reconstruction_test
    googletest
)

include_directories(
    SYSTEM ${GTEST_INCLUDE_DIRS}
    SYSTEM ${CERES_INCLUDE_DIR}
    SYSTEM ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(reconstruction_test
    ${OpenCV_LIBS}
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest.a
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest_main.a
    pthread
    glog
    ceres
    opencv_viz
    marionette
)

project(optimization_test)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter -mavx -march=native ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -mavx -march=native ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx -march=native ${CMAKE_CXX_FLAGS_RELEASE}")

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

add_executable(optimization_test
    ${CMAKE_CURRENT_LIST_DIR}/test_solver.cpp
)

add_dependencies(optimization_test
    googletest
)

include_directories(
    SYSTEM ${GTEST_INCLUDE_DIRS}
    SYSTEM ${CERES_INCLUDE_DIR}
    SYSTEM ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(optimization_test
    ${OpenCV_LIBS}
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest.a
    ${CMAKE_BINARY_DIR}/third-party/googletest-build/lib/libgtest_main.a
    pthread
    glog
    ceres
    opencv_viz
)
