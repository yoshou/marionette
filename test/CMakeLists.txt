find_package(GTest CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

enable_testing()

add_executable(reconstruction_test
    ${CMAKE_CURRENT_LIST_DIR}/test_reconstruction.cpp
)

target_link_libraries(reconstruction_test
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    ceres
    ${OpenCV_LIBS}
    marionette
)

add_test(AllTestsInMain reconstruction_test)

add_executable(optimization_test
    ${CMAKE_CURRENT_LIST_DIR}/test_solver.cpp
)

target_link_libraries(optimization_test
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    ceres
    ${OpenCV_LIBS}
    marionette
)

add_test(AllTestsInMain optimization_test)

add_executable(test_fitting
    ${CMAKE_CURRENT_LIST_DIR}/test_fitting.cpp
)

target_link_libraries(test_fitting
    marionette
)
