cmake_minimum_required(VERSION 3.8)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

include_directories(
    SYSTEM ${PCL_INCLUDE_DIRS}
    SYSTEM ${CERES_INCLUDE_DIRS}
)

add_executable(ir_viewer
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sphere_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/grid_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/drawer2d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bone_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/box_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/axis_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/widget_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/playback_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/remote_sensor_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/retarget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer_app.cpp
)

target_compile_definitions(ir_viewer
    PUBLIC
    GLOG_NO_ABBREVIATED_SEVERITIES
    GOOGLE_GLOG_DLL_DECL=
    _MBCS
    GLM_ENABLE_EXPERIMENTAL
)

add_dependencies(ir_viewer
    marionette
    stargazer_sensor_grpc_proto
)

target_include_directories(ir_viewer
    SYSTEM PRIVATE
    ${TINYGLTF_INCLUDE_DIRS}
)

target_link_libraries(ir_viewer
    ceres
    ${OpenCV_LIBS}
    yaml-cpp
    ${PCL_LIBRARIES}
    glfw
    marionette
    stargazer_sensor_grpc_proto
    glad::glad
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    $<$<NOT:$<BOOL:${WIN32}>>:protobuf::libprotobuf>
    $<$<NOT:$<BOOL:${WIN32}>>:gRPC::grpc++_unsecure>
    $<$<NOT:$<BOOL:${WIN32}>>:gRPC::grpc++_reflection>
    nlohmann_json::nlohmann_json
)

target_compile_options(ir_viewer
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter -pg>
    $<$<CONFIG:RELEASE>:-O2>
    $<$<CONFIG:DEBUG>:-g -O0>)

target_sources(ir_viewer
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/viewer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sphere_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/grid_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/drawer2d.hpp
    ${CMAKE_CURRENT_LIST_DIR}/model_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/bone_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/box_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/axis_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/widget_drawer.hpp
)

add_executable(log_viewer_app
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sphere_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/grid_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/drawer2d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bone_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/box_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/axis_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/widget_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/log_viewer_app.cpp
)

target_compile_definitions(log_viewer_app
    PUBLIC
    GLOG_NO_ABBREVIATED_SEVERITIES
    GOOGLE_GLOG_DLL_DECL=
    _MBCS
)

add_dependencies(log_viewer_app
    marionette
    stargazer_sensor_grpc_proto
)

target_link_libraries(log_viewer_app
    ceres
    ${OpenCV_LIBS}
    yaml-cpp
    ${PCL_LIBRARIES}
    glfw
    marionette
    stargazer_sensor_grpc_proto
    glad::glad
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    $<$<NOT:$<BOOL:${WIN32}>>:protobuf::libprotobuf>
    $<$<NOT:$<BOOL:${WIN32}>>:gRPC::grpc++_unsecure>
    $<$<NOT:$<BOOL:${WIN32}>>:gRPC::grpc++_reflection>
)

target_compile_options(log_viewer_app
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter -pg>
    $<$<CONFIG:RELEASE>:-O2>
    $<$<CONFIG:DEBUG>:-g -O0>)

target_sources(log_viewer_app
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/viewer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sphere_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/grid_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/drawer2d.hpp
    ${CMAKE_CURRENT_LIST_DIR}/model_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/bone_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/box_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/axis_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/widget_drawer.hpp
)

add_executable(imu_viewer_app
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sphere_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/grid_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/drawer2d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/model_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/bone_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/box_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/axis_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/widget_drawer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/qprobe_playback_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/remote_sensor_stream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/imu_viewer_app.cpp
)

target_compile_definitions(imu_viewer_app
    PUBLIC
    GLOG_NO_ABBREVIATED_SEVERITIES
    GOOGLE_GLOG_DLL_DECL=
    _MBCS
)

add_dependencies(imu_viewer_app
    marionette
    stargazer_sensor_grpc_proto
)

target_link_libraries(imu_viewer_app
    ceres
    ${OpenCV_LIBS}
    yaml-cpp
    ${PCL_LIBRARIES}
    glfw
    marionette
    stargazer_sensor_grpc_proto
    glad::glad
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>
    $<$<NOT:$<BOOL:${WIN32}>>:protobuf::libprotobuf>
    $<$<NOT:$<BOOL:${WIN32}>>:gRPC::grpc++_unsecure>
    $<$<NOT:$<BOOL:${WIN32}>>:gRPC::grpc++_reflection>
)

target_compile_options(imu_viewer_app
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter -pg>
    $<$<CONFIG:RELEASE>:-O2>
    $<$<CONFIG:DEBUG>:-g -O0>)

target_sources(imu_viewer_app
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/viewer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sphere_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/grid_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/drawer2d.hpp
    ${CMAKE_CURRENT_LIST_DIR}/model_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/bone_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/box_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/axis_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/widget_drawer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/qprobe_playback_stream.hpp
)
