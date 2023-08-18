if(WIN32)
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third-party/vcpkg/installed/x64-windows")

    find_package(OpenCV REQUIRED)
    find_package(Ceres CONFIG REQUIRED)

    set_target_properties(glog::glog PROPERTIES
        INTERFACE_LINK_LIBRARIES "gflags"
    )

    find_package(Boost REQUIRED COMPONENTS program_options)
    find_package(yaml-cpp REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(flann CONFIG REQUIRED)
    find_package(PCL CONFIG REQUIRED)
    find_package(nanoflann CONFIG REQUIRED)
    find_package(unofficial-nuklear CONFIG REQUIRED)
    find_package(glm CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)

    find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")

    include_directories(
        ${CMAKE_SOURCE_DIR}/third-party/vcpkg/installed/x64-include
    )
else()
    include(ExternalProject)

    if(NOT TARGET libjpeg-turbo)
        ExternalProject_Add(libjpeg-turbo
            PREFIX libjpeg-turbo
            GIT_REPOSITORY "https://github.com/libjpeg-turbo/libjpeg-turbo.git"
            GIT_TAG "2.1.0"
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/libjpeg-turbo"
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/libjpeg-turbo"
            "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
            "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        )
    endif()

    if(NOT TARGET glm)
        ExternalProject_Add(glm
            PREFIX glm
            GIT_REPOSITORY https://github.com/g-truc/glm.git
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/glm"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET spdlog)
        ExternalProject_Add(spdlog
            PREFIX spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG "v1.8.2"
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/spdlog"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET cereal)
        ExternalProject_Add(cereal
            PREFIX cereal
            GIT_REPOSITORY https://github.com/USCiLab/cereal.git
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/cereal"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET googletest)
        ExternalProject_Add(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.10.0
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/googletest"
            BINARY_DIR "${CMAKE_BINARY_DIR}/third-party/googletest-build"
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET json)
        ExternalProject_Add(json
            PREFIX json
            GIT_REPOSITORY https://github.com/nlohmann/json
            GIT_TAG v3.10.2
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/json"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET yaml-cpp)
        ExternalProject_Add(yaml-cpp
            PREFIX yaml-cpp
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp
            GIT_TAG yaml-cpp-0.7.0
            SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/yaml-cpp/"
            INSTALL_DIR "${CMAKE_BINARY_DIR}/yaml-cpp"
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/yaml-cpp"
            "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
            "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        )
    endif()

    if(NOT TARGET tinygltf)
        ExternalProject_Add(tinygltf
            PREFIX tinygltf
            GIT_REPOSITORY https://github.com/syoyo/tinygltf
            GIT_TAG v2.5.0
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/tinygltf"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET nuklear)
        ExternalProject_Add(nuklear
            PREFIX nuklear
            GIT_REPOSITORY https://github.com/Immediate-Mode-UI/Nuklear
            GIT_TAG master
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/nuklear"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET nanoflann)
        ExternalProject_Add(nanoflann
            PREFIX nanoflann
            GIT_REPOSITORY https://github.com/jlblancoc/nanoflann
            GIT_TAG v1.3.2
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/nanoflann"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    include_directories(
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/tinygltf
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/nuklear
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/nanoflann/include
        SYSTEM ${CMAKE_BINARY_DIR}/libjpeg-turbo/include
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/glm
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/cereal/include
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/spdlog/include
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/json/include
    )

    find_package(OpenCV 4.0 REQUIRED)
    find_package(realsense2 REQUIRED)
    find_package(Boost 1.36.0 REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(PCL REQUIRED)
    find_package(yaml-cpp REQUIRED)

    find_package(protobuf 3.17 REQUIRED)
    find_package(gRPC REQUIRED)

    include_directories(
        SYSTEM ${OpenCV_INCLUDE_DIRS}
        SYSTEM ${REALSENSE2_INCLUDE_DIRS}
        SYSTEM ${Boost_INCLUDE_DIRS}
        SYSTEM ${EIGEN3_INCLUDE_DIR}
        SYSTEM ${PCL_INCLUDE_DIRS}
        SYSTEM ${PROTOBUF_INCLUDE_DIR}
    )
endif()
