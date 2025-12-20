FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Install system packages required for vcpkg ports
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        libxinerama-dev \
        libxcursor-dev \
        xorg-dev \
        libglu1-mesa-dev \
        libltdl-dev \
        autoconf \
        automake \
        autoconf-archive \
        libtool \
        libtool-bin \
        bison \
        python3-jinja2 \
        python3-venv \
        pkg-config \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

# Update vcpkg to latest version
RUN cd ${VCPKG_ROOT} \
    && git pull \
    && ./bootstrap-vcpkg.sh

# Install vcpkg packages
RUN su vscode -c "${VCPKG_ROOT}/vcpkg install eigen3 glm nlohmann-json nanoflann gtest yaml-cpp tinygltf openfbx glad glfw3 nuklear boost-program-options boost-graph opencv4 ceres[eigensparse,lapack] grpc"

# Add vcpkg tools to PATH
ENV PATH="${VCPKG_ROOT}/installed/x64-linux/tools/protobuf:${VCPKG_ROOT}/installed/x64-linux/tools/grpc:${PATH}"
