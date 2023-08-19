FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-22.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gfortran bison python3 python3-setuptools libarchive-dev libblas-dev liblapack-dev

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config

RUN su vscode -c "${VCPKG_ROOT}/vcpkg install ceres[eigensparse,lapack] opencv4[core,sfm] boost-core boost-program-options boost-graph glm eigen3"
RUN su vscode -c "${VCPKG_ROOT}/vcpkg install grpc yaml-cpp nlohmann-json nuklear tinygltf nanoflann pcl glfw3 glad"

WORKDIR /fbxsdk/

RUN curl -L \
    https://damassets.autodesk.net/content/dam/autodesk/www/adn/fbx/2020-0-1/fbx202001_fbxsdk_linux.tar.gz \
    -o /tmp/fbx202001_fbxsdk_linux.tar.gz

RUN mkdir -p /fbxsdk/install

RUN tar -zxvf /tmp/fbx202001_fbxsdk_linux.tar.gz -C /tmp

RUN echo "yes\nn" | /tmp/fbx202001_fbxsdk_linux /fbxsdk

RUN rm /tmp/fbx202001_fbxsdk_linux.tar.gz
RUN rm -rf /tmp/fbx202001_fbxsdk_linux

RUN chmod -R 755 /fbxsdk

ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/protobuf
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/grpc
