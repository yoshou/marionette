# marionette

Tools for human motion tracking

## Install for Windows

### Install vcpkg

```console
cd third-party
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
vcpkg update
```

### Install dependencies

```console
vcpkg install ceres[eigensparse,lapack]:x64-windows
vcpkg install opencv4[sfm]:x64-windows
vcpkg install boost-core:x64-windows
vcpkg install boost-program-options:x64-windows
vcpkg install boost-graph:x64-windows
vcpkg install yaml-cpp:x64-windows
vcpkg install glad:x64-windows
vcpkg install glfw3:x64-windows
vcpkg install pcl:x64-windows
vcpkg install nanoflann:x64-windows
vcpkg install glm:x64-windows
vcpkg install eigen3:x64-windows
vcpkg install grpc:x64-windows
vcpkg install nlohmann-json:x64-windows
vcpkg install nuklear:x64-windows
vcpkg install tinygltf:x64-windows
```

### Generate project

```console
cmake -G "Visual Studio 17 2022" -A x64 ..
```

## Install for Linux

Please refer to the Dockerfile of devcontainer.
