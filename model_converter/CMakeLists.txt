cmake_minimum_required(VERSION 3.10)

find_package(unofficial-openfbx CONFIG REQUIRED)

add_executable(convert_model
    ${CMAKE_CURRENT_LIST_DIR}/convert_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fbx_loader.cpp
)

target_link_libraries(convert_model
    PRIVATE
    unoffical::openfbx::openfbx
    nlohmann_json::nlohmann_json
)
