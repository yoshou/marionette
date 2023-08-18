cmake_minimum_required(VERSION 3.8)

add_executable(convert_model
    ${CMAKE_CURRENT_LIST_DIR}/convert_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fbx_loader.cpp
)

target_include_directories(convert_model
    SYSTEM PRIVATE
    $<$<NOT:$<BOOL:${WIN32}>>:~/fbx/include>
    $<$<BOOL:${WIN32}>:C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.2.1/include>
)

target_link_libraries(convert_model
    $<$<NOT:$<BOOL:${WIN32}>>:boost_program_options>
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    $<$<NOT:$<BOOL:${WIN32}>>:xml2>
    $<$<NOT:$<BOOL:${WIN32}>>:~/fbx/lib/gcc/x64/debug/libfbxsdk.so>
    $<$<BOOL:${WIN32}>:Boost::boost>
    $<$<BOOL:${WIN32}>:Boost::program_options>
    $<$<BOOL:${WIN32}>:$<$<CONFIG:RELEASE>:C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.2.1/lib/vs2019/x64/release/libfbxsdk-md.lib>>
    $<$<BOOL:${WIN32}>:$<$<CONFIG:RELEASE>:C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.2.1/lib/vs2019/x64/release/libxml2-md.lib>>
    $<$<BOOL:${WIN32}>:$<$<CONFIG:RELEASE>:C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.2.1/lib/vs2019/x64/release/zlib-md.lib>>
    $<$<BOOL:${WIN32}>:$<$<CONFIG:DEBUG>:C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.2.1/lib/vs2019/x64/debug/libfbxsdk-md.lib>>
    $<$<BOOL:${WIN32}>:$<$<CONFIG:DEBUG>:C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.2.1/lib/vs2019/x64/debug/libxml2-md.lib>>
    $<$<BOOL:${WIN32}>:$<$<CONFIG:DEBUG>:C:/Program\ Files/Autodesk/FBX/FBX\ SDK/2020.2.1/lib/vs2019/x64/debug/zlib-md.lib>>
    nlohmann_json::nlohmann_json
)
