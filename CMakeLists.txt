cmake_minimum_required(VERSION 3.20)
project(VoiceForgeStudio VERSION 1.3.4 LANGUAGES CXX RC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enforce static C++ runtime linkage (/MT) for zero-dependency Windows portability
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_executable(VoiceForgeStudio WIN32
    main.cpp
    app.rc
)

target_link_libraries(VoiceForgeStudio PRIVATE
    Shell32
    User32
    Kernel32
    Ws2_32
)

# Embed Windows 11 Manifest directly into the compiled executable
set_target_properties(VoiceForgeStudio PROPERTIES
    WIN32_EXECUTABLE TRUE
    LINK_FLAGS "/MANIFEST:EMBED /MANIFESTINPUT:\"${CMAKE_CURRENT_SOURCE_DIR}/app.manifest\""
)
