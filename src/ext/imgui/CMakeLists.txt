
file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/glad)

add_executable(test_gui ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(test_gui PUBLIC glfw)
target_include_directories(test_gui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/glad)

add_library(ocarina-imgui STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(ocarina-imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/glad)

if(UNIX)
  target_compile_options(ocarina-imgui PRIVATE "-fPIC")
endif()

target_link_libraries(ocarina-imgui PUBLIC glfw)
set_target_properties(ocarina-imgui PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        FOLDER "ocarina")