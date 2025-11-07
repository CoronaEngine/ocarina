file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(ocarina-GUI_impl INTERFACE)

function(ocarina_add_window_plugin name)

    cmake_parse_arguments(PLUGIN "" "CATEGORY" "SOURCES" ${ARGN})
    set(lib_name ocarina-${PLUGIN_CATEGORY}-${name})
    add_library(${lib_name} MODULE ${PLUGIN_SOURCES})
    target_link_libraries(${lib_name} PRIVATE ocarina-GUI ocarina-rhi)
    set(category ocarina-${PLUGIN_CATEGORY})
    add_dependencies(${category} ${lib_name})

    set_target_properties(${lib_name} PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS OFF
            UNITY_BUILD OFF
            DEBUG_POSTFIX ""
            FOLDER "ocarina")
endfunction()

add_subdirectory(imGui)
