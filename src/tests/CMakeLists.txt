

function(ocarina_add_test name)
    cmake_parse_arguments(TEST "" "" "SOURCES" ${ARGN})
    add_executable(${name}  ${TEST_SOURCES})
    target_link_libraries(${name} ocarina-native)
    set_target_properties(${name} PROPERTIES
            PUBLIC_HEADER "${HEADER_FILES}"
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            FOLDER "ocarina/test")
    ocarina_copy_multiple_target_outputs(
        DEST_TARGET ${name}
        SOURCE_TARGETS 
        ocarina-ext 
        ocarina-core 
        ocarina-ast
        ocarina-generator
        ocarina-rhi
        ocarina-dsl
        ocarina-GUI
        ocarina-framework
        EASTL
        SDL3-shared
        ocarina-ext-stb
        ocarina-ext-tinyexr
        ocarina-GUI_impl-imGui
        ocarina-backend-cuda
    )
endfunction()


ocarina_add_test(test-callable SOURCES test_callable.cpp)
ocarina_add_test(test-parsetype SOURCES test_parse_type.cpp)
ocarina_add_test(test-kernel SOURCES test_kernel.cpp)
ocarina_add_test(test-imageio SOURCES test_image_io.cpp)
ocarina_add_test(test-accel SOURCES test_accel.cpp)
ocarina_add_test(test-gui SOURCES test_gui.cpp)
ocarina_add_test(test-pt SOURCES test_pt.cpp)
ocarina_add_test(test-raygenkernel SOURCES test_raygenkernel.cpp)
ocarina_add_test(test-doc SOURCES test_doc.cpp)
ocarina_add_test(test-list SOURCES test_list.cpp)
ocarina_add_test(test-read_buffer SOURCES test_read_buffer.cpp)
ocarina_add_test(test-trait SOURCES test_trait.cpp)
ocarina_add_test(test-vulkan-triangle SOURCES test_vulkan_triangle.cpp)
ocarina_add_test(test-vulkan-texture SOURCES test_vulkan_texture.cpp)
