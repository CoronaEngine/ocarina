

function(ocarina_add_test name)
    cmake_parse_arguments(TEST "" "" "SOURCES" ${ARGN})
    add_executable(${name}  ${TEST_SOURCES})
    target_link_libraries(${name} ocarina-native)
    set_target_properties(${name} PROPERTIES
            PUBLIC_HEADER "${HEADER_FILES}"
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            FOLDER "ocarina/test")
endfunction()


ocarina_add_test(test-parsetype SOURCES test_parse_type.cpp)
ocarina_add_test(test-precision-caps SOURCES test_precision_caps.cpp)
ocarina_add_test(test-host-bytebuffer SOURCES test_host_byte_buffer.cpp)
ocarina_add_test(test-host-dynamic-buffer SOURCES test_host_dynamic_buffer.cpp)
ocarina_add_test(test-dynamic-buffer SOURCES test_dynamic_buffer.cpp)
ocarina_add_test(test-dynamic-buffer-gpu-bridge SOURCES test_dynamic_buffer_gpu_bridge.cpp)
ocarina_add_test(test-dynamic-buffer-layout-codec SOURCES test_dynamic_buffer_layout_codec.cpp)
ocarina_add_test(test-imageio SOURCES test_image_io.cpp)
ocarina_add_test(test-half SOURCES test_half.cpp)
ocarina_add_test(test-accel-update SOURCES test_accel_update.cpp)
ocarina_add_test(test-function-corrector SOURCES test_function_corrector.cpp)
ocarina_add_test(test-counted_buffer SOURCES test_counted_buffer.cpp)
ocarina_add_test(test-buffer-access-modes SOURCES test_buffer_access_modes.cpp)
ocarina_add_test(test-bytebuffer-access-modes SOURCES test_bytebuffer_access_modes.cpp)
ocarina_add_test(test-texture-access-modes SOURCES test_texture_access_modes.cpp)
ocarina_add_test(test-trait SOURCES test_trait.cpp)
ocarina_add_test(test-soa SOURCES test_soa.cpp)
ocarina_add_test(test-bindless-array SOURCES test_bindless_array.cpp)
ocarina_add_test(test-matrix-inverse SOURCES test_matrix_inverse.cpp)
ocarina_add_test(test-swizzle SOURCES test_swizzle.cpp)
ocarina_add_test(test-texture-readwrite SOURCES test_texture_readwrite.cpp)
ocarina_add_test(test-launch-latency SOURCES test_launch_latency.cpp)
ocarina_add_test(test-printer SOURCES test_printer.cpp)
ocarina_add_test(test-printer-bytebuffer-debug SOURCES test_printer_bytebuffer_debug.cpp)
ocarina_add_test(test-printer-bindless-buffer-debug SOURCES test_printer_bindless_buffer_debug.cpp)
ocarina_add_test(test-atomic SOURCES test_atomic.cpp)
