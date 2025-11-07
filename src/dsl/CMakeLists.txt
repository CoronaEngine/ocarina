file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(ocarina-dsl SHARED ${SOURCE_FILES})
set_target_properties(ocarina-dsl PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
        FOLDER "ocarina")
target_compile_definitions(ocarina-dsl PRIVATE OC_DSL_EXPORT_DLL=1)
target_link_libraries(ocarina-dsl
        PUBLIC
        ocarina-rhi
        ocarina-ast)