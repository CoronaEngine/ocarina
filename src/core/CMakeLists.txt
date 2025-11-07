file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(ocarina-core SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ocarina-core PUBLIC
        ocarina-ext
        $<$<PLATFORM_ID:Windows>:dbghelp>
        )
target_compile_definitions(ocarina-core PRIVATE OC_CORE_EXPORT_DLL=1)
set_target_properties(ocarina-core PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
		FOLDER "ocarina")
