
file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

pybind11_add_module(ocapi ${SOURCE_FILES})
target_link_libraries(ocapi PUBLIC
        ocarina-native
        vision-window)


