# CopyDllAndPdb.cmake
# 用于拷贝DLL和PDB文件的辅助脚本
# 在构建时由CMake命令调用
#
# 需要的变量:
#   SOURCE_DIR - 源目录
#   DEST_DIR   - 目标目录

if(NOT SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is not defined")
endif()

if(NOT DEST_DIR)
    message(FATAL_ERROR "DEST_DIR is not defined")
endif()

# 确保目标目录存在
file(MAKE_DIRECTORY "${DEST_DIR}")

# 收集所有DLL文件
file(GLOB DLL_FILES "${SOURCE_DIR}/*.dll")

# 收集所有PDB文件
file(GLOB PDB_FILES "${SOURCE_DIR}/*.pdb")

# 拷贝DLL文件
foreach(dll_file ${DLL_FILES})
    get_filename_component(filename ${dll_file} NAME)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${dll_file}"
            "${DEST_DIR}/${filename}"
        RESULT_VARIABLE copy_result
    )
    if(copy_result EQUAL 0)
        message(STATUS "  Copied: ${filename}")
    endif()
endforeach()

# 拷贝PDB文件
foreach(pdb_file ${PDB_FILES})
    get_filename_component(filename ${pdb_file} NAME)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${pdb_file}"
            "${DEST_DIR}/${filename}"
        RESULT_VARIABLE copy_result
    )
    if(copy_result EQUAL 0)
        message(STATUS "  Copied: ${filename}")
    endif()
endforeach()

# 统计拷贝的文件数量
list(LENGTH DLL_FILES dll_count)
list(LENGTH PDB_FILES pdb_count)
message(STATUS "Copied ${dll_count} DLL file(s) and ${pdb_count} PDB file(s) from ${SOURCE_DIR} to ${DEST_DIR}")
