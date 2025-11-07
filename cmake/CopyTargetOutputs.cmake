# CopyTargetOutputs.cmake
# 提供在构建目标之间拷贝DLL和PDB文件的辅助函数

# 函数: copy_target_outputs
# 功能: 将源目标的输出目录下的所有 *.dll 和 *.pdb 文件拷贝到目标的输出目录
# 参数:
#   SOURCE_TARGET - 源目标名称（提供DLL和PDB文件）
#   DEST_TARGET   - 目标名称（接收DLL和PDB文件）
#
# 用法示例:
#   ocarina_copy_target_outputs(SOURCE_TARGET ocarina-ext DEST_TARGET test-gui)
#
function(ocarina_copy_target_outputs)
    cmake_parse_arguments(COPY "" "SOURCE_TARGET;DEST_TARGET" "" ${ARGN})
    
    if(NOT COPY_SOURCE_TARGET)
        message(FATAL_ERROR "ocarina_copy_target_outputs: SOURCE_TARGET is required")
    endif()
    
    if(NOT COPY_DEST_TARGET)
        message(FATAL_ERROR "ocarina_copy_target_outputs: DEST_TARGET is required")
    endif()
    
    # 确保源目标存在
    if(NOT TARGET ${COPY_SOURCE_TARGET})
        message(FATAL_ERROR "ocarina_copy_target_outputs: SOURCE_TARGET '${COPY_SOURCE_TARGET}' does not exist")
    endif()
    
    # 确保目标目标存在
    if(NOT TARGET ${COPY_DEST_TARGET})
        message(FATAL_ERROR "ocarina_copy_target_outputs: DEST_TARGET '${COPY_DEST_TARGET}' does not exist")
    endif()
    
    # 添加自定义命令，在构建后拷贝文件
    # 使用生成器表达式 $<TARGET_FILE_DIR:target> 自动获取目标的输出目录
    # 使用CMake脚本来处理通配符
    add_custom_command(TARGET ${COPY_DEST_TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying ${COPY_SOURCE_TARGET} output files to ${COPY_DEST_TARGET}..."
        COMMAND ${CMAKE_COMMAND} 
            -DSOURCE_DIR="$<TARGET_FILE_DIR:${COPY_SOURCE_TARGET}>"
            -DDEST_DIR="$<TARGET_FILE_DIR:${COPY_DEST_TARGET}>"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CopyDllAndPdb.cmake"
        COMMENT "Copying DLL and PDB files from ${COPY_SOURCE_TARGET} to ${COPY_DEST_TARGET}"
    )
    
    # 添加依赖关系，确保源目标先构建
    add_dependencies(${COPY_DEST_TARGET} ${COPY_SOURCE_TARGET})

    message(STATUS "Configured: copy output files from ${COPY_SOURCE_TARGET} to ${COPY_DEST_TARGET}")
endfunction()


# 函数: copy_multiple_target_outputs
# 功能: 将多个源目标的输出文件拷贝到一个目标
# 参数:
#   DEST_TARGET     - 目标名称（接收DLL和PDB文件）
#   SOURCE_TARGETS  - 源目标名称列表
#
# 用法示例:
#   ocarina_copy_multiple_target_outputs(
#       DEST_TARGET test-gui 
#       SOURCE_TARGETS ocarina-ext ocarina-core ocarina-gui
#   )
#
function(ocarina_copy_multiple_target_outputs)
    cmake_parse_arguments(COPY "" "DEST_TARGET" "SOURCE_TARGETS" ${ARGN})
    
    if(NOT COPY_DEST_TARGET)
        message(FATAL_ERROR "ocarina_copy_multiple_target_outputs: DEST_TARGET is required")
    endif()
    
    if(NOT COPY_SOURCE_TARGETS)
        message(FATAL_ERROR "ocarina_copy_multiple_target_outputs: SOURCE_TARGETS is required")
    endif()
    
    foreach(source_target ${COPY_SOURCE_TARGETS})
        ocarina_copy_target_outputs(
            SOURCE_TARGET ${source_target}
            DEST_TARGET ${COPY_DEST_TARGET}
        )
    endforeach()
endfunction()
