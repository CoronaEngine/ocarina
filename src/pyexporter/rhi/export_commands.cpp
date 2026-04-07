//
// Created by ling.zhu on 2024/11/7.
//

#include "pyexporter/ocapi.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_COMMAND(CMD, ...)  py::class_<CMD, ##__VA_ARGS__>(exporter.module, #CMD);

auto export_command(PythonExporter &exporter) {
    OC_EXPORT_COMMAND(Command)
    OC_EXPORT_COMMAND(BufferCommand, Command)
    OC_EXPORT_COMMAND(DataOpCommand, Command)
    OC_EXPORT_COMMAND(BufferOpCommand, DataOpCommand)
    OC_EXPORT_COMMAND(DataCopyCommand, Command)
    OC_EXPORT_COMMAND(BufferCopyCommand, DataCopyCommand)
    OC_EXPORT_COMMAND(Texture3DCopyCommand, DataCopyCommand)
    OC_EXPORT_COMMAND(BufferUploadCommand, BufferOpCommand)
    OC_EXPORT_COMMAND(BufferToTexture3DCommand, DataCopyCommand)
    OC_EXPORT_COMMAND(BufferDownloadCommand, BufferOpCommand)
    OC_EXPORT_COMMAND(TextureOpCommand, DataOpCommand)
    OC_EXPORT_COMMAND(Texture3DUploadCommand, TextureOpCommand)
    OC_EXPORT_COMMAND(Texture3DDownloadCommand, TextureOpCommand)
    OC_EXPORT_COMMAND(SynchronizeCommand, Command)
    OC_EXPORT_COMMAND(BLASBuildCommand, Command)
    OC_EXPORT_COMMAND(TLASBuildCommand, Command)
    OC_EXPORT_COMMAND(TLASUpdateCommand, Command)
    OC_EXPORT_COMMAND(ShaderDispatchCommand, Command)
}

void export_commands(PythonExporter &exporter) {
    export_command(exporter);
}

#undef OC_EXPORT_COMMAND