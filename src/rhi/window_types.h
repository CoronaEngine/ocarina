//
// Created by GitHub Copilot on 2026/4/10.
//

#pragma once

#include "core/stl.h"
#include "math/basic_types.h"

namespace ocarina {

enum WindowLibrary {
    GLFW,
    SDL3,
};

class Window;
using WindowCreator = Window *(const char *name, uint2 initial_size, WindowLibrary library, bool resizable);
using WindowDeleter = void(Window *);
using WindowWrapper = ocarina::unique_ptr<Window, WindowDeleter *>;

}// namespace ocarina