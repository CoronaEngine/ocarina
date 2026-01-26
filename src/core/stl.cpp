//
// Created by Zero on 08/06/2022.
//

#include "stl.h"
#include "EASTL/allocator.h"

namespace ocarina {

fs::path parent_path(const fs::path &p,int levels) {
    fs::path cur_path = p;
    for (int i = 0; i < levels; ++i) {
        cur_path = cur_path.parent_path();
    }
    return cur_path;
}

void clear_directory(const std::filesystem::path &dir_path) {
    try {
        if (std::filesystem::exists(dir_path) && std::filesystem::is_directory(dir_path)) {
            for (const auto &entry : std::filesystem::directory_iterator(dir_path)) {
                std::filesystem::remove_all(entry.path());
            }
            std::cout << "Directory cleared: " << dir_path << std::endl;
        } else {
            std::cout << "Directory does not exist: " << dir_path << std::endl;
        }
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error clearing directory: " << e.what() << std::endl;
    }
}

namespace detail{
    void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->allocate(size, alignment, 0u);
}

void allocator_deallocate(void *p, size_t) noexcept {
    eastl::GetDefaultAllocator()->deallocate(p, 0u);
}

void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
    auto &&allocator = eastl::GetDefaultAllocator();
    allocator->deallocate(p, 0u);
    return allocator->allocate(size, alignment, 0u);
}
}
}// namespace ocarina