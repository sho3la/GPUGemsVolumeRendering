// Single translation unit that emits the Vulkan Memory Allocator implementation.
#define VMA_IMPLEMENTATION

// VMA is noisy about nullability/unused parameters under MSVC; silence locally.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4127 4189 4324)
#endif

#include <vk_mem_alloc.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
