#include "vve/core/Paths.hpp"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace vve::core {

std::filesystem::path executableDir() {
    namespace fs = std::filesystem;
#if defined(_WIN32)
    wchar_t buffer[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if (n > 0) return fs::path(buffer).parent_path();
#elif defined(__linux__)
    char buffer[4096] = {};
    ssize_t n = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (n > 0) return fs::path(std::string(buffer, n)).parent_path();
#endif
    return fs::current_path();
}

std::filesystem::path resolveAssetDir(const std::string& relative) {
    namespace fs = std::filesystem;
    fs::path candidates[] = {executableDir() / relative,
                             fs::current_path() / relative};
    for (const auto& c : candidates) {
        if (fs::exists(c)) return c;
    }
    return {};
}

std::filesystem::path dataDir() {
    namespace fs = std::filesystem;
    if (auto d = resolveAssetDir("data"); !d.empty()) return d;
#ifdef VVE_SOURCE_DATA_DIR
    fs::path src = VVE_SOURCE_DATA_DIR;
    if (fs::exists(src)) return src;
#endif
    return {};
}

} // namespace vve::core
