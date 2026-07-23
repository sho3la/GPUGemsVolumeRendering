#pragma once

#include <cstdio>
#include <string_view>
#include <utility>

// Minimal printf-style logging. Header-only to keep the dependency surface small.
namespace vve::log {

template <class... Args>
inline void info(std::string_view fmt, Args&&... args) {
    std::fputs("[info] ", stdout);
    std::printf(fmt.data(), std::forward<Args>(args)...);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

template <class... Args>
inline void warn(std::string_view fmt, Args&&... args) {
    std::fputs("[warn] ", stderr);
    std::fprintf(stderr, fmt.data(), std::forward<Args>(args)...);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

template <class... Args>
inline void error(std::string_view fmt, Args&&... args) {
    std::fputs("[error] ", stderr);
    std::fprintf(stderr, fmt.data(), std::forward<Args>(args)...);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

} // namespace vve::log
