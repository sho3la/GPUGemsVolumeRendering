#pragma once

#include <filesystem>

namespace vve::core {

// Directory containing the running executable. Assets are resolved relative to
// this so apps work regardless of the current working directory.
std::filesystem::path executableDir();

// Returns the first of the candidate relative subdirectories that exists,
// searched next to the executable and then under the current working directory.
// Returns an empty path if none exist.
std::filesystem::path resolveAssetDir(const std::string& relative);

// Locates the "data" directory holding real volume datasets: next to the
// executable, under the working directory, or (for in-tree dev builds) the
// repository's data/ folder baked in at compile time. Empty if none found.
std::filesystem::path dataDir();

} // namespace vve::core
