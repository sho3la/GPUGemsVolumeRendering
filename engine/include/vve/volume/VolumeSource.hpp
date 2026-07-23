#pragma once

#include "vve/volume/VolumeData.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace vve::volume {

// A selectable list of volumes: the built-in synthetic datasets plus any real
// raw datasets discovered on disk (Open SciVis naming, e.g.
// "bonsai_256x256x256_uint8.raw"). Apps use it to populate their dataset combo
// and to load the chosen volume, so real scan data is available everywhere.
class VolumeSource {
public:
    // Registers the built-in synthetic datasets.
    VolumeSource();

    // Appends any "*_WxHxD_uint8|uint16.raw" files found in `dir`.
    void scanDirectory(const std::filesystem::path& dir);

    [[nodiscard]] int count() const { return static_cast<int>(m_entries.size()); }
    // NUL-free labels for ImGui, as a stable array of C strings.
    [[nodiscard]] const std::vector<std::string>& labels() const {
        return m_labels;
    }

    // Builds the volume at `index`; `syntheticDim` sizes procedural datasets.
    [[nodiscard]] VolumeData create(int index, int syntheticDim = 128) const;

    // Index of the first discovered real (raw file) dataset, or -1 if there are
    // none. Apps use it to default to real data when it is available.
    [[nodiscard]] int firstRealIndex() const;

private:
    enum class Kind { Sphere, MarschnerLobb, Blobs, Tangle, RawFile };
    struct Entry {
        Kind kind;
        std::string label;
        std::filesystem::path path; // RawFile only
        int nx = 0, ny = 0, nz = 0; // RawFile only
        int bytesPerVoxel = 1;      // RawFile only
    };

    void add(Kind kind, std::string label);

    std::vector<Entry> m_entries;
    std::vector<std::string> m_labels;
};

} // namespace vve::volume
