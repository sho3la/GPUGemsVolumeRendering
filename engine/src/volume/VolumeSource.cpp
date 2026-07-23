#include "vve/volume/VolumeSource.hpp"

#include "vve/core/Log.hpp"

#include <algorithm>
#include <regex>

namespace vve::volume {

VolumeSource::VolumeSource() {
    add(Kind::Sphere, "Sphere (synthetic)");
    add(Kind::MarschnerLobb, "Marschner-Lobb (synthetic)");
    add(Kind::Blobs, "Blobs (synthetic)");
    add(Kind::Tangle, "Tangle (synthetic)");
}

void VolumeSource::add(Kind kind, std::string label) {
    m_entries.push_back(Entry{kind, label, {}, 0, 0, 0, 1});
    m_labels.push_back(std::move(label));
}

void VolumeSource::scanDirectory(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    if (dir.empty() || !fs::exists(dir)) return;

    // e.g. bonsai_256x256x256_uint8.raw  /  skull_256x256x256_uint16.raw
    const std::regex pattern(
        R"(^(.+)_([0-9]+)x([0-9]+)x([0-9]+)_(uint8|uint16)\.raw$)",
        std::regex::icase);

    std::vector<fs::directory_entry> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file()) files.push_back(e);
    }
    std::sort(files.begin(), files.end(),
              [](const auto& a, const auto& b) { return a.path() < b.path(); });

    for (const auto& e : files) {
        std::string name = e.path().filename().string();
        std::smatch m;
        if (!std::regex_match(name, m, pattern)) continue;

        Entry entry;
        entry.kind = Kind::RawFile;
        entry.path = e.path();
        entry.nx = std::stoi(m[2]);
        entry.ny = std::stoi(m[3]);
        entry.nz = std::stoi(m[4]);
        entry.bytesPerVoxel = (m[5].str().find("16") != std::string::npos) ? 2 : 1;
        entry.label = m[1].str() + " (" + m[2].str() + "\xC2\xB3 real)";
        m_entries.push_back(entry);
        m_labels.push_back(entry.label);
        log::info("Found dataset: %s (%dx%dx%d, %d-bit)", name.c_str(), entry.nx,
                  entry.ny, entry.nz, entry.bytesPerVoxel * 8);
    }
}

int VolumeSource::firstRealIndex() const {
    for (int i = 0; i < static_cast<int>(m_entries.size()); ++i) {
        if (m_entries[i].kind == Kind::RawFile) return i;
    }
    return -1;
}

VolumeData VolumeSource::create(int index, int syntheticDim) const {
    if (index < 0 || index >= static_cast<int>(m_entries.size())) {
        return VolumeData::sphere(syntheticDim);
    }
    const Entry& e = m_entries[index];
    switch (e.kind) {
        case Kind::Sphere: return VolumeData::sphere(syntheticDim);
        case Kind::MarschnerLobb: return VolumeData::marschnerLobb(syntheticDim);
        case Kind::Blobs: return VolumeData::blobs(syntheticDim);
        case Kind::Tangle: return VolumeData::tangle(syntheticDim);
        case Kind::RawFile:
            return VolumeData::loadRaw(e.path.string(), e.nx, e.ny, e.nz,
                                       e.bytesPerVoxel);
    }
    return VolumeData::sphere(syntheticDim);
}

} // namespace vve::volume
