#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace vve::volume {

// A 1D transfer function mapping scalar value -> (color, opacity), stored as
// control points and baked into an RGBA lookup texture (GPU Gems §39.4.3).
// Color is premultiplied by opacity ("opacity-weighted color") as the chapter
// requires for correct compositing.
class TransferFunction {
public:
    struct Stop {
        float t;         // position in [0,1]
        glm::vec3 color; // straight (non-premultiplied) RGB
        float opacity;   // alpha in [0,1]
    };

    TransferFunction() = default;

    void clear() { m_stops.clear(); }
    void addStop(float t, const glm::vec3& color, float opacity);
    [[nodiscard]] std::vector<Stop>& stops() { return m_stops; }
    [[nodiscard]] const std::vector<Stop>& stops() const { return m_stops; }

    // Bakes the control points into `width` RGBA8 texels. When `premultiply`
    // is true, RGB is multiplied by A (required for the "over"/"under" operators).
    [[nodiscard]] std::vector<uint8_t> bakeRGBA8(int width = 256,
                                                 bool premultiply = true) const;

    // --- Presets ----------------------------------------------------------
    static TransferFunction grayscaleRamp();
    static TransferFunction fire();      // black-red-yellow-white emission
    static TransferFunction coolWarm();  // diverging scientific colormap
    static TransferFunction bonsai();    // CT bonsai: green foliage, brown trunk
    static TransferFunction ctSkull();   // CT head: translucent skin + grey bone

private:
    std::vector<Stop> m_stops;
};

} // namespace vve::volume
