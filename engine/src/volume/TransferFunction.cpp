#include "vve/volume/TransferFunction.hpp"

#include <algorithm>
#include <cmath>

namespace vve::volume {

void TransferFunction::addStop(float t, const glm::vec3& color, float opacity) {
    m_stops.push_back({t, color, opacity});
    std::sort(m_stops.begin(), m_stops.end(),
              [](const Stop& a, const Stop& b) { return a.t < b.t; });
}

std::vector<uint8_t> TransferFunction::bakeRGBA8(int width,
                                                 bool premultiply) const {
    std::vector<uint8_t> out(static_cast<size_t>(width) * 4, 0);
    if (m_stops.empty()) return out;

    for (int i = 0; i < width; ++i) {
        float t = static_cast<float>(i) / (width - 1);

        // Find the surrounding stops and linearly interpolate.
        glm::vec3 color = m_stops.front().color;
        float opacity = m_stops.front().opacity;
        if (t <= m_stops.front().t) {
            color = m_stops.front().color;
            opacity = m_stops.front().opacity;
        } else if (t >= m_stops.back().t) {
            color = m_stops.back().color;
            opacity = m_stops.back().opacity;
        } else {
            for (size_t s = 0; s + 1 < m_stops.size(); ++s) {
                const Stop& a = m_stops[s];
                const Stop& b = m_stops[s + 1];
                if (t >= a.t && t <= b.t) {
                    float f = (b.t > a.t) ? (t - a.t) / (b.t - a.t) : 0.0f;
                    color = glm::mix(a.color, b.color, f);
                    opacity = glm::mix(a.opacity, b.opacity, f);
                    break;
                }
            }
        }

        glm::vec3 rgb = premultiply ? color * opacity : color;
        rgb = glm::clamp(rgb, glm::vec3(0.0f), glm::vec3(1.0f));
        out[i * 4 + 0] = static_cast<uint8_t>(std::lround(rgb.r * 255.0f));
        out[i * 4 + 1] = static_cast<uint8_t>(std::lround(rgb.g * 255.0f));
        out[i * 4 + 2] = static_cast<uint8_t>(std::lround(rgb.b * 255.0f));
        out[i * 4 + 3] = static_cast<uint8_t>(
            std::lround(std::clamp(opacity, 0.0f, 1.0f) * 255.0f));
    }
    return out;
}

TransferFunction TransferFunction::grayscaleRamp() {
    TransferFunction tf;
    tf.addStop(0.0f, glm::vec3(0.0f), 0.0f);
    tf.addStop(0.2f, glm::vec3(0.1f), 0.0f);
    tf.addStop(1.0f, glm::vec3(1.0f), 1.0f);
    return tf;
}

TransferFunction TransferFunction::fire() {
    TransferFunction tf;
    tf.addStop(0.0f, glm::vec3(0.0f, 0.0f, 0.0f), 0.0f);
    tf.addStop(0.3f, glm::vec3(0.7f, 0.1f, 0.0f), 0.15f);
    tf.addStop(0.6f, glm::vec3(1.0f, 0.5f, 0.0f), 0.55f);
    tf.addStop(0.85f, glm::vec3(1.0f, 0.9f, 0.4f), 0.8f);
    tf.addStop(1.0f, glm::vec3(1.0f, 1.0f, 1.0f), 1.0f);
    return tf;
}

TransferFunction TransferFunction::coolWarm() {
    TransferFunction tf;
    tf.addStop(0.0f, glm::vec3(0.23f, 0.30f, 0.75f), 0.0f);
    tf.addStop(0.5f, glm::vec3(0.86f, 0.86f, 0.86f), 0.4f);
    tf.addStop(1.0f, glm::vec3(0.71f, 0.02f, 0.15f), 0.9f);
    return tf;
}

} // namespace vve::volume
