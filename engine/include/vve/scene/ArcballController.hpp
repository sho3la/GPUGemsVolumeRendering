#pragma once

#include "vve/scene/Camera.hpp"

namespace vve::core { class Window; }

namespace vve::scene {

// Orbits a Camera around its target using spherical coordinates. Left-drag
// rotates, scroll zooms. Subscribes to Window input callbacks.
class ArcballController {
public:
    explicit ArcballController(Camera& camera);

    // Wires up window callbacks (cursor, mouse button, scroll).
    void attach(core::Window& window);

    void setDistance(float d) { m_distance = d; update(); }
    void setRadius(float r) { m_distance = r; update(); }
    // Sets the initial orbit angles (radians): yaw around Y, pitch above the XZ
    // plane. Lets an app frame an elongated volume from a chosen side.
    void setOrientation(float yaw, float pitch) {
        m_yaw = yaw;
        m_pitch = pitch;
        update();
    }

private:
    void update();

    Camera& m_camera;
    float m_yaw = 0.0f;      // radians around Y
    float m_pitch = 0.3f;    // radians above the XZ plane
    float m_distance = 3.0f;

    bool m_dragging = false;
    double m_lastX = 0.0;
    double m_lastY = 0.0;
};

} // namespace vve::scene
