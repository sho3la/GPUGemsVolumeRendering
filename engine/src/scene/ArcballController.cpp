#include "vve/scene/ArcballController.hpp"

#include "vve/core/Window.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace {
// True when ImGui is using the mouse (hovering a window/widget), so camera
// controls should ignore the event. Valid because ImGui::NewFrame ran last
// frame; one-frame latency is imperceptible.
bool imguiWantsMouse() {
    return ImGui::GetCurrentContext() != nullptr &&
           ImGui::GetIO().WantCaptureMouse;
}
} // namespace

namespace vve::scene {

ArcballController::ArcballController(Camera& camera) : m_camera(camera) {
    update();
}

void ArcballController::attach(core::Window& window) {
    window.onMouseButton = [this](int button, int action, int) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            // Only begin a drag if ImGui isn't handling the click; always allow
            // release so a drag started over the 3D view ends cleanly.
            if (action == GLFW_PRESS) {
                m_dragging = !imguiWantsMouse();
            } else {
                m_dragging = false;
            }
        }
    };
    window.onCursorMove = [this](double x, double y) {
        if (m_dragging) {
            double dx = x - m_lastX;
            double dy = y - m_lastY;
            m_yaw -= static_cast<float>(dx) * 0.01f;
            m_pitch += static_cast<float>(dy) * 0.01f;
            const float limit = glm::radians(89.0f);
            m_pitch = std::clamp(m_pitch, -limit, limit);
            update();
        }
        m_lastX = x;
        m_lastY = y;
    };
    window.onScroll = [this](double, double yoffset) {
        if (imguiWantsMouse()) return;
        m_distance *= std::pow(0.9f, static_cast<float>(yoffset));
        m_distance = std::clamp(m_distance, 0.5f, 20.0f);
        update();
    };
}

void ArcballController::update() {
    glm::vec3 target = m_camera.target();
    glm::vec3 offset{
        m_distance * std::cos(m_pitch) * std::sin(m_yaw),
        m_distance * std::sin(m_pitch),
        m_distance * std::cos(m_pitch) * std::cos(m_yaw)};
    m_camera.setPosition(target + offset);
}

} // namespace vve::scene
