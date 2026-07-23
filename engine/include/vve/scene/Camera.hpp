#pragma once

#include <glm/glm.hpp>

namespace vve::scene {

// Simple perspective camera. View is derived from an orbit position looking at
// a target; projection uses Vulkan's [0,1] depth range (GLM_FORCE_DEPTH_ZERO).
class Camera {
public:
    [[nodiscard]] glm::mat4 view() const;
    [[nodiscard]] glm::mat4 projection(float aspect) const;
    [[nodiscard]] glm::vec3 position() const { return m_position; }

    void setPosition(const glm::vec3& p) { m_position = p; }
    void setTarget(const glm::vec3& t) { m_target = t; }
    void setFovY(float radians) { m_fovY = radians; }

    [[nodiscard]] glm::vec3 target() const { return m_target; }
    [[nodiscard]] glm::vec3 forward() const;

private:
    glm::vec3 m_position{0.0f, 0.0f, 3.0f};
    glm::vec3 m_target{0.0f, 0.0f, 0.0f};
    glm::vec3 m_up{0.0f, 1.0f, 0.0f};
    float m_fovY = glm::radians(45.0f);
    float m_nearZ = 0.05f;
    float m_farZ = 100.0f;
};

} // namespace vve::scene
