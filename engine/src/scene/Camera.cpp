#include "vve/scene/Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace vve::scene {

glm::mat4 Camera::view() const {
    return glm::lookAt(m_position, m_target, m_up);
}

glm::mat4 Camera::projection(float aspect) const {
    return glm::perspective(m_fovY, aspect, m_nearZ, m_farZ);
}

glm::vec3 Camera::forward() const {
    return glm::normalize(m_target - m_position);
}

} // namespace vve::scene
