#include "vve/volume/SliceProxyGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace vve::volume::SliceProxyGeometry {

namespace {
// The 12 edges of a box, as pairs of corner indices (bit 0=x,1=y,2=z).
constexpr std::array<std::pair<int, int>, 12> kEdges = {{
    {0, 1}, {2, 3}, {4, 5}, {6, 7}, // x-aligned
    {0, 2}, {1, 3}, {4, 6}, {5, 7}, // y-aligned
    {0, 4}, {1, 5}, {2, 6}, {3, 7}  // z-aligned
}};

glm::vec3 corner(const glm::vec3& mn, const glm::vec3& mx, int i) {
    return {(i & 1) ? mx.x : mn.x,
            (i & 2) ? mx.y : mn.y,
            (i & 4) ? mx.z : mn.z};
}
} // namespace

SliceMesh generateSliced(const glm::vec3& boxMin, const glm::vec3& boxMax,
                         const glm::vec3& sliceDir, int numSlices,
                         bool reverse) {
    SliceMesh mesh;
    if (numSlices < 1) return mesh;
    std::vector<Vertex>& out = mesh.vertices;

    glm::vec3 dir = glm::normalize(sliceDir);
    glm::vec3 boxSize = boxMax - boxMin;

    // Project all corners onto the slice axis to find the sampling range.
    float dMin = 1e30f, dMax = -1e30f;
    for (int i = 0; i < 8; ++i) {
        float d = glm::dot(corner(boxMin, boxMax, i), dir);
        dMin = std::min(dMin, d);
        dMax = std::max(dMax, d);
    }

    // A 2D basis in the slice plane, used to sort intersection points by angle.
    glm::vec3 ref = std::abs(dir.x) < 0.9f ? glm::vec3(1, 0, 0)
                                           : glm::vec3(0, 1, 0);
    glm::vec3 uAxis = glm::normalize(glm::cross(ref, dir));
    glm::vec3 vAxis = glm::cross(dir, uAxis);

    // Sample plane positions at slice centers (avoids the box faces).
    const float spacing = (dMax - dMin) / static_cast<float>(numSlices);

    struct Hit {
        glm::vec3 pos;
        float angle;
    };
    std::vector<Hit> hits;
    hits.reserve(6);

    for (int s = 0; s < numSlices; ++s) {
        int idx = reverse ? (numSlices - 1 - s) : s;
        float d = dMin + (idx + 0.5f) * spacing;

        hits.clear();
        glm::vec3 centroid(0.0f);
        for (const auto& e : kEdges) {
            glm::vec3 a = corner(boxMin, boxMax, e.first);
            glm::vec3 b = corner(boxMin, boxMax, e.second);
            float da = glm::dot(a, dir);
            float db = glm::dot(b, dir);
            float denom = db - da;
            if (std::abs(denom) < 1e-8f) continue;
            float t = (d - da) / denom;
            if (t < 0.0f || t > 1.0f) continue;
            glm::vec3 p = a + t * (b - a);
            hits.push_back({p, 0.0f});
            centroid += p;
        }
        if (hits.size() < 3) continue;
        centroid /= static_cast<float>(hits.size());
        uint32_t sliceFirst = static_cast<uint32_t>(out.size());

        for (auto& h : hits) {
            glm::vec3 rel = h.pos - centroid;
            h.angle = std::atan2(glm::dot(rel, vAxis), glm::dot(rel, uAxis));
        }
        std::sort(hits.begin(), hits.end(),
                  [](const Hit& a, const Hit& b) { return a.angle < b.angle; });

        auto toVertex = [&](const glm::vec3& p) {
            Vertex v;
            v.position = p;
            v.uvw = (p - boxMin) / boxSize;
            return v;
        };

        // Triangle fan around hits[0].
        for (size_t i = 1; i + 1 < hits.size(); ++i) {
            out.push_back(toVertex(hits[0].pos));
            out.push_back(toVertex(hits[i].pos));
            out.push_back(toVertex(hits[i + 1].pos));
        }
        uint32_t sliceCount = static_cast<uint32_t>(out.size()) - sliceFirst;
        if (sliceCount > 0) {
            mesh.firstVertex.push_back(sliceFirst);
            mesh.vertexCount.push_back(sliceCount);
        }
    }

    return mesh;
}

std::vector<Vertex> generate(const glm::vec3& boxMin, const glm::vec3& boxMax,
                             const glm::vec3& sliceDir, int numSlices,
                             bool reverse) {
    return generateSliced(boxMin, boxMax, sliceDir, numSlices, reverse).vertices;
}

} // namespace vve::volume::SliceProxyGeometry
