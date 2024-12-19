#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Link HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas-export.h"
#include "hipblas.h"
#include "hipsolver.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include "lbvh.cuh"
//#include "lbvh/bvh.cuh"

#include <random>

struct Ray {
  float4 origin;
  float4 direction;
};

struct Triangle {
  float4 v1, v2, v3;
  int id;
};

struct HitRay {
  float distanceResults;
  int hitResults;
  int idResults;
  float3 intersectionPoint;
};

struct aabb_getter {
  __device__ __inline__ lbvh::aabb<float>
  operator()(const Triangle &tri) const noexcept {
    lbvh::aabb<float> retval;
    retval.lower =
        make_float4(fminf(fminf(tri.v1.x, tri.v2.x), tri.v3.x),
                    fminf(fminf(tri.v1.y, tri.v2.y), tri.v3.y),
                    fminf(fminf(tri.v1.z, tri.v2.z), tri.v3.z), 0.0f);
    retval.upper =
        make_float4(fmaxf(fmaxf(tri.v1.x, tri.v2.x), tri.v3.x),
                    fmaxf(fmaxf(tri.v1.y, tri.v2.y), tri.v3.y),
                    fmaxf(fmaxf(tri.v1.z, tri.v2.z), tri.v3.z), 0.0f);
    return retval;
  }
};

struct AABB {
  float4 min;
  float4 max;
};

struct merge_aabb {
  __device__ __inline__ lbvh::aabb<float>
  operator()(const lbvh::aabb<float> &a, const lbvh::aabb<float> &b) const {
    return lbvh::merge(a, b);
  }
};

struct CenterGlobalSpaceBox {
  float4 min;
  float4 max;
  float width;
  float height;
  float depth;
  float radius;
  float volume;
  float4 position;
};

__host__ __device__ __inline__ float length(const float3 &v) {
  return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ __inline__ float length(const float4 &v) {
  return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ __inline__ float angleScalar(const float4 v1,
                                                 const float4 v2) {
  float p = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
  float n1 = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
  float n2 = sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
  float d = n1 * n2;
  if (d > 0.0f) {
    float r = p / d;
    r = fmaxf(-1.0f, fminf(1.0f, r)); // Clamp r to [-1, 1]
    return acosf(r);
  }
  return 0.0f;
}

__host__ __device__ __inline__ float
calculateHalfOpeningAngle(const Triangle &triangle, const float4 &origin) {
  // This function will be used to speed up the calculations and will adapt the
  // limit angle of the sameDirection function
  float4 barycenter = {(triangle.v1.x + triangle.v2.x + triangle.v3.x) / 3.0f,
                       (triangle.v1.y + triangle.v2.y + triangle.v3.y) / 3.0f,
                       (triangle.v1.z + triangle.v2.z + triangle.v3.z) / 3.0f,
                       0.0f};

  float distance =
      sqrt(pow(barycenter.x - origin.x, 2) + pow(barycenter.y - origin.y, 2) +
           pow(barycenter.z - origin.z, 2));

  float4 edge1 = {triangle.v2.x - triangle.v1.x, triangle.v2.y - triangle.v1.y,
                  triangle.v2.z - triangle.v1.z, 0.0f};

  float4 edge2 = {triangle.v3.x - triangle.v1.x, triangle.v3.y - triangle.v1.y,
                  triangle.v3.z - triangle.v1.z, 0.0f};

  float4 cross = {edge1.y * edge2.z - edge1.z * edge2.y,
                  edge1.z * edge2.x - edge1.x * edge2.z,
                  edge1.x * edge2.y - edge1.y * edge2.x, 0.0f};

  float area =
      0.5f * sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
  float solidAngle = area / (distance * distance);
  float halfOpeningAngle = asin(sqrt(solidAngle / (4 * M_PI)));

  return halfOpeningAngle;
}

__host__ __device__ __inline__ bool
sameDirection(const Triangle &tri, const Ray &ray,
              const float &angleLim) { // To be modified soon according to the
                                       // radius of the triangle object
  float4 dT;
  dT.x = (tri.v1.x + tri.v2.x + tri.v3.x) / 3.0f - ray.origin.x;
  dT.y = (tri.v1.y + tri.v2.y + tri.v3.y) / 3.0f - ray.origin.y;
  dT.z = (tri.v1.z + tri.v2.z + tri.v3.z) / 3.0f - ray.origin.z;
  float angle1 = angleScalar(dT, ray.direction);
  float angle2 = calculateHalfOpeningAngle(tri, ray.origin);
  // return (angle <= angleLim);
  return (angle1 <= angleLim) && (angle1 <= angle2);
}

__host__ __device__ __inline__ bool
sameDirectionTest(const Triangle &tri, const Ray &ray, const float &angleLim) {
  float4 dT1 = tri.v1 - ray.origin;
  float4 dT2 = tri.v2 - ray.origin;
  float4 dT3 = tri.v3 - ray.origin;
  float4 dT4;
  dT3.x = (tri.v1.x + tri.v2.x + tri.v3.x) / 3.0f - ray.origin.x;
  dT3.y = (tri.v1.y + tri.v2.y + tri.v3.y) / 3.0f - ray.origin.y;
  dT3.z = (tri.v1.z + tri.v2.z + tri.v3.z) / 3.0f - ray.origin.z;

  bool b1 = (fabs(angleScalar(dT1, ray.direction)) <= angleLim);
  bool b2 = (fabs(angleScalar(dT2, ray.direction)) <= angleLim);
  bool b3 = (fabs(angleScalar(dT3, ray.direction)) <= angleLim);
  bool b4 = (fabs(angleScalar(dT4, ray.direction)) <= angleLim);
  return (b1 || b2 || b3 || b4);
}

struct distance_calculator {
  __device__ __inline__ float operator()(const float4 point,
                                         const Triangle &tri) const noexcept {
    float4 center = make_float4((tri.v1.x + tri.v2.x + tri.v3.x) / 3.0f,
                                (tri.v1.y + tri.v2.y + tri.v3.y) / 3.0f,
                                (tri.v1.z + tri.v2.z + tri.v3.z) / 3.0f, 0.0f);
    return (point.x - center.x) * (point.x - center.x) +
           (point.y - center.y) * (point.y - center.y) +
           (point.z - center.z) * (point.z - center.z);
  }
};

struct distance_calculator2 {
  __device__ __inline__ float operator()(const float4 point,
                                         const Triangle &tri) const noexcept {
    // Function that will be used to resolve the problems of the vertices at the
    // edge of the triangle.
    float4 edge1 = tri.v2 - tri.v1;
    float4 edge2 = tri.v3 - tri.v1;
    float3 normal = make_float3(edge1.y * edge2.z - edge1.z * edge2.y,
                                edge1.z * edge2.x - edge1.x * edge2.z,
                                edge1.x * edge2.y - edge1.y * edge2.x);
    float normLength =
        sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (normLength > 0) {
      normal.x /= normLength;
      normal.y /= normLength;
      normal.z /= normLength;
    }

    float D =
        -(normal.x * tri.v1.x + normal.y * tri.v1.y + normal.z * tri.v1.z);
    float distance =
        fabs(normal.x * point.x + normal.y * point.y + normal.z * point.z + D);
    return distance;
  }
};

struct distance_calculator3 {
  __device__ __inline__ float
  operator()(const float4 &point, const Triangle &triangle) const noexcept {
    float4 edge1 = {triangle.v2.x - triangle.v1.x,
                    triangle.v2.y - triangle.v1.y,
                    triangle.v2.z - triangle.v1.z, 0.0f};
    float4 edge2 = {triangle.v3.x - triangle.v1.x,
                    triangle.v3.y - triangle.v1.y,
                    triangle.v3.z - triangle.v1.z, 0.0f};
    float4 v0 = {triangle.v1.x - point.x, triangle.v1.y - point.y,
                 triangle.v1.z - point.z, 0.0f};

    float a = edge1.x * edge1.x + edge1.y * edge1.y + edge1.z * edge1.z;
    float b = edge1.x * edge2.x + edge1.y * edge2.y + edge1.z * edge2.z;
    float c = edge2.x * edge2.x + edge2.y * edge2.y + edge2.z * edge2.z;
    float d = edge1.x * v0.x + edge1.y * v0.y + edge1.z * v0.z;
    float e = edge2.x * v0.x + edge2.y * v0.y + edge2.z * v0.z;

    float det = a * c - b * b;
    float s = b * e - c * d;
    float t = b * d - a * e;

    if (s + t <= det) {
      if (s < 0.0f) {
        if (t < 0.0f) {
          // Région 4
          if (d < 0.0f) {
            t = 0.0f;
            if (-d >= a) {
              s = 1.0f;
            } else {
              s = -d / a;
            }
          } else {
            s = 0.0f;
            if (e >= 0.0f) {
              t = 0.0f;
            } else if (-e >= c) {
              t = 1.0f;
            } else {
              t = -e / c;
            }
          }
        } else {
          // Région 3
          s = 0.0f;
          if (e >= 0.0f) {
            t = 0.0f;
          } else if (-e >= c) {
            t = 1.0f;
          } else {
            t = -e / c;
          }
        }
      } else if (t < 0.0f) {
        // Région 5
        t = 0.0f;
        if (d >= 0.0f) {
          s = 0.0f;
        } else if (-d >= a) {
          s = 1.0f;
        } else {
          s = -d / a;
        }
      } else {
        // Région 0
        float invDet = 1.0f / det;
        s *= invDet;
        t *= invDet;
      }
    } else {
      if (s < 0.0f) {
        // Région 2
        float tmp0 = b + d;
        float tmp1 = c + e;
        if (tmp1 > tmp0) {
          float numer = tmp1 - tmp0;
          float denom = a - 2.0f * b + c;
          if (numer >= denom) {
            s = 1.0f;
            t = 0.0f;
          } else {
            s = numer / denom;
            t = 1.0f - s;
          }
        } else {
          s = 0.0f;
          if (tmp1 <= 0.0f) {
            t = 1.0f;
          } else if (e >= 0.0f) {
            t = 0.0f;
          } else {
            t = -e / c;
          }
        }
      } else if (t < 0.0f) {
        // Région 6
        float tmp0 = b + e;
        float tmp1 = a + d;
        if (tmp1 > tmp0) {
          float numer = tmp1 - tmp0;
          float denom = a - 2.0f * b + c;
          if (numer >= denom) {
            t = 1.0f;
            s = 0.0f;
          } else {
            t = numer / denom;
            s = 1.0f - t;
          }
        } else {
          t = 0.0f;
          if (tmp1 <= 0.0f) {
            s = 1.0f;
          } else if (d >= 0.0f) {
            s = 0.0f;
          } else {
            s = -d / a;
          }
        }
      } else {
        // Région 1
        float numer = c + e - b - d;
        if (numer <= 0.0f) {
          s = 0.0f;
          t = 1.0f;
        } else {
          float denom = a - 2.0f * b + c;
          if (numer >= denom) {
            s = 1.0f;
            t = 0.0f;
          } else {
            s = numer / denom;
            t = 1.0f - s;
          }
        }
      }
    }

    float4 closestPoint = {triangle.v1.x + s * edge1.x + t * edge2.x,
                           triangle.v1.y + s * edge1.y + t * edge2.y,
                           triangle.v1.z + s * edge1.z + t * edge2.z, 0.0f};

    float dx = point.x - closestPoint.x;
    float dy = point.y - closestPoint.y;
    float dz = point.z - closestPoint.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
  }
};

__host__ __device__ __inline__ void normalizeRayDirection(Ray &ray) {
  float len = sqrtf(ray.direction.x * ray.direction.x +
                    ray.direction.y * ray.direction.y +
                    ray.direction.z * ray.direction.z);
  if (len > 0) {
    float invLen = 1.0f / len;
    ray.direction.x *= invLen;
    ray.direction.y *= invLen;
    ray.direction.z *= invLen;
  }
}

__device__ __inline__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ __inline__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __inline__ float3 float4_to_float3(const float4 &v) {
  return make_float3(v.x, v.y, v.z);
}

__device__ __inline__ bool computeBarycentricCoordinates(const float4 &P,
                                                         const Triangle &tri,
                                                         float &u, float &v) {
  float4 v0 = tri.v2 - tri.v1;
  float4 v1 = tri.v3 - tri.v1;
  float4 v2 = P - tri.v1;

  float d00 = dot(float4_to_float3(v0), float4_to_float3(v0));
  float d01 = dot(float4_to_float3(v0), float4_to_float3(v1));
  float d11 = dot(float4_to_float3(v1), float4_to_float3(v1));
  float d20 = dot(float4_to_float3(v2), float4_to_float3(v0));
  float d21 = dot(float4_to_float3(v2), float4_to_float3(v1));

  float denom = d00 * d11 - d01 * d01;
  if (denom == 0)
    return false;

  u = (d11 * d20 - d01 * d21) / denom;
  v = (d00 * d21 - d01 * d20) / denom;

  return (u >= 0 && v >= 0 &&
          (u + v) <= 1); // Check if the point is inside or on the edge
}

__device__ __inline__ bool
raySphereIntersection(const float4 &rayOrigin, const float4 &rayDirection,
                      const float4 &sphereCenter, const float sphereRadius,
                      float4 &intersectionPoint, float &distance) {
  // Calculate oc (rayOrigin - sphereCenter)
  float4 oc =
      make_float4(rayOrigin.x - sphereCenter.x, rayOrigin.y - sphereCenter.y,
                  rayOrigin.z - sphereCenter.z, 0.0f);

  // Calculate a, b, c for the quadratic equation
  float a = oc.x * oc.x + oc.y * oc.y +
            oc.z * oc.z; // Incorrect usage, corrected below
  float a_correct = rayDirection.x * rayDirection.x +
                    rayDirection.y * rayDirection.y +
                    rayDirection.z * rayDirection.z;
  float b = 2.0f * (oc.x * rayDirection.x + oc.y * rayDirection.y +
                    oc.z * rayDirection.z);
  float c =
      oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - sphereRadius * sphereRadius;

  float discriminant = b * b - 4 * a_correct * c;

  if (discriminant < 0) {
    return false; // No intersection
  }

  float sqrtDiscriminant = sqrtf(discriminant);
  float t1 = (-b - sqrtDiscriminant) / (2.0f * a_correct);
  float t2 = (-b + sqrtDiscriminant) / (2.0f * a_correct);

  // Choose the smallest positive distance
  if (t1 > 0 && t1 < t2) {
    distance = t1;
  } else if (t2 > 0) {
    distance = t2;
  } else {
    return false; // Both intersections are behind the ray
  }

  // Calculate the intersection point
  intersectionPoint =
      make_float4(rayOrigin.x + distance * rayDirection.x,
                  rayOrigin.y + distance * rayDirection.y,
                  rayOrigin.z + distance * rayDirection.z, 0.0f);

  return true;
}

__device__ __inline__ bool
rayTriangleIntersect(const Ray &ray, const Triangle &triangle, float &t) {
  float4 edge1 = triangle.v2 - triangle.v1;
  float4 edge2 = triangle.v3 - triangle.v1;
  float4 h =
      make_float4(ray.direction.y * edge2.z - ray.direction.z * edge2.y,
                  ray.direction.z * edge2.x - ray.direction.x * edge2.z,
                  ray.direction.x * edge2.y - ray.direction.y * edge2.x, 0);
  float a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;
  if (a > -1e-8 && a < 1e-8)
    return false;
  float f = 1.0f / a;
  float4 s = ray.origin - triangle.v1;
  float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
  if (u < 0.0 || u > 1.0)
    return false;
  float4 q =
      make_float4(s.y * edge1.z - s.z * edge1.y, s.z * edge1.x - s.x * edge1.z,
                  s.x * edge1.y - s.y * edge1.x, 0);
  float v = f * (ray.direction.x * q.x + ray.direction.y * q.y +
                 ray.direction.z * q.z);
  if (v < 0.0 || u + v > 1.0)
    return false;
  t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);
  return (t > 1e-8);
}

template <typename T, typename U>
__global__ void process_single_point_new(lbvh::bvh_device<T, U> bvh_dev,
                                         float4 pos) {
  const auto calc = distance_calculator();
  const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);

  if (nest.first != 0xFFFFFFFF) {
    printf("Nearest object index: %u\n", nest.first);
    printf("Distance to nearest object: %f\n", nest.second);

    // Display the coordinates of the nearest object
    const auto &nearest_object = bvh_dev.objects[nest.first];
    printf("Nearest object coordinates:\n");
    printf("  v1: (%f, %f, %f)\n", nearest_object.v1.x, nearest_object.v1.y,
           nearest_object.v1.z);
    printf("  v2: (%f, %f, %f)\n", nearest_object.v2.x, nearest_object.v2.y,
           nearest_object.v2.z);
    printf("  v3: (%f, %f, %f)\n", nearest_object.v3.x, nearest_object.v3.y,
           nearest_object.v3.z);

    // Display the coordinates of the query point
    printf("Query point coordinates: (%f, %f, %f)\n", pos.x, pos.y, pos.z);
  } else {
    printf("No nearest object found (BVH might be empty)\n");
  }
}

template <typename T, typename U>
__global__ void rayTracingKernel(lbvh::bvh_device<T, U> bvh_dev, Ray ray,
                                 float4 *result) {
  const auto calc = distance_calculator();

  bool isView = true;

  normalizeRayDirection(ray);
  // Point along the ray at a unit distance from the origin
  float epsilon = 0.001f;
  float4 pos =
      ray.origin +
      ray.direction *
          epsilon; // un petit décalage por chercher dans la bonne direction

  // Use query_device to find the closest object
  // auto query = lbvh::nearest(query_point).set_max_depth(max_depth);
  const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);

  if (nest.first != 0xFFFFFFFF) {
    // An object has been found
    const auto &hit_triangle = bvh_dev.objects[nest.first];
    if (isView) {
      printf("Nearest object index: %u\n", nest.first);
      printf("Distance to nearest object: %f\n", nest.second);
      printf("v1=%f %f %f\n", hit_triangle.v1.x, hit_triangle.v1.y,
             hit_triangle.v1.z);
      printf("v2=%f %f %f\n", hit_triangle.v2.x, hit_triangle.v2.y,
             hit_triangle.v2.z);
      printf("v3=%f %f %f\n", hit_triangle.v3.x, hit_triangle.v3.y,
             hit_triangle.v3.z);
    }

    // Calculate the intersection point
    float t;
    if (rayTriangleIntersect(ray, hit_triangle, t)) {
      float4 hit_point = ray.origin + ray.direction * t;
      *result = hit_point;
      if (isView) {
        printf("Ray hit triangle %d at point (%f, %f, %f)\n", nest.first,
               hit_point.x, hit_point.y, hit_point.z);
        printf("t=%f\n", t);
      }
    } else {
      *result = make_float4(INFINITY, INFINITY, INFINITY, INFINITY);
      if (isView)
        printf("Nearest object found but not intersected by ray\n");
    }

  } else {
    // No items found
    *result = make_float4(INFINITY, INFINITY, INFINITY, INFINITY);
    if (isView)
      printf("Ray did not hit any triangle\n");
  }
}

template <typename T, typename U>
__global__ void rayTracingKernel(lbvh::bvh_device<T, U> bvh_dev, Ray *rays,
                                 HitRay *d_HitRays, int numRays) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  bool isView = true;
  // isView = false;

  Ray ray = rays[idx];
  const auto calc = distance_calculator();

  // Point along the ray at a unit distance from the origin
  float4 pos = ray.origin + ray.direction;

  // Use query_device to find the closest object
  const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);

  // Initialization of results
  d_HitRays[idx].hitResults = -1;
  d_HitRays[idx].distanceResults = INFINITY; // distance
  d_HitRays[idx].intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  d_HitRays[idx].idResults = -1;

  if (nest.first != 0xFFFFFFFF) {
    // An object has been found
    const auto &hit_triangle = bvh_dev.objects[nest.first];

    if (isView) {
      printf(
          "oRay %d: Nearest object index: %u Distance to nearest object: %f\n",
          idx, nest.first, nest.second);
      // printf("oRay %d: v1=%f %f %f\n", idx, hit_triangle.v1.x,
      // hit_triangle.v1.y, hit_triangle.v1.z); printf("oRay %d: v2=%f %f %f\n",
      // idx, hit_triangle.v2.x, hit_triangle.v2.y, hit_triangle.v2.z);
      // printf("oRay %d: v3=%f %f %f\n", idx, hit_triangle.v3.x,
      // hit_triangle.v3.y, hit_triangle.v3.z);
    }

    // Calculate the intersection point
    float t;
    if (rayTriangleIntersect(ray, hit_triangle, t)) {
      float4 hit_point = ray.origin + ray.direction * t;
      if (isView) {
        printf("Ray %d hit triangle %d at point (%f, %f, %f) Distance:%f\n",
               idx, nest.first, hit_point.x, hit_point.y, hit_point.z, t);
      }
      d_HitRays[idx].hitResults = nest.first;
      d_HitRays[idx].distanceResults = t; // distance
      d_HitRays[idx].intersectionPoint =
          make_float3(hit_point.x, hit_point.y, hit_point.z);
      d_HitRays[idx].idResults = hit_triangle.id;
    } else {
      if (isView)
        printf("Ray %d: Nearest object found but not intersected by ray\n",
               idx);
    }
  } else {
    // No items found
    if (isView)
      printf("Ray %d did not hit any triangle\n", idx);
  }
}

__device__ __inline__ float dot(const float4 &a, const float4 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __inline__ float4 cross(const float4 &a, const float4 &b) {
  return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x, 0.0f);
}

__device__ __inline__ bool pointInTriangle(const float4 &point,
                                           const Triangle &triangle) {
  float4 v0 = triangle.v3 - triangle.v1;
  float4 v1 = triangle.v2 - triangle.v1;
  float4 v2 = point - triangle.v1;

  float dot00 = dot(v0, v0);
  float dot01 = dot(v0, v1);
  float dot02 = dot(v0, v2);
  float dot11 = dot(v1, v1);
  float dot12 = dot(v1, v2);

  float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
  float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  return (u >= 0) && (v >= 0) && (u + v <= 1);
}

__device__ __inline__ bool pointInTriangle2(const float4 &point,
                                            const Triangle &triangle,
                                            float epsilon = 1e-5f) {
  float4 v0 = triangle.v2 - triangle.v1;
  float4 v1 = triangle.v3 - triangle.v1;
  float4 v2 = point - triangle.v1;
  bool isView = true;
  isView = false;

  // Check if point lies on the plane of the triangle
  float4 normal = cross(v0, v1);
  float distanceToPlane = dot(normal, v2) / length(normal);

  if (isView)
    printf("DistanceToPlan: %f\n", distanceToPlane);

  if (abs(distanceToPlane) > epsilon) {
    return false; // Point does not lie on the plane
  }

  float d00 = dot(v0, v0);
  float d01 = dot(v0, v1);
  float d11 = dot(v1, v1);
  float d20 = dot(v2, v0);
  float d21 = dot(v2, v1);

  float denom = d00 * d11 - d01 * d01;
  float v = (d11 * d20 - d01 * d21) / denom;
  float w = (d00 * d21 - d01 * d20) / denom;
  float u = 1.0f - v - w;
  if (isView) {
    printf("Point: (%f, %f, %f)\n", point.x, point.y, point.z);
    printf("Triangle: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n",
           triangle.v1.x, triangle.v1.y, triangle.v1.z, triangle.v2.x,
           triangle.v2.y, triangle.v2.z, triangle.v3.x, triangle.v3.y,
           triangle.v3.z);
    printf("u: %f, v: %f, w: %f\n", u, v, w);
  }

  bool result = (u >= -epsilon) && (v >= -epsilon) && (w >= -epsilon) &&
                (u + v + w <= 1.0f + epsilon);
  if (isView)
    printf("Result: %s\n", result ? "true" : "false");

  return result;
}

__device__ __inline__ bool pointInTriangle(const float4 &point,
                                           const Triangle &triangle,
                                           float3 &intersectionPoint) {
  float4 v0 = triangle.v3 - triangle.v1;
  float4 v1 = triangle.v2 - triangle.v1;
  float4 v2 = point - triangle.v1;

  float dot00 = dot(v0, v0);
  float dot01 = dot(v0, v1);
  float dot02 = dot(v0, v2);
  float dot11 = dot(v1, v1);
  float dot12 = dot(v1, v2);

  float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
  float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  bool isInside = (u >= 0) && (v >= 0) && (u + v <= 1);

  if (isInside) {
    // Calculer le point d'intersection en utilisant les coordonnées
    // barycentriques
    float w = 1.0f - u - v;
    intersectionPoint.x =
        w * triangle.v1.x + u * triangle.v2.x + v * triangle.v3.x;
    intersectionPoint.y =
        w * triangle.v1.y + u * triangle.v2.y + v * triangle.v3.y;
    intersectionPoint.z =
        w * triangle.v1.z + u * triangle.v2.z + v * triangle.v3.z;
  }

  return isInside;
}

__device__ __inline__ bool pointInTriangle(const float4 &point,
                                           const Triangle &triangle,
                                           float3 &intersectionPoint,
                                           float tolerance = 1e-5f) {
  float4 v0 = triangle.v3 - triangle.v1;
  float4 v1 = triangle.v2 - triangle.v1;
  float4 v2 = point - triangle.v1;

  float dot00 = dot(v0, v0);
  float dot01 = dot(v0, v1);
  float dot02 = dot(v0, v2);
  float dot11 = dot(v1, v1);
  float dot12 = dot(v1, v2);

  float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
  float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  bool isInside =
      (u >= -tolerance) && (v >= -tolerance) && (u + v <= 1.0f + tolerance);

  if (isInside) {
    float w = 1.0f - u - v;
    u = fmaxf(0.0f, fminf(1.0f, u));
    v = fmaxf(0.0f, fminf(1.0f, v));
    w = fmaxf(0.0f, fminf(1.0f, w));

    float sum = u + v + w;
    u /= sum;
    v /= sum;
    w /= sum;

    intersectionPoint.x =
        w * triangle.v1.x + u * triangle.v2.x + v * triangle.v3.x;
    intersectionPoint.y =
        w * triangle.v1.y + u * triangle.v2.y + v * triangle.v3.y;
    intersectionPoint.z =
        w * triangle.v1.z + u * triangle.v2.z + v * triangle.v3.z;
  }

  return isInside;
}

__global__ void checkPointInTriangles(float4 point, Triangle *triangles,
                                      int numTriangles, int *result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    if (pointInTriangle(point, triangles[idx])) {
      atomicAdd(result, 1);
      // result = idx;
      return;
    }
  }
}

__device__ __inline__ float4 normalize(float4 v) {
  float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  if (length > 0) {
    float invLength = 1.0f / length;
    v.x *= invLength;
    v.y *= invLength;
    v.z *= invLength;
  }
  return v;
}

//**************************************************************************************************************
//--------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------
//**************************************************************************************************************

//**************************************************************************************************************
//--------------------------------------------------------------------------------------------------------------

template <typename T, typename U>
__global__ void rayTracingKernelExploration002(lbvh::bvh_device<T, U> bvh_dev,
                                               Ray *rays, HitRay *d_HitRays,
                                               int numRays) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  // Load ray data into registers for faster access
  Ray ray = rays[idx];

  // Initialize hit results in registers to reduce global memory writes
  HitRay hitResult;
  hitResult.hitResults = -1;
  hitResult.distanceResults = INFINITY;
  hitResult.intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  hitResult.idResults = -1;

  constexpr float epsilon = 0.001f;
  constexpr float epsilonC = 0.01f;
  constexpr int maxIterations = 50;

  // Precompute directions array in shared memory for better performance
  __shared__ float4 directions[14];
  if (threadIdx.x < 14) {
    directions[threadIdx.x] = make_float4(
        (threadIdx.x % 2 == 0 ? 1.0f : -1.0f) * ((threadIdx.x / 6) % 2),
        (threadIdx.x % 6 == 0 ? -1.0f : (threadIdx.x / 2) % 2),
        (threadIdx.x % 3 == 0 ? -1.0f : (threadIdx.x / 3) % 2), 0.0f);
  }
  __syncthreads();

  // Step 1: Initial exploration
  for (int i = 0; i < 14; ++i) {
    float4 currentPosition = ray.origin + directions[i] * epsilonC;
    const auto nearestTriangleIndex = lbvh::query_device(
        bvh_dev, lbvh::nearest(currentPosition), distance_calculator());

    if (nearestTriangleIndex.first != 0xFFFFFFFF) {
      const Triangle &hitTriangle = bvh_dev.objects[nearestTriangleIndex.first];
      if (pointInTriangle2(currentPosition, hitTriangle, 0.0001f)) {
        float t;
        if (rayTriangleIntersect(ray, hitTriangle, t)) {
          float4 hit_point = ray.origin + ray.direction * t;
          hitResult.hitResults = nearestTriangleIndex.first;
          hitResult.distanceResults = t;
          hitResult.intersectionPoint =
              make_float3(hit_point.x, hit_point.y, hit_point.z);
          hitResult.idResults = hitTriangle.id;
          d_HitRays[idx] = hitResult;
          return; // Exit early if intersection found
        }
      }
    }
  }

  // Step 2: Refined search

  float delta = -epsilon;
  float4 currentPosition, currentPositionLast;
  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    currentPositionLast = currentPosition;
    currentPosition = ray.origin + ray.direction * delta;

    if (currentPosition == currentPositionLast) {
      delta += epsilon;
      currentPosition = ray.origin + ray.direction * delta;
    }

    auto nearestTriangleIndex = lbvh::query_device(
        bvh_dev, lbvh::nearest(currentPosition), distance_calculator());
    if (nearestTriangleIndex.first != 0xFFFFFFFF) {
      const Triangle &hitTriangle = bvh_dev.objects[nearestTriangleIndex.first];
      float4 triangleCenter =
          (hitTriangle.v1 + hitTriangle.v2 + hitTriangle.v3) / 3.0f;
      float4 directionToTriangle = triangleCenter - ray.origin;

      float angleToTriangle =
          fabs(angleScalar(directionToTriangle, ray.direction));
      float distanceToTriangle = length(directionToTriangle);

      float t;
      if (rayTriangleIntersect(ray, hitTriangle, t)) {
        float4 hit_point = ray.origin + ray.direction * t;
        hitResult.hitResults = nearestTriangleIndex.first;
        hitResult.distanceResults = length(ray.origin - hit_point);
        hitResult.intersectionPoint =
            make_float3(hit_point.x, hit_point.y, hit_point.z);
        hitResult.idResults = hitTriangle.id;
        d_HitRays[idx] = hitResult;
        return;
      }

      delta += (angleToTriangle > 0.2f) ? distanceToTriangle * 0.75f + epsilon
                                        : epsilon;
    } else {
      delta += epsilon;
    }
  }

  // Write final result to global memory
  d_HitRays[idx] = hitResult;
}

//--------------------------------------------------------------------------------------------------------------
//**************************************************************************************************************

//**************************************************************************************************************
//--------------------------------------------------------------------------------------------------------------

__host__ __device__ __inline__ void initializeDirections(float4 *directions) {
  const float c1 = 1.0f / sqrt(3.0f);
  const float4 predefinedDirections[14] = {
      make_float4(1.0f, 0.0f, 0.0f, 0.0f), make_float4(-1.0f, 0.0f, 0.0f, 0.0f),
      make_float4(0.0f, 1.0f, 0.0f, 0.0f), make_float4(0.0f, -1.0f, 0.0f, 0.0f),
      make_float4(0.0f, 0.0f, 1.0f, 0.0f), make_float4(0.0f, 0.0f, -1.0f, 0.0f),
      make_float4(c1, c1, c1, 0.0f),       make_float4(c1, c1, -c1, 0.0f),
      make_float4(-c1, c1, c1, 0.0f),      make_float4(-c1, c1, -c1, 0.0f),
      make_float4(c1, -c1, c1, 0.0f),      make_float4(c1, -c1, -c1, 0.0f),
      make_float4(-c1, -c1, c1, 0.0f),     make_float4(-c1, -c1, -c1, 0.0f)};

  for (int i = 0; i < 14; ++i) {
    directions[i] = predefinedDirections[i];
  }
}

__global__ void initializeDirectionsKernel(float4 *directions) {
  initializeDirections(directions);
}

template <typename T, typename U>
__global__ void rayTracingKernelExplorationOptimized(
    lbvh::bvh_device<T, U> bvh_dev, Ray *rays, HitRay *d_HitRays, int numRays,
    float4 *directions, const CenterGlobalSpaceBox *d_gBox) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  Ray ray = rays[idx];

  d_HitRays[idx].hitResults = -1;
  d_HitRays[idx].distanceResults = INFINITY; // distance
  d_HitRays[idx].intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  d_HitRays[idx].idResults = -1;

  constexpr float epsilon = 0.001f;
  constexpr float angleLimit = 0.6f;
  constexpr int maxIterations = 50;
  constexpr float angleToTriangleLim = 0.2f;

  float delta = -epsilon * 1.0f; // Initial delta for ray advancement
  bool foundCandidate = false;
  Triangle closestTriangle;
  int closestTriangleId = -1;
  int step = 1;
  float t;
  bool isViewInfo = true;
  isViewInfo = false;

  if (step == 1) {
    step = 2;
    const float epsilonC = 0.01f;
    for (int i = 0; i < 14; ++i) {
      float4 currentPosition = ray.origin + directions[i] * epsilonC;
      const auto nearestTriangleIndex = lbvh::query_device(
          bvh_dev, lbvh::nearest(currentPosition), distance_calculator());

      if (nearestTriangleIndex.first != 0xFFFFFFFF) {
        const Triangle &hitTriangle =
            bvh_dev.objects[nearestTriangleIndex.first];
        if (pointInTriangle2(currentPosition, hitTriangle, 0.01f)) {

          if (isViewInfo)
            printf("in step1-level1\n");

          rayTriangleIntersect(rays[idx], hitTriangle, t);
          {
            if (isViewInfo)
              printf("in step1-level2 it=%i\n", i);
            float4 hit_point = ray.origin + ray.direction * t;
            d_HitRays[idx].hitResults = nearestTriangleIndex.first;
            d_HitRays[idx].distanceResults = t; // distance
            d_HitRays[idx].intersectionPoint =
                make_float3(hit_point.x, hit_point.y, hit_point.z);
            d_HitRays[idx].idResults = hitTriangle.id;
            step = -1;
          }
          break; // Exit the loop once we find a valid intersection
        }
      }
    }
  }

  //__syncthreads();

  if (step == 2) {
    if (isViewInfo)
      printf("in step2-level1\n");

    float4 intersectionPointWithSphereScene;
    float distanceSphereScene = INFINITY;
    bool isSceneIntersection = raySphereIntersection(
        ray.origin, ray.direction, d_gBox->position, d_gBox->radius,
        intersectionPointWithSphereScene, distanceSphereScene);

    if (isViewInfo) {
      printf("isSceneIntersection=%i\n", isSceneIntersection);
      printf("  dist=%f\n", distanceSphereScene);
      printf("  position=<%f,%f,%f>\n", intersectionPointWithSphereScene.x,
             intersectionPointWithSphereScene.y,
             intersectionPointWithSphereScene.z);
    }

    if (isSceneIntersection == 0)
      return; // no objects are in this direction

    float distRayOriTogBoxCenter = length(ray.origin - d_gBox->position);
    float rfar = INFINITY;
    if (distRayOriTogBoxCenter < 2.0f * d_gBox->radius) {
      rfar = 2.0f * d_gBox->radius;
    } else {
      rfar = 2.0f * distRayOriTogBoxCenter;
    }

    float dp = 0.0f;
    float angleToTriangle = INFINITY;
    float distanceToTriangle = 0.0f;
    float halfOpeningAngle = INFINITY;
    float inActivationAngle = INFINITY;
    float4 currentPosition;
    float4 currentPositionLast = currentPosition;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
      currentPositionLast = currentPosition;
      currentPosition = ray.origin + ray.direction * delta;

      if (currentPosition == currentPositionLast) {
        delta += distanceToTriangle * 0.15f + epsilon;
        currentPosition = ray.origin + ray.direction * delta;
      }

      auto nearestTriangleIndex = lbvh::query_device(
          bvh_dev, lbvh::nearest(currentPosition), distance_calculator());
      if (nearestTriangleIndex.first != 0xFFFFFFFF) {
        const Triangle &hitTriangle =
            bvh_dev.objects[nearestTriangleIndex.first];
        float4 triangleCenter =
            (hitTriangle.v1 + hitTriangle.v2 + hitTriangle.v3) / 3.0f;
        float4 directionToTriangle = triangleCenter - ray.origin;

        angleToTriangle = fabs(angleScalar(directionToTriangle, ray.direction));
        distanceToTriangle = length(directionToTriangle);
        halfOpeningAngle =
            calculateHalfOpeningAngle(hitTriangle, currentPosition);
        inActivationAngle =
            halfOpeningAngle - angleToTriangle - angleToTriangleLim;

        // if (halfOpeningAngle > 0.01f) { // Close object check
        if (rayTriangleIntersect(ray, hitTriangle, t)) {
          if (isViewInfo)
            printf("in step2-level2 it=%i\n", iteration);
          float4 hit_point = ray.origin + ray.direction * t;
          d_HitRays[idx].hitResults = nearestTriangleIndex.first;
          d_HitRays[idx].distanceResults = length(ray.origin - hit_point);
          d_HitRays[idx].intersectionPoint =
              make_float3(hit_point.x, hit_point.y, hit_point.z);
          d_HitRays[idx].idResults = hitTriangle.id;
          return;
        }
        //}
      }

      if (angleToTriangle > angleToTriangleLim) {
        // delta += epsilon * exp(iteration*0.25f);
        delta += distanceToTriangle * 0.75f + epsilon;
      }

      /*
            if (inActivationAngle > 0.0f) {
              // delta += epsilon * exp(iteration*0.25f);
              delta += distanceToTriangle * 0.75f + epsilon;
            }
      */

    } // END FOR
  }
}

//--------------------------------------------------------------------------------------------------------------
//**************************************************************************************************************

//**************************************************************************************************************
//--------------------------------------------------------------------------------------------------------------

__device__ __inline__ void updateHitResults(HitRay &hitRay,
                                            unsigned int triangleIndex, float t,
                                            const Ray &ray,
                                            const Triangle &hitTriangle) {
  float4 hit_point = ray.origin + ray.direction * t;
  hitRay.hitResults = triangleIndex;
  hitRay.distanceResults = t;
  hitRay.intersectionPoint = make_float3(hit_point.x, hit_point.y, hit_point.z);
  hitRay.idResults = hitTriangle.id;
}

template <typename T, typename U>
__global__ void rayTracingKernelExplorationOptimized2(
    lbvh::bvh_device<T, U> bvh_dev, Ray *rays, HitRay *d_HitRays, int numRays,
    float4 *directions, const CenterGlobalSpaceBox *d_gBox) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  Ray ray = rays[idx];
  HitRay &hitRay = d_HitRays[idx];

  // Initialize hit results
  d_HitRays[idx].hitResults = -1;
  d_HitRays[idx].distanceResults = INFINITY; // distance
  d_HitRays[idx].intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
  d_HitRays[idx].idResults = -1;

  constexpr float epsilon = 0.001f;
  constexpr int maxIterations = 50;
  constexpr float epsilonC = 0.01f;
  constexpr float angleToTriangleLim = 0.2f;

  bool isViewInfo = true;
  isViewInfo = false;

  float t;

  // Step 1: Quick intersection check
  for (int i = 0; i < 14; ++i) {
    float4 currentPosition = ray.origin + directions[i] * epsilonC;
    auto nearestTriangleIndex = lbvh::query_device(
        bvh_dev, lbvh::nearest(currentPosition), distance_calculator());

    if (nearestTriangleIndex.first != 0xFFFFFFFF) {
      const Triangle &hitTriangle = bvh_dev.objects[nearestTriangleIndex.first];
      if (pointInTriangle2(currentPosition, hitTriangle, 0.01f)) {
        if (isViewInfo)
          printf("in step1-level1\n");

        rayTriangleIntersect(ray, hitTriangle, t);
        {
          if (isViewInfo)
            printf("in step1-level2 it=%i\n", i);
          updateHitResults(hitRay, nearestTriangleIndex.first, t, ray,
                           hitTriangle);
          return;
        }
      }
    }
  }

  // Step 2: Iterative ray marching
  if (isViewInfo)
    printf("in step2-level1\n");
  float delta = epsilon;
  float4 currentPosition, lastPosition = ray.origin;

  float4 intersectionPointWithSphereScene;
  float distanceSphereScene = INFINITY;
  bool isSceneIntersection = raySphereIntersection(
      ray.origin, ray.direction, d_gBox->position, d_gBox->radius,
      intersectionPointWithSphereScene, distanceSphereScene);

  if (isViewInfo) {
    printf("isSceneIntersection=%i\n", isSceneIntersection);
    printf("  dist=%f\n", distanceSphereScene);
    printf("  position=<%f,%f,%f>\n", intersectionPointWithSphereScene.x,
           intersectionPointWithSphereScene.y,
           intersectionPointWithSphereScene.z);
  }

  if (isSceneIntersection == 0)
    return; // no objects are in this direction

  float distRayOriTogBoxCenter = length(ray.origin - d_gBox->position);
  float rfar = INFINITY;
  if (distRayOriTogBoxCenter < 2.0f * d_gBox->radius) {
    rfar = 2.0f * d_gBox->radius;
  } else {
    rfar = 2.0f * distRayOriTogBoxCenter;
  }

  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    currentPosition = ray.origin + ray.direction * delta;
    if (currentPosition == lastPosition) {
      delta += epsilon;
      continue;
    }
    lastPosition = currentPosition;
    // if (delta>rfar) printf("OUT it=%i\n",iteration);
    if (delta > rfar)
      return; // we find ourselves outside the encompassing sphere of the scene.
              // There is nothing more to look for after that.

    auto nearestTriangleIndex = lbvh::query_device(
        bvh_dev, lbvh::nearest(currentPosition), distance_calculator());
    if (nearestTriangleIndex.first != 0xFFFFFFFF) {
      const Triangle &hitTriangle = bvh_dev.objects[nearestTriangleIndex.first];
      float4 directionToTriangle =
          ((hitTriangle.v1 + hitTriangle.v2 + hitTriangle.v3) / 3.0f) -
          ray.origin;

      float angleToTriangle =
          fabs(angleScalar(directionToTriangle, ray.direction));
      float distanceToTriangle = length(directionToTriangle);

      float halfOpeningAngle =
          calculateHalfOpeningAngle(hitTriangle, currentPosition);
      float inActivationAngle = inActivationAngle =
          halfOpeningAngle - angleToTriangle - angleToTriangleLim;

      if (rayTriangleIntersect(ray, hitTriangle, t)) {
        if (isViewInfo)
          printf("in step2-level2 it=%i\n", iteration);
        updateHitResults(hitRay, nearestTriangleIndex.first, t, ray,
                         hitTriangle);
        return;
      }

      delta += (angleToTriangle > angleToTriangleLim)
                   ? (distanceToTriangle * 0.75f + epsilon)
                   : epsilon;

      /*
            delta += (inActivationAngle > 0.0f)
                         ? (distanceToTriangle * 0.75f + epsilon)
                         : epsilon;
      */
    } else {
      delta += epsilon;
    }
  }
}

//--------------------------------------------------------------------------------------------------------------
//**************************************************************************************************************

//**************************************************************************************************************
//--------------------------------------------------------------------------------------------------------------

__host__ __device__ __inline__ void
showInformationAABB(const lbvh::aabb<float> &aabb) {

  float width = aabb.upper.x - aabb.lower.x;
  float height = aabb.upper.y - aabb.lower.y;
  float depth = aabb.upper.z - aabb.lower.z;
  float lengthMax = length(aabb.upper - aabb.lower);
  float volume = width * height * depth;
  float4 position = make_float4((aabb.upper.x + aabb.lower.x) * 0.5f,
                                (aabb.upper.y + aabb.lower.y) * 0.5f,
                                (aabb.upper.z + aabb.lower.z) * 0.5f, 0.0f);

  printf("\n");
  printf("[INFO]: Bounding box:\n");
  printf("[INFO]:   Lower corner: (%.6f, %.6f, %.6f)\n", aabb.lower.x,
         aabb.lower.y, aabb.lower.z);
  printf("[INFO]:   Upper corner: (%.6f, %.6f, %.6f)\n", aabb.upper.x,
         aabb.upper.y, aabb.upper.z);

  printf("[INFO]: Position      : <%f,%f,%f>\n", position.x, position.y,
         position.z);

  printf("[INFO]: Dimensions:\n");
  printf("[INFO]:   Width       : %.6f\n", width);
  printf("[INFO]:   Height      : %.6f\n", height);
  printf("[INFO]:   Depth       : %.6f\n", depth);

  printf("[INFO]:   Volume      : %.6f\n", volume);
  printf("[INFO]:   Radius      : %.6f\n", lengthMax * 0.5f);

  printf("\n");
}

__host__ __device__ __inline__ void
buildInformationGlobalAABB(const lbvh::aabb<float> &aabb,
                           CenterGlobalSpaceBox &gBox) {
  gBox.min = aabb.lower;
  gBox.max = aabb.upper;
  gBox.width = aabb.upper.x - aabb.lower.x;
  gBox.height = aabb.upper.y - aabb.lower.y;
  gBox.depth = aabb.upper.z - aabb.lower.z;
  gBox.radius = length(aabb.upper - aabb.lower) * 0.5f;
  gBox.volume = gBox.width * gBox.height * gBox.depth;
  gBox.position = make_float4((aabb.upper.x + aabb.lower.x) * 0.5f,
                              (aabb.upper.y + aabb.lower.y) * 0.5f,
                              (aabb.upper.z + aabb.lower.z) * 0.5f, 0.0f);
}

__host__ __device__ __inline__ void
printGlobalBoxInfo(const CenterGlobalSpaceBox &gBox) {
  printf("\n");
  printf("[INFO]: Bounding box:\n");
  printf("[INFO]:   Lower corner: <%.3f, %.3f, %.3f>\n", gBox.min.x, gBox.min.y,
         gBox.min.z);
  printf("[INFO]:   Upper corner: <%.3f, %.3f, %.3f>\n", gBox.max.x, gBox.max.y,
         gBox.max.z);
  printf("[INFO]: Position      : <%.3f, %.3f, %.3f>\n", gBox.position.x,
         gBox.position.y, gBox.position.z);
  printf("[INFO]: Dimensions:\n");
  printf("[INFO]:   Width       : %.6f\n", gBox.width);
  printf("[INFO]:   Height      : %.6f\n", gBox.height);
  printf("[INFO]:   Depth       : %.6f\n", gBox.depth);
  printf("[INFO]:   Volume      : %.6f\n", gBox.volume);
  printf("[INFO]:   Radius      : %.6f\n", gBox.radius);
  printf("\n");
}

__global__ void printGlobalBoxInfoKernel(CenterGlobalSpaceBox *d_gBox) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printGlobalBoxInfo(*d_gBox);
  }
}

//--------------------------------------------------------------------------------------------------------------
//**************************************************************************************************************

//**************************************************************************************************************
//--------------------------------------------------------------------------------------------------------------

bool loadOBJTriangle(const std::string &filename,
                     std::vector<Triangle> &triangles, const int &id) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return false;
  }

  std::vector<float4> vertices;
  std::string line;
  bool isView =
      false; // Set this to true if you want to print vertex and face info

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string type;
    iss >> type;

    if (type == "v") {
      float x, y, z;
      if (iss >> x >> y >> z) {
        vertices.push_back(make_float4(
            x, y, z, 1.0f)); // Using w=1.0f for homogeneous coordinates
        if (isView)
          std::cout << "v=<" << x << "," << y << "," << z << ">\n";
      }
    } else if (type == "f") {
      unsigned int i1, i2, i3;
      if (iss >> i1 >> i2 >> i3) {
        if (isView)
          std::cout << "f=<" << i1 << "," << i2 << "," << i3 << ">\n";

        // Check if indices are within bounds
        if (i1 > 0 && i2 > 0 && i3 > 0 && i1 <= vertices.size() &&
            i2 <= vertices.size() && i3 <= vertices.size()) {

          Triangle tri;
          tri.v1 = vertices[i1 - 1];
          tri.v2 = vertices[i2 - 1];
          tri.v3 = vertices[i3 - 1];
          tri.id = id;
          triangles.push_back(tri);
        } else {
          std::cerr << "Invalid face indices: " << i1 << " " << i2 << " " << i3
                    << std::endl;
        }
      }
    }
  }

  std::cout << "Loaded " << vertices.size() << " vertices and "
            << triangles.size() << " triangles from " << filename << std::endl;
  return true;
}

//--------------------------------------------------------------------------------------------------------------
//**************************************************************************************************************

void Test002(int mode) {

  std::chrono::steady_clock::time_point t_begin_0, t_begin_1;
  std::chrono::steady_clock::time_point t_end_0, t_end_1;
  long int t_laps;

  std::vector<Triangle> triangles;
  // Load object
  loadOBJTriangle("Test.obj", triangles, 1);

  int nbTriangle = triangles.size();

  // Building the LBVH
  t_begin_0 = std::chrono::steady_clock::now();
  // lbvh::bvh<float, Triangle, aabb_getter> bvh(triangles.begin(),
  // triangles.end(), true);

  lbvh::bvh<float, Triangle, aabb_getter> bvh;
  bvh = lbvh::bvh<float, Triangle, aabb_getter>(triangles.begin(),
                                                triangles.end(), true);

  lbvh::bvh_device<float, Triangle> bvh_dev;

  bvh_dev = bvh.get_device_repr();

  // const auto bvh_dev = bvh.get_device_repr();
  t_end_0 = std::chrono::steady_clock::now();

  // Computes the bounding box of all objects in the space to define the
  // workspace.
  lbvh::aabb<float> global_aabb = thrust::reduce(
      thrust::device, bvh_dev.aabbs, bvh_dev.aabbs + bvh_dev.num_objects,
      lbvh::aabb<float>(), merge_aabb());
  // showInformationAABB(global_aabb);
  CenterGlobalSpaceBox h_gBox;
  CenterGlobalSpaceBox *d_gBox;
  buildInformationGlobalAABB(global_aabb, h_gBox);
  hipMalloc((void **)&d_gBox, sizeof(CenterGlobalSpaceBox));
  hipMemcpy(d_gBox, &h_gBox, sizeof(CenterGlobalSpaceBox),
            hipMemcpyHostToDevice);

  printGlobalBoxInfo(h_gBox);
  // printGlobalBoxInfoKernel<<<1, 1>>>(d_gBox);
  // hipDeviceSynchronize();

  //

  // Building rays

  /*
  int numRays = 21;
  thrust::host_vector<Ray> h_rays(numRays);
  h_rays[0].origin = make_float4(5.0f, 0.0f, 6.25f, 1.0f);
  h_rays[0].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[0]);

  h_rays[1].origin = make_float4(0.0f, 0.0f, 10.0f, 1.0f);
  h_rays[1].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[1]);


  h_rays[2].origin = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  h_rays[2].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[2]);

  h_rays[3].origin = make_float4(0.0f, 0.0f, 10.0f, 1.0f);
  h_rays[3].direction = make_float4(0.1f, 0.1f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[3]);

  h_rays[4].origin = make_float4(5.0f, 0.0f, 10.0f, 1.0f);
  h_rays[4].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[4]);

  h_rays[5].origin = make_float4(5.0f, 0.0f, 10.123f, 1.0f);
  h_rays[5].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[5]);

  h_rays[6].origin = make_float4(5.0f, 0.0f, 20.0f, 1.0f);
  h_rays[6].direction = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
  normalizeRayDirection(h_rays[6]);

  h_rays[7].origin = make_float4(1.0f, 1.0f, 10.0f, 1.0f);
  h_rays[7].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[7]);

  h_rays[8].origin = make_float4(0.0f, 0.0f, 2.1f, 1.0f);
  h_rays[8].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[8]);

  h_rays[9].origin = make_float4(0.0f, 0.0f, 1.01f, 1.0f);
  h_rays[9].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[9]);

  h_rays[10].origin = make_float4(-40.0f, 0.0f, 0.0f, 1.0f);
  h_rays[10].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[10]);

  h_rays[11].origin = make_float4(2.5f, 0.0f, 0.0f, 1.0f);
  h_rays[11].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[11]);

  h_rays[12].origin = make_float4(1.5f, 0.0f, 0.0f, 1.0f);
  h_rays[12].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[12]);

  h_rays[13].origin = make_float4(0.0f, 0.0f, 1.00001f, 1.0f);
  h_rays[13].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[13]);

  h_rays[14].origin = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
  h_rays[14].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  normalizeRayDirection(h_rays[14]);


  h_rays[15].origin = make_float4(20.0f, 0.0f, 0.0f, 1.0f);
  h_rays[15].direction = make_float4(-1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[15]);

  h_rays[16].origin = make_float4(0.9f, 0.89f, 0.0f, 1.0f);
  h_rays[16].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[16]);

  h_rays[17].origin = make_float4(0.9f, 0.9f, 0.0f, 1.0f);
  h_rays[17].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[17]);

  h_rays[18].origin = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  h_rays[18].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[18]);

  h_rays[19].origin = make_float4(0.9f, 0.9f, 0.9f, 1.0f);
  h_rays[19].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[19]);

  h_rays[20].origin = make_float4(-1.0f, -1.0f, -1.0f, 1.0f);
  h_rays[20].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  normalizeRayDirection(h_rays[20]);

  */

  int numRays = 1;
  thrust::host_vector<Ray> h_rays(numRays);
  // h_rays[0].origin = make_float4(0.0f, 0.0f, 1.00001f, 1.0f);

  h_rays[0].origin = make_float4(1.0f, 1.0f, 1.0, 1.0f);

  // h_rays[0].origin = make_float4(4.5f, 0.4f, 0.5, 1.0f);
  //  h_rays[0].origin = make_float4(0.8f, 0.7f, 0.7f, 1.0f);

  h_rays[0].origin = make_float4(0.5f, 0.5f, 0.5, 1.0f);

  // h_rays[0].origin = make_float4(0.9f, 0.9f, 0.9, 1.0f);
  // h_rays[0].origin = make_float4(0.9f, 0.9f, 1.0, 1.0f);

  // h_rays[0].origin = make_float4(1.5f, 0.0f, 0.0, 1.0f);

  // h_rays[0].origin = make_float4(-15.5f, 0.5f, 0.5, 1.0f);

  // h_rays[0].origin = make_float4(2.75f, 0.0f, 0.0, 1.0f);

  // h_rays[0].origin = make_float4(5.6f, 0.0f, 0.0, 1.0f);

  h_rays[0].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
  h_rays[0].direction = make_float4(-1.0f, 0.0f, 0.0f, 0.0f);

  normalizeRayDirection(h_rays[0]);

  // h_rays[0].origin = make_float4(0.5f, 0.5f, 0.5, 1.0f);
  // h_rays[0].direction = make_float4(-1.0f, 0.0f, 0.0f, 0.0f);
  // normalizeRayDirection(h_rays[0]);

  Ray *d_rays;
  hipMalloc(&d_rays, numRays * sizeof(Ray));
  hipMemcpy(d_rays, h_rays.data(), numRays * sizeof(Ray),
            hipMemcpyHostToDevice);

  HitRay *d_hitRays;
  hipMalloc(&d_hitRays, numRays * sizeof(HitRay));

  hipEvent_t start1, stop1;
  hipEventCreate(&start1);
  hipEventCreate(&stop1);
  hipEventRecord(start1);

  t_begin_1 = std::chrono::steady_clock::now();
  int threadsPerBlock = 512;
  int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;

  // if (mode==0) {  rayTracingKernel<<float, Triangle> <<<blocksPerGrid,
  // threadsPerBlock>>>(bvh_dev, d_rays, d_hitRays, numRays); }
  if (mode == 1) {
  }
  if (mode == 2) {
    rayTracingKernelExploration002<float, Triangle>
        <<<blocksPerGrid, threadsPerBlock>>>(bvh_dev, d_rays, d_hitRays,
                                             numRays);
  }

  if (mode == 3) {
    const int numDirections = 14;
    float4 *h_directions;
    float4 *d_directions;
    h_directions = new float4[numDirections];
    initializeDirections(h_directions);
    hipMalloc(&d_directions, numDirections * sizeof(float4));
    initializeDirectionsKernel<<<1, 1>>>(d_directions);

    rayTracingKernelExplorationOptimized<float, Triangle>
        <<<blocksPerGrid, threadsPerBlock>>>(bvh_dev, d_rays, d_hitRays,
                                             numRays, d_directions, d_gBox);

    hipFree(d_directions);
    delete[] h_directions;
  }

  if (mode == 4) {
    const int numDirections = 14;
    float4 *h_directions;
    float4 *d_directions;
    h_directions = new float4[numDirections];
    initializeDirections(h_directions);
    hipMalloc(&d_directions, numDirections * sizeof(float4));
    initializeDirectionsKernel<<<1, 1>>>(d_directions);

    rayTracingKernelExplorationOptimized2<float, Triangle>
        <<<blocksPerGrid, threadsPerBlock>>>(bvh_dev, d_rays, d_hitRays,
                                             numRays, d_directions, d_gBox);

    hipFree(d_directions);
    delete[] h_directions;
  }

  hipDeviceSynchronize();

  hipEventRecord(stop1);
  hipEventSynchronize(stop1);
  float milliseconds1 = 0;
  hipEventElapsedTime(&milliseconds1, start1, stop1);

  hipEventDestroy(start1);
  hipEventDestroy(stop1);
  t_end_1 = std::chrono::steady_clock::now();

  thrust::host_vector<HitRay> h_hitRays(numRays);
  hipMemcpy(h_hitRays.data(), d_hitRays, numRays * sizeof(HitRay),
            hipMemcpyDeviceToHost);

  std::cout << "\n";
  std::cout << "Debriefing\n";
  std::cout << "\n";
  for (int i = 0; i < numRays; i++) {
    if (h_hitRays[i].idResults != -1) {

      std::cout << "Ori <" << h_rays[i].origin.x << "," << h_rays[i].origin.y
                << "," << h_rays[i].origin.z << "> "
                << "dir <" << h_rays[i].direction.x << ","
                << h_rays[i].direction.y << "," << h_rays[i].direction.z << "> "

                << "[" << i << "] " << h_hitRays[i].hitResults
                << " distance = " << h_hitRays[i].distanceResults << " "
                << " position = "
                << " <" << h_hitRays[i].intersectionPoint.x << ","
                << h_hitRays[i].intersectionPoint.y << ","
                << h_hitRays[i].intersectionPoint.z << ">"
                << "\n";
    }
    if (h_hitRays[i].idResults == -1) {
      std::cout << "Ori <" << h_rays[i].origin.x << "," << h_rays[i].origin.y
                << "," << h_rays[i].origin.z << "> "
                << "dir <" << h_rays[i].direction.x << ","
                << h_rays[i].direction.y << "," << h_rays[i].direction.z << "> "
                << "[" << i << "] "
                << "No Hit !!!"
                << "\n";
    }
  }

  t_laps =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0)
          .count();
  std::cout << "[INFO]: Elapsed microseconds inside LBVH + transfer triangle : "
            << t_laps << " us\n";

  t_laps =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1)
          .count();
  std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing : " << t_laps
            << " us\n";

  std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing : "
            << milliseconds1 << " ms with hip chrono\n";

  hipFree(d_rays);
  hipFree(d_hitRays);
  hipFree(d_gBox);
  h_rays.clear();
  h_hitRays.clear();
}

__global__ void onKernel(float4 *nothing) {
  // nothing void
}

void runGPU() {
  hipEvent_t start1, stop1;
  hipEventCreate(&start1);
  hipEventCreate(&stop1);
  hipEventRecord(start1);

  float4 *d_nothing;
  hipMalloc(&d_nothing, 14 * sizeof(float4));
  onKernel<<<1, 1>>>(d_nothing);

  hipEventRecord(stop1);
  hipEventSynchronize(stop1);
  float milliseconds1 = 0;
  hipEventElapsedTime(&milliseconds1, start1, stop1);

  std::cout << "[INFO]: Elapsed microseconds inside graphics card preheating : "
            << milliseconds1 << " ms with hip chrono\n";
}

int main(int argc, char *argv[]) {

  bool isPreheating = false;

  if (argc > 0)
    isPreheating = bool(atoi(argv[1]));
  // std::cout<<argv[1];

  std::cout << "\n";
  if (isPreheating)
    runGPU();
  std::cout << "\n";

  /*
    std::cout << "[INFO]: Methode 3\n";
    Test002(3);
    std::cout << "\n";
  */

  /*
    std::cout << "[INFO]: Methode 2 again\n";
    Test002(3);
    std::cout << "\n";
  */

  std::cout << "[INFO]: Methode 4\n";
  Test002(4);
  std::cout << "\n";

  std::cout << "[INFO]: WELL DONE :-) FINISHED !\n";
  return 0;
}
