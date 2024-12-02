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
  __device__ lbvh::aabb<float> operator()(const Triangle &tri) const noexcept {
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

__host__ __device__
float length(const float3& v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__
float length(const float4& v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/*
__host__ __device__
bool intersectRayAABB(const Ray& ray, const AABB& aabb) {
    float3 invDir = make_float3(1.0f,1.0f,1.0f) / make_float3(ray.direction.x, ray.direction.y, ray.direction.z);
    float3 t0 = (make_float3(aabb.min.x, aabb.min.y, aabb.min.z) - make_float3(ray.origin.x, ray.origin.y, ray.origin.z)) * invDir;
    float3 t1 = (make_float3(aabb.max.x, aabb.max.y, aabb.max.z) - make_float3(ray.origin.x, ray.origin.y, ray.origin.z)) * invDir;

    float3 tMin = fminf(t0, t1);
    float3 tMax = fmaxf(t0, t1);

    float tNear = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
    float tFar = fminf(fminf(tMax.x, tMax.y), tMax.z);

    return tNear <= tFar && tFar >= 0.0f; // Return true if there is an intersection
}
*/
/*
__host__ __device__
AABB calculateRayBoundingBox(const Ray& ray, float delta) {
    float4 minPoint = ray.origin + ray.direction * delta;
    float4 maxPoint = minPoint;
    float epsilon = delta;

    minPoint.x -= epsilon; maxPoint.x += epsilon;
    minPoint.y -= epsilon; maxPoint.y += epsilon;
    minPoint.z -= epsilon; maxPoint.z += epsilon;

    return AABB{ minPoint, maxPoint };
}
*/

/*
__host__ __device__ 
float angleScalar(const float4 v1, const float4 v2) {
	float p = (v1.x) * (v2.x) + (v1.y) * (v2.y) + (v1.z) * (v2.z);
	float n1 = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
	float n2 = sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
	float d = n1 * n2;
	float res = 0.0f;
	if (d > 0.0f) {
		float r = p / d;
		if (r > 1.0f) r = 1.0f;
		res = acos(r);
	}
	return (res);  // in radian
}
*/

__host__ __device__ 
float angleScalar(const float4 v1, const float4 v2) {
    float p = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    float n1 = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    float n2 = sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    float d = n1 * n2;
    if (d > 0.0f) {
        float r = p / d;
        r = fmaxf(-1.0f, fminf(1.0f, r));  // Clamp r to [-1, 1]
        return acosf(r);
    }
    return 0.0f; 
}


__host__ __device__
float calculateHalfOpeningAngle(const Triangle& triangle, const float4& origin) {
    // This function will be used to speed up the calculations and will adapt the limit angle of the sameDirection function
    float4 barycenter = {
        (triangle.v1.x + triangle.v2.x + triangle.v3.x) / 3.0f,
        (triangle.v1.y + triangle.v2.y + triangle.v3.y) / 3.0f,
        (triangle.v1.z + triangle.v2.z + triangle.v3.z) / 3.0f,
        0.0f 
    };

    float distance = sqrt(pow(barycenter.x - origin.x, 2) +
                          pow(barycenter.y - origin.y, 2) +
                          pow(barycenter.z - origin.z, 2));

    float4 edge1 = {
        triangle.v2.x - triangle.v1.x,
        triangle.v2.y - triangle.v1.y,
        triangle.v2.z - triangle.v1.z,
        0.0f  
    };

    float4 edge2 = {
        triangle.v3.x - triangle.v1.x,
        triangle.v3.y - triangle.v1.y,
        triangle.v3.z - triangle.v1.z,
        0.0f  
    };

    float4 cross = {
        edge1.y * edge2.z - edge1.z * edge2.y,
        edge1.z * edge2.x - edge1.x * edge2.z,
        edge1.x * edge2.y - edge1.y * edge2.x,
        0.0f 
    };

    float area = 0.5f * sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
    float solidAngle = area / (distance * distance);
    float halfOpeningAngle = asin(sqrt(solidAngle / (4 * M_PI)));

    return halfOpeningAngle;
}

__host__ __device__ 
bool sameDirection(const Triangle& tri,const Ray& ray,const float & angleLim)
{   // To be modified soon according to the radius of the triangle object
	float4 dT; 
	dT.x = (tri.v1.x + tri.v2.x + tri.v3.x) / 3.0f - ray.origin.x;
	dT.y = (tri.v1.y + tri.v2.y + tri.v3.y) / 3.0f - ray.origin.y;
	dT.z = (tri.v1.z + tri.v2.z + tri.v3.z) / 3.0f - ray.origin.z;
	float angle1 = angleScalar(dT, ray.direction);
  float angle2=calculateHalfOpeningAngle(tri,ray.origin);
	//return (angle <= angleLim);
  return (angle1 <= angleLim) && (angle1 <=angle2 );
}


__host__ __device__
bool sameDirectionTest(const Triangle& tri,const Ray& ray,const float & angleLim)
{	
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
  __device__ float operator()(const float4 point,
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
  __device__ float operator()(const float4 point, const Triangle &tri) const noexcept {
    // Function that will be used to resolve the problems of the vertices at the edge of the triangle.
    float4 edge1 = tri.v2 - tri.v1;
    float4 edge2 = tri.v3 - tri.v1;
    float3 normal = make_float3(edge1.y * edge2.z - edge1.z * edge2.y,
                                edge1.z * edge2.x - edge1.x * edge2.z,
                                edge1.x * edge2.y - edge1.y * edge2.x);
    float normLength = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (normLength > 0) {
      normal.x /= normLength;
      normal.y /= normLength;
      normal.z /= normLength;
    }

    float D = -(normal.x * tri.v1.x + normal.y * tri.v1.y + normal.z * tri.v1.z);
    float distance = fabs(normal.x * point.x + normal.y * point.y + normal.z * point.z + D);
    return distance;
  }
};


struct distance_calculator3 {
  __device__  float operator()(const float4& point, 
                               const Triangle& triangle) const noexcept {
      float4 edge1 = {
          triangle.v2.x - triangle.v1.x,
          triangle.v2.y - triangle.v1.y,
          triangle.v2.z - triangle.v1.z,
          0.0f
      };
      float4 edge2 = {
          triangle.v3.x - triangle.v1.x,
          triangle.v3.y - triangle.v1.y,
          triangle.v3.z - triangle.v1.z,
          0.0f
      };
      float4 v0 = {
          triangle.v1.x - point.x,
          triangle.v1.y - point.y,
          triangle.v1.z - point.z,
          0.0f
      };


      float a = edge1.x*edge1.x + edge1.y*edge1.y + edge1.z*edge1.z;
      float b = edge1.x*edge2.x + edge1.y*edge2.y + edge1.z*edge2.z;
      float c = edge2.x*edge2.x + edge2.y*edge2.y + edge2.z*edge2.z;
      float d = edge1.x*v0.x + edge1.y*v0.y + edge1.z*v0.z;
      float e = edge2.x*v0.x + edge2.y*v0.y + edge2.z*v0.z;


      float det = a*c - b*b;
      float s = b*e - c*d;
      float t = b*d - a*e;

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
                  float denom = a - 2.0f*b + c;
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
                  float denom = a - 2.0f*b + c;
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
                  float denom = a - 2.0f*b + c;
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


      float4 closestPoint = {
          triangle.v1.x + s*edge1.x + t*edge2.x,
          triangle.v1.y + s*edge1.y + t*edge2.y,
          triangle.v1.z + s*edge1.z + t*edge2.z,
          0.0f
      };


      float dx = point.x - closestPoint.x;
      float dy = point.y - closestPoint.y;
      float dz = point.z - closestPoint.z;
      return sqrt(dx*dx + dy*dy + dz*dz);
    }
};







__host__ __device__ inline void normalizeRayDirection(Ray &ray) {
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

__device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 float4_to_float3(const float4& v) {
    return make_float3(v.x, v.y, v.z);
}


__device__ bool computeBarycentricCoordinates(const float4 &P, const Triangle &tri, float &u, float &v) {
    float4 v0 = tri.v2 - tri.v1;
    float4 v1 = tri.v3 - tri.v1;
    float4 v2 = P - tri.v1;

    float d00 = dot(float4_to_float3(v0), float4_to_float3(v0));
    float d01 = dot(float4_to_float3(v0), float4_to_float3(v1));
    float d11 = dot(float4_to_float3(v1), float4_to_float3(v1));
    float d20 = dot(float4_to_float3(v2), float4_to_float3(v0));
    float d21 = dot(float4_to_float3(v2), float4_to_float3(v1));

    float denom = d00 * d11 - d01 * d01;
    if (denom == 0) return false; 

    u = (d11 * d20 - d01 * d21) / denom;
    v = (d00 * d21 - d01 * d20) / denom;

    return (u >= 0 && v >= 0 && (u + v) <= 1); // Check if the point is inside or on the edge
}




__device__ bool rayTriangleIntersect(const Ray &ray, const Triangle &triangle,
                                     float &t) {
  float4 edge1 = triangle.v2 - triangle.v1;
  float4 edge2 = triangle.v3 - triangle.v1;
  float4 h =
      make_float4(ray.direction.y * edge2.z - ray.direction.z * edge2.y,
                  ray.direction.z * edge2.x - ray.direction.x * edge2.z,
                  ray.direction.x * edge2.y - ray.direction.y * edge2.x, 0);
  float a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;
  if (a > -1e-6 && a < 1e-6)
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
  return (t > 1e-6);
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
  float epsilon=0.001f;
  float4 pos = ray.origin + ray.direction*epsilon; // un petit décalage por chercher dans la bonne direction

  // Use query_device to find the closest object
  //auto query = lbvh::nearest(query_point).set_max_depth(max_depth);
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
  //isView = false;

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
      printf("oRay %d: Nearest object index: %u Distance to nearest object: %f\n", idx, nest.first,nest.second);
      //printf("oRay %d: v1=%f %f %f\n", idx, hit_triangle.v1.x, hit_triangle.v1.y, hit_triangle.v1.z);
      //printf("oRay %d: v2=%f %f %f\n", idx, hit_triangle.v2.x, hit_triangle.v2.y, hit_triangle.v2.z);
      //printf("oRay %d: v3=%f %f %f\n", idx, hit_triangle.v3.x, hit_triangle.v3.y, hit_triangle.v3.z);
    }

    // Calculate the intersection point
    float t;
    if (rayTriangleIntersect(ray, hit_triangle, t)) {
      float4 hit_point = ray.origin + ray.direction * t;
      if (isView) {
        printf("Ray %d hit triangle %d at point (%f, %f, %f) Distance:%f\n", idx, nest.first, hit_point.x, hit_point.y, hit_point.z,t);
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



template <typename T, typename U>
__global__ void rayTracingKernelSurfaceEdge(lbvh::bvh_device<T, U> bvh_dev, Ray *rays,
                                 HitRay *d_HitRays, int numRays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays)
        return;

    Ray ray = rays[idx];
    normalizeRayDirection(ray);


    const float epsilon = 0.001f;
    float4 pos = ray.origin + ray.direction * epsilon;


    const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), distance_calculator());


    d_HitRays[idx].hitResults = -1;
    d_HitRays[idx].distanceResults = INFINITY; // distance
    d_HitRays[idx].intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
    d_HitRays[idx].idResults = -1;

    if (nest.first != 0xFFFFFFFF) {
        const auto &hit_triangle = bvh_dev.objects[nest.first];

        float t;
        if (rayTriangleIntersect(ray, hit_triangle, t)) {
            float4 hit_point = ray.origin + ray.direction * t;

            float u, v;
            if (computeBarycentricCoordinates(hit_point, hit_triangle, u, v)) {
                d_HitRays[idx].hitResults = nest.first;
                d_HitRays[idx].distanceResults = t; 
                d_HitRays[idx].intersectionPoint =
                    make_float3(hit_point.x, hit_point.y, hit_point.z);
                d_HitRays[idx].idResults = hit_triangle.id;

                printf("Ray %d hit triangle %d at point (%f, %f, %f) Distance:%f\n", idx, nest.first,
                       hit_point.x, hit_point.y, hit_point.z, t);
            } else {
                printf("Ray %d: Intersection at vertex or outside triangle\n", idx);
            }
        } else {
            printf("Ray %d: Nearest object found but not intersected by ray\n", idx);
        }
    } else {
        // Aucun objet trouvé
        printf("Ray %d did not hit any triangle\n", idx);
    }
}



template <typename T, typename U>
__global__ void rayTracingKernelExploration001(lbvh::bvh_device<T, U> bvh_dev, Ray* rays,
    HitRay* d_HitRays, int numRays) {
  // The objective of this function is to explore in the direction of the ray the candidate triangle which intersects.
  // Like an explorer drone that encounters a wall
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays)
        return;

    bool isView = true;
    isView = false;

    Ray ray = rays[idx];
    const auto calc = distance_calculator();

    // Initialization of results
    d_HitRays[idx].hitResults = -1;
    d_HitRays[idx].distanceResults = INFINITY; // distance
    d_HitRays[idx].intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
    d_HitRays[idx].idResults = -1;

    constexpr float epsilon = 0.001f;
    constexpr float angleLim = 0.6f;
    constexpr int maxLoops = 20;

    float angle1 = INFINITY;
    float angle2 = INFINITY;
    float distToTri = 0.0f;
    bool flag = true;
    bool flagOk = false;
    bool flagFindCandidate = false;
    Triangle hit_tri;
    int idNest  = -1;
    int idNestC = -1;
    int nbLoop = 1;
    //float delta = epsilon;
    float delta = -epsilon; //PB inside triangle

    while (flag)
    {
        float4 pos = ray.origin + ray.direction * delta;
        //printf("Pos=%f %f %f\n",pos.x,pos.y,pos.z);
        const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);
        flag = false;
        nbLoop++;
        if (nest.first != 0xFFFFFFFF) {
            const auto& hit_triangle = bvh_dev.objects[nest.first];
            float4 dT;
            dT.x = (hit_triangle.v1.x + hit_triangle.v2.x + hit_triangle.v3.x) / 3.0f - ray.origin.x;
            dT.y = (hit_triangle.v1.y + hit_triangle.v2.y + hit_triangle.v3.y) / 3.0f - ray.origin.y;
            dT.z = (hit_triangle.v1.z + hit_triangle.v2.z + hit_triangle.v3.z) / 3.0f - ray.origin.z;
            angle1 = fabs(angleScalar(dT, ray.direction));
            distToTri = sqrt(dT.x * dT.x + dT.y * dT.y + dT.z * dT.z);
            flagOk = true; 
            idNest = nest.first;
            hit_tri = hit_triangle;
            float angle2=calculateHalfOpeningAngle(hit_triangle,ray.origin);
            //printf("angle1=%f\n",angle1);
            //printf("angle2=%f\n",angle2);
            //if (angle1 > angleLim) { flag = true; flagOk = false; delta = delta+ distToTri*0.5f + epsilon;  }
            //if (!qinfo) { flag = true; flagOk = false; delta = epsilon * exp(nbLoop-1); }
            if (angle1 > angleLim) { flag = true; flagOk = false; delta = epsilon * exp(nbLoop-1);  }
            if ( angle1 < 1.785f ) { flagFindCandidate = true; idNestC = idNest;  }
            if ( angle2 > 1.0f ) { flag = false; flagOk = true;}
        } 
        else
        {
            delta = epsilon * exp(nbLoop);
        }

        if (nbLoop > maxLoops) { 
            flag = false; flagOk = false; 
            if (flagFindCandidate) { 
                flagOk = true; 
                idNest = idNestC; 
                const auto& hit_triangle = bvh_dev.objects[idNest];
                hit_tri = hit_triangle;
            } 
        }

    }

    if (isView) printf("Ray %d Level 1 finished\n", idx);

    if (flagOk)
    {
        float t;
        if (rayTriangleIntersect(ray, hit_tri, t)) {
            float4 hit_point = ray.origin + ray.direction * t;
            if (isView) {
                printf("Ray %d hit triangle %d at point (%f, %f, %f) Distance:%f\n", idx, idNest, hit_point.x, hit_point.y, hit_point.z, t);
            }
            d_HitRays[idx].hitResults = idNest;
            d_HitRays[idx].distanceResults = t; // distance
            d_HitRays[idx].intersectionPoint = make_float3(hit_point.x, hit_point.y, hit_point.z);
            d_HitRays[idx].idResults = hit_tri.id;
        }
        else {
            if (isView)
                printf("Ray %d: Nearest object found but not intersected by ray\n",
                    idx);
        }
    }
    else {
        // No items found
        if (isView)
            printf("Ray %d did not hit any triangle\n", idx);
    }
}


template <typename T, typename U>
__global__ void rayTracingKernelExploration002(lbvh::bvh_device<T, U> bvh_dev, Ray* rays,
    HitRay* d_HitRays, int numRays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    Ray ray = rays[idx];
    //d_HitRays[idx] = { -1, INFINITY, make_float3(INFINITY), -1 }; // Initialize results

    d_HitRays[idx].hitResults = -1;
    d_HitRays[idx].distanceResults = INFINITY; // distance
    d_HitRays[idx].intersectionPoint = make_float3(INFINITY,INFINITY,INFINITY);
    d_HitRays[idx].idResults = -1;

    constexpr float epsilon = 0.001f;
    constexpr float angleLimit = 0.6f;
    constexpr int maxIterations = 20;

    float delta = -epsilon; // Initial delta for ray advancement
    bool foundCandidate = false;
    Triangle closestTriangle;
    int closestTriangleId = -1;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        float4 currentPosition = ray.origin + ray.direction * delta;
        const auto nearestTriangleIndex = lbvh::query_device(bvh_dev, lbvh::nearest(currentPosition), distance_calculator());

        if (nearestTriangleIndex.first != 0xFFFFFFFF) {
            const Triangle& hitTriangle = bvh_dev.objects[nearestTriangleIndex.first];
            float4 triangleCenter = (hitTriangle.v1 + hitTriangle.v2 + hitTriangle.v3) / 3.0f;
            float4 directionToTriangle = triangleCenter - ray.origin;

            float angleToTriangle = fabs(angleScalar(directionToTriangle, ray.direction));
            float distanceToTriangle = length(directionToTriangle);
            float halfOpeningAngle = calculateHalfOpeningAngle(hitTriangle, ray.origin);

            if (halfOpeningAngle > 0.4f) { // Close object check
                float t;
                if (rayTriangleIntersect(ray, hitTriangle, t)) {
                    /*
                    d_HitRays[idx] = {
                        nearestTriangleIndex.first,
                        t - length(ray.origin - currentPosition), // Corrected distance
                        make_float3(ray.origin + ray.direction * t),
                        hitTriangle.id
                    };
                    */
                    float4 hit_point = ray.origin + ray.direction * t;
                    d_HitRays[idx].hitResults = nearestTriangleIndex.first;
                    d_HitRays[idx].distanceResults = t - length(ray.origin - currentPosition); // Corrected distance
                    d_HitRays[idx].intersectionPoint = make_float3(hit_point.x, hit_point.y, hit_point.z);
                    d_HitRays[idx].idResults = hitTriangle.id;
                    return; // Exit early on successful intersection
                }
            }
            else if (angleToTriangle > angleLimit) {
                delta += distanceToTriangle * 0.5f + epsilon; // Move further away
            }
            else {
                foundCandidate = true;
                closestTriangleId = nearestTriangleIndex.first;
                closestTriangle = hitTriangle;
            }
        }
        else {
            delta += epsilon * exp(iteration); // Adjust delta for next iteration
        }
    }

    // If no intersection was found but a candidate was identified
    if (foundCandidate && closestTriangleId != -1) {
        float t;
        if (rayTriangleIntersect(ray, closestTriangle, t)) {
            /*
            d_HitRays[idx] = {
                closestTriangleId,
                t,
                make_float3(ray.origin + ray.direction * t),
                closestTriangle.id
            };
            */
            float4 hit_point = ray.origin + ray.direction * t;
            d_HitRays[idx].hitResults = closestTriangleId;
            d_HitRays[idx].distanceResults = t; // distance
            d_HitRays[idx].intersectionPoint = make_float3(hit_point.x, hit_point.y, hit_point.z);
            d_HitRays[idx].idResults = closestTriangle.id;
            //d_HitRays[idx].idResults = hit_tri.id;
        }
    }
}


__host__ __device__
lbvh::aabb<float> calculateRayBoundingBox(const Ray& ray, float delta) {
    float4 minPoint = ray.origin + ray.direction * delta;
    float4 maxPoint = minPoint;
    float epsilon = delta;

    minPoint.x -= epsilon; maxPoint.x += epsilon;
    minPoint.y -= epsilon; maxPoint.y += epsilon;
    minPoint.z -= epsilon; maxPoint.z += epsilon;

    lbvh::aabb<float> boundingBox;
    boundingBox.lower = make_float4(minPoint.x, minPoint.y, minPoint.z, 0);
    boundingBox.upper = make_float4(maxPoint.x, maxPoint.y, maxPoint.z, 0);

    return boundingBox;
}



template <typename T, typename U>
__global__ void rayTracingKernelExploration003(lbvh::bvh_device<T, U> bvh_dev, Ray* rays,
    HitRay* d_HitRays, int numRays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    Ray ray = rays[idx];
    d_HitRays[idx].hitResults = -1;
    d_HitRays[idx].distanceResults = INFINITY;
    d_HitRays[idx].intersectionPoint = make_float3(INFINITY, INFINITY, INFINITY);
    d_HitRays[idx].idResults = -1;

    constexpr float epsilon = 0.001f;
    constexpr int maxIterations = 20;
    constexpr int maxOverlappingTriangles = 10; // Adjust as needed

    float delta = -epsilon;
    bool foundCandidate = false;
    Triangle closestTriangle;
    int closestTriangleId = -1;

    for (int iteration = 0; iteration < maxIterations; ++iteration) { 
        unsigned int buffer[maxOverlappingTriangles];
        auto rayBoundingBox = calculateRayBoundingBox(ray, delta);
        auto num_found = lbvh::query_device(bvh_dev, lbvh::overlaps(rayBoundingBox), maxOverlappingTriangles, buffer);

        float4 currentPosition = ray.origin + ray.direction * delta;

        for (unsigned int i = 0; i < num_found; ++i) {
            const auto triangleIndex = buffer[i];
            const Triangle& hitTriangle = bvh_dev.objects[triangleIndex];

            float4 triangleCenter = (hitTriangle.v1 + hitTriangle.v2 + hitTriangle.v3) / 3.0f;
            float4 directionToTriangle = triangleCenter - ray.origin;

            float angleToTriangle = fabs(angleScalar(directionToTriangle, ray.direction));
            float distanceToTriangle = length(directionToTriangle);
            float halfOpeningAngle = calculateHalfOpeningAngle(hitTriangle, ray.origin);

            if (halfOpeningAngle > 0.4f) {
                float t;
                if (rayTriangleIntersect(ray, hitTriangle, t)) {
                    float4 hit_point = ray.origin + ray.direction * t;
                    d_HitRays[idx].hitResults = triangleIndex;
                    d_HitRays[idx].distanceResults = t - length(ray.origin - currentPosition);
                    d_HitRays[idx].intersectionPoint = make_float3(hit_point.x, hit_point.y, hit_point.z);
                    d_HitRays[idx].idResults = hitTriangle.id;
                    return;
                }
            }
            else if (angleToTriangle > 0.6f) {
                delta += distanceToTriangle * 0.5f + epsilon;
            }
            else {
                foundCandidate = true;
                closestTriangleId = triangleIndex;
                closestTriangle = hitTriangle;
            }
        }

        delta += epsilon;
    }

    if (foundCandidate && closestTriangleId != -1) {
        float t;
        if (rayTriangleIntersect(ray, closestTriangle, t)) {
            float4 hit_point = ray.origin + ray.direction * t;
            d_HitRays[idx].hitResults = closestTriangleId;
            d_HitRays[idx].distanceResults = t;
            d_HitRays[idx].intersectionPoint = make_float3(hit_point.x, hit_point.y, hit_point.z);
            d_HitRays[idx].idResults = closestTriangle.id;
        }
    }
}





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

void Test001() {

  std::vector<Triangle> triangles;

  // Load object
  loadOBJTriangle("Test.obj", triangles, 1001);

  // Building the LBVH

  lbvh::bvh<float, Triangle, aabb_getter> bvh2;

  lbvh::bvh<float, Triangle, aabb_getter> bvh(triangles.begin(),
                                              triangles.end(), true);
  const auto bvh_dev = bvh.get_device_repr();

  // float4 query_point = make_float4(5.0,0.0,2.0,1.0f);
  // process_single_point_new<float, Triangle><<<1, 1>>>(bvh_dev, query_point);
  // hipDeviceSynchronize();

  float4 *d_result;
  hipMalloc(&d_result, sizeof(float4));
  Ray ray1;
  ray1.origin = make_float4(5.0f, 0.0f, 6.25f, 1.0f);
  ray1.direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
  rayTracingKernel<float, Triangle><<<1, 1>>>(bvh_dev, ray1, d_result);
  hipDeviceSynchronize();
  float4 h_result;
  hipMemcpy(&h_result, d_result, sizeof(float4), hipMemcpyDeviceToHost);
  hipFree(d_result);
}

void Test002() {

  std::chrono::steady_clock::time_point t_begin_0, t_begin_1;
  std::chrono::steady_clock::time_point t_end_0, t_end_1;
  long int t_laps;

  std::vector<Triangle> triangles;
  // Load object
  loadOBJTriangle("Test.obj", triangles, 1001);

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

  // Building rays

  
  int numRays = 16;
  thrust::host_vector<Ray> h_rays(numRays);
  h_rays[0].origin = make_float4(5.0f, 0.0f, 6.25f, 1.0f);
  h_rays[0].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[0]);

  h_rays[1].origin = make_float4(0.0f, 0.0f, 10.0f, 1.0f);
  h_rays[1].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[1]);


  h_rays[2].origin = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  h_rays[2].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[2]);

  h_rays[3].origin = make_float4(0.0f, 0.0f, 10.0f, 1.0f);
  h_rays[3].direction = make_float4(0.1f, 0.1f, -1.0f, 0.0f); normalizeRayDirection(h_rays[3]);

  h_rays[4].origin = make_float4(5.0f, 0.0f, 10.0f, 1.0f);
  h_rays[4].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[4]);

  h_rays[5].origin = make_float4(5.0f, 0.0f, 10.123f, 1.0f);
  h_rays[5].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[5]);

  h_rays[6].origin = make_float4(5.0f, 0.0f, 20.0f, 1.0f);
  h_rays[6].direction = make_float4(0.0f, 0.0f, 1.0f, 0.0f); normalizeRayDirection(h_rays[6]);

  h_rays[7].origin = make_float4(1.0f, 1.0f, 10.0f, 1.0f);
  h_rays[7].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[7]);

  h_rays[8].origin = make_float4(0.0f, 0.0f, 2.1f, 1.0f);
  h_rays[8].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[8]);

  h_rays[9].origin = make_float4(0.0f, 0.0f, 1.01f, 1.0f);
  h_rays[9].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[9]);

  h_rays[10].origin = make_float4(-40.0f, 0.0f, 0.0f, 1.0f);
  h_rays[10].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f); normalizeRayDirection(h_rays[10]);

  h_rays[11].origin = make_float4(2.5f, 0.0f, 0.0f, 1.0f);
  h_rays[11].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f); normalizeRayDirection(h_rays[11]);

  h_rays[12].origin = make_float4(1.5f, 0.0f, 0.0f, 1.0f);
  h_rays[12].direction = make_float4(1.0f, 0.0f, 0.0f, 0.0f); normalizeRayDirection(h_rays[12]);

  h_rays[13].origin = make_float4(0.0f, 0.0f, 1.00001f, 1.0f);
  h_rays[13].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[13]);

  h_rays[14].origin = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
  h_rays[14].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[14]);


  h_rays[15].origin = make_float4(20.0f, 0.0f, 0.0f, 1.0f);
  h_rays[15].direction = make_float4(-1.0f, 0.0f, 0.0f, 0.0f); normalizeRayDirection(h_rays[15]);
  

/*
  int numRays = 1;
  thrust::host_vector<Ray> h_rays(numRays);
  //h_rays[0].origin = make_float4(0.0f, 0.0f, 1.00001f, 1.0f);
  h_rays[0].origin = make_float4(0.0f, 0.0f, 1.001f, 1.0f);
  //h_rays[0].origin = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
  h_rays[0].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f); normalizeRayDirection(h_rays[0]);
*/

  Ray *d_rays;
  hipMalloc(&d_rays, numRays * sizeof(Ray));
  hipMemcpy(d_rays, h_rays.data(), numRays * sizeof(Ray),
            hipMemcpyHostToDevice);

  HitRay *d_hitRays;
  hipMalloc(&d_hitRays, numRays * sizeof(HitRay));

  t_begin_1 = std::chrono::steady_clock::now();
  int threadsPerBlock = 512;
  int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;
  //rayTracingKernel<float, Triangle>
  //rayTracingKernelExploration001<float, Triangle>
  rayTracingKernelExploration002<float, Triangle>
  //rayTracingKernelExploration003<float, Triangle>
  //rayTracingKernelSurfaceEdge<float, Triangle>
      <<<blocksPerGrid, threadsPerBlock>>>(bvh_dev, d_rays, d_hitRays, numRays);
  hipDeviceSynchronize();
  t_end_1 = std::chrono::steady_clock::now();

  thrust::host_vector<HitRay> h_hitRays(numRays);
  hipMemcpy(h_hitRays.data(), d_hitRays, numRays * sizeof(HitRay),
            hipMemcpyDeviceToHost);

  std::cout << "\n";
  std::cout << "Debriefing\n";
  std::cout << "\n";
  for (int i = 0; i < numRays; i++) {
    if (h_hitRays[i].idResults != -1) {
      std::cout << "[" << i << "] " << h_hitRays[i].hitResults << " distance = "
                << h_hitRays[i].distanceResults << " " << " position = "
                << " <" << h_hitRays[i].intersectionPoint.x << ","
                << h_hitRays[i].intersectionPoint.y << ","
                << h_hitRays[i].intersectionPoint.z << ">"
                << "\n";
    }
    if (h_hitRays[i].idResults == -1) {
      std::cout << "[" << i << "] " << "No Hit !!!"<< "\n";
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

  hipFree(d_rays);
  hipFree(d_hitRays);
  h_rays.clear();
  h_hitRays.clear();
}

int main() {
  // Test001();
  Test002();
  std::cout << "[INFO]: WELL DONE :-) FINISHED !"
            << "\n";
  return 0;
}