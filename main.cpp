
#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>

#include <cstdint>
#include <cassert>

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <optional>
#include <random>
#include <cfloat>
#include <stdexcept>

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>


  
//Link HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include "hipsolver.h"
#include "hipblas-export.h"
   


#include <thrust/device_vector.h> 
#include <thrust/transform.h> 
#include <thrust/functional.h> 
#include <thrust/execution_policy.h>
#include <thrust/random.h>

#include "lbvh.cuh"
#include "lbvh/bvh.cuh"


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


struct aabb_getter
{
    __device__
    lbvh::aabb<float> operator()(const Triangle& tri) const noexcept
    {
        lbvh::aabb<float> retval;
        retval.lower = make_float4(
            fminf(fminf(tri.v1.x, tri.v2.x), tri.v3.x),
            fminf(fminf(tri.v1.y, tri.v2.y), tri.v3.y),
            fminf(fminf(tri.v1.z, tri.v2.z), tri.v3.z),
            0.0f
        );
        retval.upper = make_float4(
            fmaxf(fmaxf(tri.v1.x, tri.v2.x), tri.v3.x),
            fmaxf(fmaxf(tri.v1.y, tri.v2.y), tri.v3.y),
            fmaxf(fmaxf(tri.v1.z, tri.v2.z), tri.v3.z),
            0.0f
        );
        return retval;
    }
};

struct distance_calculator
{
    __device__
    float operator()(const float4 point, const Triangle& tri) const noexcept
    {
        float4 center = make_float4(
            (tri.v1.x + tri.v2.x + tri.v3.x) / 3.0f,
            (tri.v1.y + tri.v2.y + tri.v3.y) / 3.0f,
            (tri.v1.z + tri.v2.z + tri.v3.z) / 3.0f,
            0.0f
        );
        return (point.x - center.x) * (point.x - center.x) +
               (point.y - center.y) * (point.y - center.y) +
               (point.z - center.z) * (point.z - center.z);
    }
};

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle, float& t) {
    float4 edge1 = triangle.v2 - triangle.v1;
    float4 edge2 = triangle.v3 - triangle.v1;
    float4 h = make_float4(
        ray.direction.y * edge2.z - ray.direction.z * edge2.y,
        ray.direction.z * edge2.x - ray.direction.x * edge2.z,
        ray.direction.x * edge2.y - ray.direction.y * edge2.x,
        0
    );
    float a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;
    if (a > -1e-6 && a < 1e-6) return false;
    float f = 1.0f / a;
    float4 s = ray.origin - triangle.v1;
    float u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
    if (u < 0.0 || u > 1.0) return false;
    float4 q = make_float4(
        s.y * edge1.z - s.z * edge1.y,
        s.z * edge1.x - s.x * edge1.z,
        s.x * edge1.y - s.y * edge1.x,
        0
    );
    float v = f * (ray.direction.x * q.x + ray.direction.y * q.y + ray.direction.z * q.z);
    if (v < 0.0 || u + v > 1.0) return false;
    t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);
    return (t > 1e-6);
}


template <typename T, typename U>
__global__ void process_single_point_new(lbvh::bvh_device<T, U> bvh_dev, float4 pos)
{
    const auto calc = distance_calculator();
    const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);
    
    if (nest.first != 0xFFFFFFFF) {
        printf("Nearest object index: %u\n", nest.first);
        printf("Distance to nearest object: %f\n", nest.second);
        
        // Display the coordinates of the nearest object
        const auto& nearest_object = bvh_dev.objects[nest.first];
        printf("Nearest object coordinates:\n");
        printf("  v1: (%f, %f, %f)\n", nearest_object.v1.x, nearest_object.v1.y, nearest_object.v1.z);
        printf("  v2: (%f, %f, %f)\n", nearest_object.v2.x, nearest_object.v2.y, nearest_object.v2.z);
        printf("  v3: (%f, %f, %f)\n", nearest_object.v3.x, nearest_object.v3.y, nearest_object.v3.z);
        
        // Display the coordinates of the query point
        printf("Query point coordinates: (%f, %f, %f)\n", pos.x, pos.y, pos.z);
    } else {
        printf("No nearest object found (BVH might be empty)\n");
    }
}



template <typename T, typename U>
__global__ void rayTracingKernel(lbvh::bvh_device<T, U> bvh_dev, Ray ray, float4* result) {
    const auto calc = distance_calculator();

    bool isView=true;
    
    // Point along the ray at a unit distance from the origin
    float4 pos = ray.origin + ray.direction;
    
    // Use query_device to find the closest object
    const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);
    
    if (nest.first != 0xFFFFFFFF) {
        // An object has been found
        const auto& hit_triangle = bvh_dev.objects[nest.first];
        if (isView)
        {
            printf("Nearest object index: %u\n",nest.first);
            printf("Distance to nearest object: %f\n",nest.second);
            printf("v1=%f %f %f\n",hit_triangle.v1.x, hit_triangle.v1.y, hit_triangle.v1.z);
            printf("v2=%f %f %f\n",hit_triangle.v2.x, hit_triangle.v2.y, hit_triangle.v2.z);
            printf("v3=%f %f %f\n",hit_triangle.v3.x, hit_triangle.v3.y, hit_triangle.v3.z);
        }

        // Calculate the intersection point
        float t;
        if (rayTriangleIntersect(ray, hit_triangle, t)) {
            float4 hit_point = ray.origin + ray.direction * t;
            *result = hit_point;
            if (isView)
            {
                printf("Ray hit triangle %d at point (%f, %f, %f)\n", nest.first, hit_point.x, hit_point.y, hit_point.z);
                printf("t=%f\n",t);
            }
        } else {
            *result = make_float4(INFINITY, INFINITY, INFINITY, INFINITY);
            if (isView) printf("Nearest object found but not intersected by ray\n");
        }
        
        
    } else {
        // No items found
        *result = make_float4(INFINITY, INFINITY, INFINITY, INFINITY);
        if (isView) printf("Ray did not hit any triangle\n");
    }
}



template <typename T, typename U>
__global__ void rayTracingKernel(lbvh::bvh_device<T, U> bvh_dev, Ray* rays, HitRay* d_HitRays, int numRays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRays) return;

    bool isView=true; isView=false;

    Ray ray = rays[idx];
    const auto calc = distance_calculator();
    
    // Point along the ray at a unit distance from the origin
    float4 pos = ray.origin + ray.direction;
    
    // Use query_device to find the closest object
    const auto nest = lbvh::query_device(bvh_dev, lbvh::nearest(pos), calc);

    // Initialization of results
    d_HitRays[idx].hitResults = -1;
    d_HitRays[idx].distanceResults = INFINITY; //distance
    d_HitRays[idx].intersectionPoint=make_float3(INFINITY, INFINITY, INFINITY);
    d_HitRays[idx].idResults = -1;
    
    if (nest.first != 0xFFFFFFFF) {
        // An object has been found
        const auto& hit_triangle = bvh_dev.objects[nest.first];

        if (isView)
        {
            printf("Ray %d: Nearest object index: %u\n", idx, nest.first);
            printf("Ray %d: Distance to nearest object: %f\n", idx, nest.second);
            printf("Ray %d: v1=%f %f %f\n", idx, hit_triangle.v1.x, hit_triangle.v1.y, hit_triangle.v1.z);
            printf("Ray %d: v2=%f %f %f\n", idx, hit_triangle.v2.x, hit_triangle.v2.y, hit_triangle.v2.z);
            printf("Ray %d: v3=%f %f %f\n", idx, hit_triangle.v3.x, hit_triangle.v3.y, hit_triangle.v3.z);
        }

        // Calculate the intersection point
        float t;
        if (rayTriangleIntersect(ray, hit_triangle, t)) {
            float4 hit_point = ray.origin + ray.direction * t;
            if (isView)
            {
                printf("Ray %d hit triangle %d at point (%f, %f, %f)\n", idx, nest.first, hit_point.x, hit_point.y, hit_point.z);
                printf("distance:%f\n",t);
            }
            d_HitRays[idx].hitResults = nest.first;
            d_HitRays[idx].distanceResults = t; //distance
            d_HitRays[idx].intersectionPoint=make_float3( hit_point.x, hit_point.y, hit_point.z);
            d_HitRays[idx].idResults = hit_triangle.id;
        } else {
            if (isView) printf("Ray %d: Nearest object found but not intersected by ray\n", idx);
        }
    } else {
        // No items found
       if (isView) printf("Ray %d did not hit any triangle\n", idx);
    }
}



bool loadOBJTriangle(const std::string& filename, std::vector<Triangle>& triangles, const int& id) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::vector<float4> vertices;
    std::string line;
    bool isView = false;  // Set this to true if you want to print vertex and face info

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            if (iss >> x >> y >> z) {
                vertices.push_back(make_float4(x, y, z, 1.0f));  // Using w=1.0f for homogeneous coordinates
                if (isView) std::cout << "v=<" << x << "," << y << "," << z << ">\n";
            }
        } else if (type == "f") {
            unsigned int i1, i2, i3;
            if (iss >> i1 >> i2 >> i3) {
                if (isView) std::cout << "f=<" << i1 << "," << i2 << "," << i3 << ">\n";
                
                // Check if indices are within bounds
                if (i1 > 0 && i2 > 0 && i3 > 0 && 
                    i1 <= vertices.size() && i2 <= vertices.size() && i3 <= vertices.size()) {
                    
                    Triangle tri;
                    tri.v1 = vertices[i1-1];
                    tri.v2 = vertices[i2-1];
                    tri.v3 = vertices[i3-1];
                    tri.id = id;
                    triangles.push_back(tri);
                } else {
                    std::cerr << "Invalid face indices: " << i1 << " " << i2 << " " << i3 << std::endl;
                }
            }
        }
    }

    std::cout << "Loaded " << vertices.size() << " vertices and " << triangles.size() << " triangles from " << filename << std::endl;
    return true;
}




void Test001()
{

    std::vector<Triangle> triangles;

    // Load object
    loadOBJTriangle("Test.obj",triangles,1001);
    
    // Building the LBVH
    lbvh::bvh<float, Triangle, aabb_getter> bvh(triangles.begin(), triangles.end(), true);
    const auto bvh_dev = bvh.get_device_repr();

    //float4 query_point = make_float4(5.0,0.0,2.0,1.0f);
    //process_single_point_new<float, Triangle><<<1, 1>>>(bvh_dev, query_point);
    //hipDeviceSynchronize();

    float4* d_result;
    hipMalloc(&d_result, sizeof(float4));
    Ray ray1;
    ray1.origin = make_float4(5.0f, 0.0f, 6.25f, 1.0f);
    ray1.direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);
    rayTracingKernel<float, Triangle><<<1, 1>>>(bvh_dev,ray1,d_result);
    hipDeviceSynchronize();
    float4 h_result;
    hipMemcpy(&h_result, d_result, sizeof(float4), hipMemcpyDeviceToHost);
    hipFree(d_result);
}


void Test002()
{

    std::chrono::steady_clock::time_point t_begin_0,t_begin_1;
    std::chrono::steady_clock::time_point t_end_0,t_end_1;
    long int t_laps;

    std::vector<Triangle> triangles;
    // Load object
    loadOBJTriangle("Test.obj",triangles,1001);
    
    // Building the LBVH
    t_begin_0 = std::chrono::steady_clock::now();
    lbvh::bvh<float, Triangle, aabb_getter> bvh(triangles.begin(), triangles.end(), true);
    const auto bvh_dev = bvh.get_device_repr();
    t_end_0 = std::chrono::steady_clock::now();

    // Building rays
    int numRays=2;
    thrust::host_vector<Ray> h_rays(numRays);
    h_rays[0].origin = make_float4(5.0f, 0.0f, 6.25f, 1.0f);
    h_rays[0].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);

    h_rays[1].origin = make_float4(0.0f, 0.0f, 10.0f, 1.0f);
    h_rays[1].direction = make_float4(0.0f, 0.0f, -1.0f, 0.0f);

    Ray* d_rays;
    hipMalloc(&d_rays, numRays * sizeof(Ray));
    hipMemcpy(d_rays, h_rays.data(), numRays * sizeof(Ray), hipMemcpyHostToDevice);

    HitRay* d_hitRays;
    hipMalloc(&d_hitRays, numRays*sizeof(HitRay)); 

    t_begin_1 = std::chrono::steady_clock::now();
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRays + threadsPerBlock - 1) / threadsPerBlock;
    rayTracingKernel<float, Triangle><<<blocksPerGrid, threadsPerBlock>>>(bvh_dev,d_rays,d_hitRays,numRays);
    hipDeviceSynchronize();  
    t_end_1 = std::chrono::steady_clock::now();

    thrust::host_vector<HitRay> h_hitRays(numRays);
    hipMemcpy(h_hitRays.data(),d_hitRays, numRays * sizeof(HitRay), hipMemcpyDeviceToHost);

    std::cout<<"\n";
    std::cout<<"Debriefing\n";
    std::cout<<"\n";
    for (int i=0;i<numRays;i++)
    {
        if (h_hitRays[i].idResults!=-1)
        {
            std::cout<<"["<<i<<"] "<<h_hitRays[i].hitResults<<" "
            <<h_hitRays[i].distanceResults<<" "
            <<h_hitRays[i].idResults
            <<" <"<<h_hitRays[i].intersectionPoint.x<<","<<h_hitRays[i].intersectionPoint.y<<","<<h_hitRays[i].intersectionPoint.z<<">"
            <<"\n";
        }

    }


    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0).count();
    std::cout << "[INFO]: Elapsed microseconds inside LBVH + transfer triangle : "<<t_laps<< " us\n";

    t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1).count();
    std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing : "<<t_laps<< " us\n";


    hipFree(d_rays);
    hipFree(d_hitRays);
    h_rays.clear();
    h_hitRays.clear();

}


int main(){
    //Test001();
    Test002();
    std::cout << "[INFO]: WELL DONE :-) FINISHED !"<<"\n";
    return 0;
}


