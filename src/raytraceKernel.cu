// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include <time.h>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
  
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

// HW TODO: IMPLEMENT THIS FUNCTION
// Does initial raycast from camera.
// (0, 0) is at top right corner of screen.
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
  float resWidth = (float) resolution.x;
  float resHeight = (float) resolution.y;
  
  glm::vec3 c(view);              // View direction (unit vector) from eye
  glm::vec3 e(eye);               // Camera center position
  glm::vec3 m = e + c;            // Midpoint of screen
  glm::vec3 u(up);                // Up vector
  glm::vec3 a = glm::cross(c, u); // c x u TODO: make sure this is well defined
  glm::vec3 b = glm::cross(a, c); // a x c TODO: make sure this is well defined
  
  glm::vec3 v;	                  // Vertical vector from "m" to top of screen
  glm::vec3 h;	                  // Horizontal vector from "m" to right of screen

  // Calculate v & h
  {
    float phi = fov.y * PI / 180.0f / 2.0f;
    float screenRatio = resHeight / resWidth;
    v = b * tan(phi) / (float)glm::length(b);
    float theta = atan(glm::length(v)/screenRatio / (float)glm::length(c));
    h = a * (float)glm::length(c) * tan(theta) / (float)glm::length(a);
  }
  
  // Obtain a unit vector in the direction from the eye to a pixel point (x, y) on screen
  float sx = ((float) x) / ((float) (resWidth - 1));
  float sy = ((float) y) / ((float) (resHeight - 1));
  glm::vec3 p = m - (2*sx - 1)*h - (2*sy - 1)*v; // World position of point (x, y) on screen 
  glm::vec3 rayUnitVec = glm::normalize(p-e);

  ray r;
  r.origin = eye;
  r.direction = rayUnitVec;
  //r.origin = glm::vec3(0,0,0);
  //r.direction = glm::vec3(0,0,-1);
  
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

__device__ glm::vec3 reflect(glm::vec3 const & I, glm::vec3 const & N)
{
  return I - 2.0f * glm::dot(N, I) * N;
}

__device__ bool isRayUnblocked(glm::vec3 const & point1, glm::vec3 const & point2, staticGeom* geoms, int numberOfGeoms)
{
  glm::vec3 DIRECTION(point2 - point1);
  float DISTANCE = glm::length(DIRECTION);

  // Offset start position in ray direction by small distance to prevent self collisions
  float DELTA = 0.001f;
  ray r;
  r.origin = point1 + DELTA * DIRECTION;
  r.direction = glm::normalize(DIRECTION);

  for (int i=0; i<numberOfGeoms; ++i)
  {
    float intersectionDistance;
    glm::vec3 intersectionPoint;
    glm::vec3 normal;

    switch (geoms[i].type)
    {
      case SPHERE:
        intersectionDistance = sphereIntersectionTest(geoms[i], r, intersectionPoint, normal);
        break;
      case CUBE:
        intersectionDistance = boxIntersectionTest(geoms[i], r, intersectionPoint, normal);
        break;
      case MESH:
        intersectionDistance = -1.0f;
        break;
    }

	  // Does not intersect so check next primitive
	  if (intersectionDistance <= 0.0f) continue;

    // Take into consideration intersection only between the two points.
	  if (intersectionDistance < DISTANCE) return false;
  }

  return true;
}

// HW TODO: IMPLEMENT THIS FUNCTION
// Core raytracer kernel (Assumes geometry material index is valid)
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if ( x >= resolution.x || y >= resolution.y ) return;

  ray r;
  r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);

  // ============================================
  // Determine closest intersection with geometry
  // ============================================

  float distance = -1.0f;
  glm::vec3 intersection;
  glm::vec3 normal;
  int materialIdx;
  for (int i = 0; i < numberOfGeoms; ++i)
  {
    float newDistance;
    glm::vec3 newIntersection;
    glm::vec3 newNormal;
    switch (geoms[i].type)
    {
      case SPHERE:
        newDistance = sphereIntersectionTest(geoms[i], r, newIntersection, newNormal);
        break;
      case CUBE:
        newDistance = boxIntersectionTest(geoms[i], r, newIntersection, newNormal);
        break;
      case MESH:
        newDistance = -1.0f;
        break;
    }
    if ( newDistance < 0.0f ) continue;
    if ( distance < 0.0f || (distance > 0.0f && newDistance < distance) )
    {
      distance = newDistance;
      intersection = newIntersection;
      normal = newNormal;
      materialIdx = geoms[i].materialid;
    }
  }
  
  // ============================================
  // Paint pixel
  // ============================================

  // No hit
  if ( distance < 0.0f )
  {
    colors[index] = glm::vec3(0.0f, 0.0f, 0.0f);
    //colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
    return;
  }

  // Simple local reflectance model (local illumination model formula)
  float reflectivity = 0.0f;
  float transmittance = 1.0f - reflectivity;
  glm::vec3 materialColor = materials[materialIdx].color;
	glm::vec3 reflectedColor(0.0f, 0.0f, 0.0f);
  glm::vec3 ambientLightColor(1.0f, 1.0f, 1.0f);
  
  

  float AMBIENT_WEIGHT = 0.2f;	// Ka - Ambient reflectivity factor
  float DIFFUSE_WEIGHT = 0.3f;	// Kd - Diffuse reflectivity factor
  float SPECULAR_WEIGHT = 0.5f;	// Ks - Specular reflectivity factor

  glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
  glm::vec3 color = AMBIENT_WEIGHT * ambientLightColor * materialColor;

  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(-0.15f, 0.15f);
  for ( int i = 0; i < 1; ++i)
  {
	glm::vec3 lightPosition(0.5f + (float) u01(rng), 0.75f, -0.5f  + (float) u01(rng));
    // Unit vector from intersection point to light source
    glm::vec3 LIGHT_DIRECTION = glm::normalize(lightPosition - intersection);
    // Direction of reflected light at intersection point
    glm::vec3 LIGHT_REFLECTION = glm::normalize(reflect(-1.0f*LIGHT_DIRECTION, normal));

    // Determine diffuse term
    float diffuseTerm;
    diffuseTerm = glm::dot(normal, LIGHT_DIRECTION);
    diffuseTerm = glm::clamp(diffuseTerm, 0.0f, 1.0f);

    // Determine specular term
    float specularTerm = 0.0f;
    if ( materials[materialIdx].specularExponent - 0.0f > 0.001f )
    {
      float SPECULAR_EXPONENT = materials[materialIdx].specularExponent;
      glm::vec3 EYE_DIRECTION = glm::normalize(cam.position - intersection);
      specularTerm = glm::dot(LIGHT_REFLECTION, EYE_DIRECTION);
      specularTerm = pow(fmaxf(specularTerm, 0.0f), SPECULAR_EXPONENT);
      specularTerm = glm::clamp(specularTerm, 0.0f, 1.0f);
    }
    

    if (isRayUnblocked(intersection, lightPosition, geoms, numberOfGeoms))
    {
      color += DIFFUSE_WEIGHT * lightColor * materialColor * diffuseTerm / 1.0f;
      color += SPECULAR_WEIGHT * lightColor * specularTerm / 1.0f;
    }
  }
  
  colors[index] = reflectivity*reflectedColor + transmittance*color;
}



// HW TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  clock_t time1, time2;
  time1 = clock();
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 16;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");

  time2 = clock();
  float execution_time = ((float) (time2 - time1)) / CLOCKS_PER_SEC;
  printf ("Execution time: %f\n", execution_time);
}
