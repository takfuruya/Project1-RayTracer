-------------------------------------------------------------------------------
Fall 2013 CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Implemented features:
* Raycasting from a camera into a scene through a pixel grid
* Phong lighting for one point light source
* Diffuse lambertian surfaces
* Raytraced shadows
* Cube intersection testing
* Sphere surface point sampling

Additional features:

* Soft shadows 
* Interactive camera (arrow keys to navigate in YZ plane)



![](https://raw.github.com/takfuruya/Project1-RayTracer/master/screenshots/1.png)
![](https://raw.github.com/takfuruya/Project1-RayTracer/master/screenshots/1.png)
![](https://raw.github.com/takfuruya/Project1-RayTracer/master/screenshots/1.png)
![](https://raw.github.com/takfuruya/Project1-RayTracer/master/screenshots/1.png)


Note:
Soft shadows is noisy since the number of samples is 1. By increasing this, the
shadows would appear more soft on every frame but this is not possible on Moore 100
computers since the display driver stops before computation finishes. It needs
admin rights to change these settings.

-------------------------------------------------------------------------------
Performance Evaluation
-------------------------------------------------------------------------------

I tested on different block sizes by changing the tileSize parameter in 
cudaRaytraceCore(). This adjusts the number of threads per block.

tileSize	Execution time (sec/frame)
4x4			Display driver error
7x7			OS shut down
8x8			1.386
16x16		1.663
32x32		Cuda did not compile

This test was done with 800 x 800 resolution output.
Looking at the specs on Moore 100 pc, it is compute capability 1.1 so has warp size of 32 with 
max warps per SM of 24 (and 8 blocks per SM).
