// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"

using namespace std;


// ============================================
// ================== Main ====================
// ============================================

int main(int argc, char** argv)
{
	// Set up pathtracer
	bool is_scene_loaded	= false;
	is_render_done			= false;
	target_frame			= 0;
	is_single_frame_mode	= false;


	// Read command line arguments and load scene file.
	for ( int i=1; i<argc; ++i )
	{
		// header=data (e.g. scene=my_scene.txt)
		string header, data;
		istringstream liness(argv[i]);
		getline(liness, header, '=');
		getline(liness, data, '=');

		if ( strcmp(header.c_str(), "scene") == 0 )
		{
			render_scene = new scene(data);
			is_scene_loaded = true;
		}
		else if ( strcmp(header.c_str(), "frame") == 0 )
		{
			target_frame = atoi(data.c_str());
			is_single_frame_mode = true;
		}
	}

	if ( !is_scene_loaded )
	{
		cout << "Error: scene file needed!" << endl;
		return 0;
	}


	// Set up camera from loaded pathtracer settings.
	iterations = 0;
	cam = &(render_scene->renderCam);
	width = cam->resolution[0];
	height = cam->resolution[1];

	if ( target_frame >= cam->frames )
	{
		cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
		target_frame = 0;
	}


	// Launch CUDA/GL

	Init(argc, argv); // Initialize GLUT & GLEW

	initCuda();


	GLuint passthroughProgram;
	passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(SpecialInput);

	glutMainLoop();

	return 0;
}


// ============================================
// =============== Runtime ====================
// ============================================


// GLUT display callback.
void display()
{
	runCuda();

	string title = "565Raytracer | " + utilityCore::convertIntToString(iterations) + " Iterations";
	glutSetWindowTitle(title.c_str());

	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
			GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glClear(GL_COLOR_BUFFER_BIT);   

	// VAO, shader program, and texture already bound
	glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

	glutPostRedisplay();
	glutSwapBuffers();
}


void runCuda()
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer.
	
	if ( iterations < cam->iterations )
	{
		uchar4 *dptr = NULL;
		++ iterations;
		cudaGLMapBufferObject((void**)&dptr, pbo);

		// Pack geom and material arrays.
		geom* geoms = new geom[render_scene->objects.size()];
		material* materials = new material[render_scene->materials.size()];

		for ( int i=0; i < render_scene->objects.size(); ++i )
		{
			geoms[i] = render_scene->objects[i];
		}
		for ( int i=0; i<render_scene->materials.size(); ++i )
		{
			materials[i] = render_scene->materials[i];
		}

		// Execute kernel.
		cudaRaytraceCore(dptr, cam, target_frame, iterations, materials, render_scene->materials.size(), geoms, render_scene->objects.size() );
		
		cudaGLUnmapBufferObject(pbo);
		return;
	}

	if( !is_render_done )
	{
		// Output image file.
		image outputImage(cam->resolution.x, cam->resolution.y);

		for(int x=0; x<cam->resolution.x; x++)
		{
			for(int y=0; y<cam->resolution.y; y++)
			{
				int index = x + (y * cam->resolution.x);
				outputImage.writePixelRGB(cam->resolution.x-1-x,y,cam->image[index]);
			}
		}
      
		gammaSettings gamma;
		gamma.applyGamma = true;
		gamma.gamma = 1.0/2.2;
		gamma.divisor = 1.0f;//cam->iterations;
		outputImage.setGammaSettings(gamma);
		string filename = cam->imageName;
		string s;
		stringstream out;
		out << target_frame;
		s = out.str();
		utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
		utilityCore::replaceString(filename, ".png", "."+s+".png");
		//outputImage.saveImageRGB(filename);
		//cout << "Saved frame " << s << " to " << filename << endl;
		is_render_done = true;
		if(is_single_frame_mode==true)
		{
			cudaDeviceReset(); 
			exit(0);
		}
	}

	if ( target_frame < cam->frames-1 )
	{
		//clear image buffer and move onto next frame
		target_frame++;
		iterations = 0;
		for ( int i=0; i<cam->resolution.x*cam->resolution.y; ++i )
		{
			cam->image[i] = glm::vec3(0,0,0);
		}
		cudaDeviceReset(); 
		is_render_done = false;
	}
}


const unsigned char KEY_ESC = 27;
const unsigned char KEY_W = 'w';

void keyboard(unsigned char key, int x, int y)
{
	switch (key) 
	{
		case KEY_ESC:
			exit(1);
			break;
	}
}

void SpecialInput(int key, int x, int y)
{
	glm::vec3 increment(0.0f);

	switch(key)
	{
		case GLUT_KEY_UP:
			increment.z = -0.1f;
			break;
		case GLUT_KEY_DOWN:
			increment.z = 0.1f;
			break;
		case GLUT_KEY_LEFT:
			increment.x = -0.1f;
			break;
		case GLUT_KEY_RIGHT:
			increment.x = 0.1f;
			break;
	}
	*(cam->positions) = *(cam->positions) + increment;
}




// ============================================
// Initialize GLUT & GLEW =====================
// ============================================

void Init(int argc, char* argv[])
{
	// Initialize GLUT.
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("565Raytracer");

	// Initialize GLEW.
	glewInit();
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		cout << "glewInit failed, aborting." << endl;
		exit(1);
	}

	initVAO();
	initTextures();
}


void initPBO(GLuint* pbo){
  if (pbo) {
    // set up vertex data parameter
    int num_texels = width*height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  //runCuda();
}

void initTextures(){
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);
    
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
 
void shut_down(int return_code){
  exit(return_code);
}
