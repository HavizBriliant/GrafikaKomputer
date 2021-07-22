include <windows.h>
include <GL/glut.h>
include <GL/glut.h>
include <stdlib.h>
import cv2
import numpy as np

//Initializes 3D rendering
void initRendering() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING); //Enable lighting
    glEnable(GL_LIGHT0); //Enable light #0
    glEnable(GL_LIGHT1); //Enable light #1
    glEnable(GL_NORMALIZE); //Automatically normalize normals
    glShadeModel(GL_SMOOTH); //Enable smooth shading
    }

//Called when the window is resized
void handleResize(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    lLoadIdentity();
    gluPerspective(45.0, (double)w / (double)h, 1.0, 200.0);
    }

float _angle = -70.0f;

//Draws the 3D scene
void drawScene() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW); // keep it like this
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -14.0f);

//Add ambient light
    GLfloat ambientColor[] = {0.2f, 0.2f, 0.2f, 1.0f}; //Color (0.2, 0.2, 0.2)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientColor);

//Add positioned light
    GLfloat lightColor0[] = {0.5f, 0.5f, 0.5f, 1.0f}; //Color (0.5, 0.5, 0.5)
    GLfloat lightPos0[] = {0.0f, -8.0f, 8.0f, 1.0f}; //Positioned at (4, 0, 8)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor0);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

//Add directed light
    GLfloat lightColor1[] = {0.5f, 0.2f, 0.2f, 1.0f}; //Color (0.5, 0.2, 0.2)

//Coming from the direction (-1, 0.5, 0.5)
    GLfloat lightPos1[] = {-1.0f, 0.5f, 0.5f, 0.0f};
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor1);
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);

    glRotatef(10, 1.0f, 0.0f, 0.0f);
    glRotatef(-10, 0.0f, 0.0f, 1.0f);
    glRotatef(_angle,0.0f, 1.0f, 0.0f);
	//glRotatef(10, 1.0f, 0.0f, 0.0f);
	//glRotatef(-10, 0.0f, 0.0f, 1.0f);
	//glRotatef(_angle,0.0f, 1.0f, 0.0f);
    glColor3f(1.0f, 1.0f, 0.0f);
    glBegin(GL_QUADS);

	//Front
    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-2.0f, -0.2f, 2.0f);
    glVertex3f(2.0f, -0.2f, 2.0f);
    glVertex3f(2.0f, 0.2f, 2.0f);
    glVertex3f(-2.0f, 0.2f, 2.0f);

	//Right
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(2.0f, -0.2f, -2.0f);
    glVertex3f(2.0f, 0.2f, -2.0f);
    glVertex3f(2.0f, 0.2f, 2.0f);
    glVertex3f(2.0f, -0.2f, 2.0f);

	//Back
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(-2.0f, -0.2f, -2.0f);
    glVertex3f(-2.0f, 0.2f, -2.0f);
    glVertex3f(2.0f, 0.2f, -2.0f);
    glVertex3f(2.0f, -0.2f, -2.0f);

	//Left
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-2.0f, -0.2f, -2.0f);
    glVertex3f(-2.0f, -0.2f, 2.0f);
    glVertex3f(-2.0f, 0.2f, 2.0f);
    glVertex3f(-2.0f, 0.2f, -2.0f);

	//top
    glNormal3f(0.0f,1.0f,0.0f);
    glVertex3f(2.0f, 0.2f, 2.0f);
    glVertex3f(-2.0f, 0.2f, 2.0f);
    glVertex3f(-2.0f, 0.2f, -2.0f);
    glVertex3f(2.0f, 0.2f, -2.0f);

	//bottom
    glNormal3f(0.0f,-1.0f,0.0f);
    glVertex3f(2.0f, -0.2f, 2.0f);
    glVertex3f(-2.0f, -0.2f, 2.0f);
    glVertex3f(-2.0f, -0.2f, -2.0f);
    glVertex3f(2.0f, -0.2f, -2.0f);

//table front leg
	//front
    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(1.8f,-0.2f,1.6f);
    glVertex3f(1.4f, -0.2f, 1.6f);
    glVertex3f(1.4f, -3.0f, 1.6f);
    glVertex3f(1.8f, -3.0f, 1.6f);
	
	//back
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(1.8f,-0.2f,1.2f);
    glVertex3f(1.4f, -0.2f, 1.2f);
    glVertex3f(1.4f, -3.0f, 1.2f);
    glVertex3f(1.8f, -3.0f, 1.2f);

	//right
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(1.8f,-0.2f,1.6f);
    glVertex3f(1.8f, -0.2f, 1.2f);
    glVertex3f(1.8f, -3.0f, 1.2f);
    glVertex3f(1.8f, -3.0f, 1.6f);

	//left
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(1.4f,-0.2f,1.6f);
    glVertex3f(1.4f, -0.2f, 1.2f);
    glVertex3f(1.4f, -3.0f, 1.2f);
    glVertex3f(1.4f, -3.0f, 1.6f);

//back leg back
	//front
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(1.8f,-0.2f,-1.2f);
    glVertex3f(1.4f, -0.2f, -1.2f);
    glVertex3f(1.4f, -3.0f, -1.2f);
    glVertex3f(1.8f, -3.0f, -1.2f);

	//back
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(1.8f,-0.2f,-1.6f);
    glVertex3f(1.4f, -0.2f, -1.6f);
    glVertex3f(1.4f, -3.0f, -1.6f);
    glVertex3f(1.8f, -3.0f, -1.6f);

	//right
	glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(1.8f,-0.2f,-1.6f);
    glVertex3f(1.8f, -0.2f, -1.2f);
    glVertex3f(1.8f, -3.0f, -1.2f);
    glVertex3f(1.8f, -3.0f, -1.6f);

	//left
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(1.4f,-0.2f,-1.6f);
    glVertex3f(1.4f, -0.2f, -1.2f);
    glVertex3f(1.4f, -3.0f, -1.2f);
    glVertex3f(1.4f, -3.0f, -1.6f);

	//leg left front
    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(-1.8f,-0.2f,1.6f);
    glVertex3f(-1.4f, -0.2f, 1.6f);
    glVertex3f(-1.4f, -3.0f, 1.6f);
    glVertex3f(-1.8f, -3.0f, 1.6f);

	//back
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(-1.8f,-0.2f,1.2f);
    glVertex3f(-1.4f, -0.2f, 1.2f);
    glVertex3f(-1.4f, -3.0f, 1.2f);
    glVertex3f(-1.8f, -3.0f, 1.2f);

	//right
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.8f,-0.2f,1.6f);
    glVertex3f(-1.8f, -0.2f, 1.2f);
    glVertex3f(-1.8f, -3.0f, 1.2f);
    glVertex3f(-1.8f, -3.0f, 1.6f);

	//left
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-1.4f,-0.2f,1.6f);
    glVertex3f(-1.4f, -0.2f, 1.2f);
    glVertex3f(-1.4f, -3.0f, 1.2f);
    glVertex3f(-1.4f, -3.0f, 1.6f);

//left leg back front
	//front
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(-1.8f,-0.2f,-1.2f);
    glVertex3f(-1.4f, -0.2f, -1.2f);
    glVertex3f(-1.4f, -3.0f, -1.2f);
    glVertex3f(-1.8f, -3.0f, -1.2f);

    //back
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(-1.8f,-0.2f,-1.6f);
    glVertex3f(-1.4f, -0.2f, -1.6f);
    glVertex3f(-1.4f, -3.0f, -1.6f);
    glVertex3f(-1.8f, -3.0f, -1.6f);

    //right
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-1.8f,-0.2f,-1.6f);
    glVertex3f(-1.8f, -0.2f, -1.2f);
    glVertex3f(-1.8f, -3.0f, -1.2f);
    glVertex3f(-1.8f, -3.0f, -1.6f);

    //left
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-1.4f,-0.2f,-1.6f);
    glVertex3f(-1.4f, -0.2f, -1.2f);
    glVertex3f(-1.4f, -3.0f, -1.2f);
    glVertex3f(-1.4f, -3.0f, -1.6f);

//chair back
    //front
    glColor3f(1,0,0);
    //glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-1.8f, 0.2f, -1.8f);
    glVertex3f(1.8f, 0.2f, -1.8f);
    glVertex3f(1.8f, 3.5f, -1.8f);
    glVertex3f(-1.8f, 3.5f, -1.8f);

    //back
    //glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-1.8f, 0.2f, -2.0f);
    glVertex3f(1.8f, 0.2f, -2.0f);
    glVertex3f(1.8f, 3.5f, -2.0f);
    glVertex3f(-1.8f, 3.5f, -2.0f);


    //glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-1.8f, 0.2f, -2.0f);
    glVertex3f(-1.8f, 3.5f, -2.0f);
    glVertex3f(-1.8f, 3.5f, -1.8f);
    glVertex3f(-1.8f, 0.2f, -1.8f);

    glVertex3f(1.8f, 0.2f, -2.0f);
    glVertex3f(1.8f, 3.5f, -2.0f);
    glVertex3f(1.8f, 3.5f, -1.8f);
    glVertex3f(1.8f, 0.2f, -1.8f);

    glVertex3f(-1.8f, 3.5f, -2.0f);
    glVertex3f(-1.8f, 3.5f, -1.8f);
    glVertex3f(1.8f, 3.5f, -1.8f);
    glVertex3f(1.8f, 3.5f, -2.0f);
    glEnd();
    glutSwapBuffers();
    }

void update(int value) {
    _angle += 1.5f;
    if (_angle > 360) {
      _angle -= 360;
    }
    glutPostRedisplay();
    glutTimerFunc(25, update, 0);
    }

int main(int argc, char** argv) {
    //Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(600, 600);

    //Create the window
    glutCreateWindow("Lighting");
    initRendering();

    //Set handler functions
    glutDisplayFunc(drawScene);
    glutReshapeFunc(handleResize);
    update(0);
    glutMainLoop();
    return 0;
    }
	
f = 500
rotXval = 90
rotYval = 90
rotZval = 90
distXval = 500
distYval = 500
distZval = 500

def onFchange(val):
    global f
    f = val
def onRotXChange(val):
    global rotXval
    rotXval = val
def onRotYChange(val):
    global rotYval
    rotYval = val
def onRotZChange(val):
    global rotZval
    rotZval = val
def onDistXChange(val):
    global distXval
    distXval = val
def onDistYChange(val):
    global distYval
    distYval = val
def onDistZChange(val):
    global distZval
    distZval = val

if __name__ == '__main__':

    #Read input image, and create output image
    src = cv2.imread('Chair.c')
    src = cv2.resize(src,(640,480))
    dst = np.zeros_like(src)
    h, w = src.shape[:2]

    #Create user interface with trackbars that will allow to modify the parameters of the transformation
    wndname1 = "Source:"
    wndname2 = "WarpPerspective: "
    cv2.namedWindow(wndname1, 1)
    cv2.namedWindow(wndname2, 1)
    cv2.createTrackbar("f", wndname2, f, 1000, onFchange)
    cv2.createTrackbar("Rotation X", wndname2, rotXval, 180, onRotXChange)
    cv2.createTrackbar("Rotation Y", wndname2, rotYval, 180, onRotYChange)
    cv2.createTrackbar("Rotation Z", wndname2, rotZval, 180, onRotZChange)
    cv2.createTrackbar("Distance X", wndname2, distXval, 1000, onDistXChange)
    cv2.createTrackbar("Distance Y", wndname2, distYval, 1000, onDistYChange)
    cv2.createTrackbar("Distance Z", wndname2, distZval, 1000, onDistZChange)

    #Show original image
    cv2.imshow(wndname1, src)

    k = -1
    while k != 27:

        if f <= 0: f = 1
        rotX = (rotXval - 90)*np.pi/180
        rotY = (rotYval - 90)*np.pi/180
        rotZ = (rotZval - 90)*np.pi/180
        distX = distXval - 500
        distY = distYval - 500
        distZ = distZval - 500

        # Camera intrinsic matrix
        K = np.array([[f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0,   1, 0]])

        # K inverse
        Kinv = np.zeros((4,3))
        Kinv[:3,:3] = np.linalg.inv(K[:3,:3])*f
        Kinv[-1,:] = [0, 0, 1]

        # Rotation matrices around the X,Y,Z axis
        RX = np.array([[1,           0,            0, 0],
                    [0,np.cos(rotX),-np.sin(rotX), 0],
                    [0,np.sin(rotX),np.cos(rotX) , 0],
                    [0,           0,            0, 1]])

        RY = np.array([[ np.cos(rotY), 0, np.sin(rotY), 0],
                    [            0, 1,            0, 0],
                    [ -np.sin(rotY), 0, np.cos(rotY), 0],
                    [            0, 0,            0, 1]])

        RZ = np.array([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                    [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                    [            0,            0, 1, 0],
                    [            0,            0, 0, 1]])

        # Composed rotation matrix with (RX,RY,RZ)
        R = np.linalg.multi_dot([ RX , RY , RZ ])

        # Translation matrix
        T = np.array([[1,0,0,distX],
                    [0,1,0,distY],
                    [0,0,1,distZ],
                    [0,0,0,1]])

        # Overall homography matrix
        H = np.linalg.multi_dot([K, R, T, Kinv])

        # Apply matrix transformation
        cv2.warpPerspective(src, H, (w, h), dst, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)

        # Show the image
        cv2.imshow(wndname2, dst)
        k = cv2.waitKey(1)
		

int specularLightstatus = 1;
int ambientLightStatus = 1;
void init(void);
void display(void);
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);
int main(int argc, char** argv)
{
   glutInit(&argc, argv);
   glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
   glutInitWindowSize (500, 500);
   glutInitWindowPosition (100, 100);
   glutCreateWindow ("Lighting");
   init();
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutKeyboardFunc(keyboard);
   glutMainLoop();
   return 0;
}
/*  Initialize material property, light source, lighting model,
 *  and depth buffer.
 */
void init(void)
{
   GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
   GLfloat mat_shininess[] = { 50.0 };
   GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
   glClearColor (0.0, 0.0, 0.0, 0.0);
   glShadeModel (GL_SMOOTH);
   glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
   glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
   glLightfv(GL_LIGHT0, GL_POSITION, light_position);
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);
}
void display(void)
{
   glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glutSolidSphere (1.0, 20, 16);
   glFlush ();
}
void reshape (int w, int h)
{
   glViewport (0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity();
   if (w <= h)
      glOrtho (-1.5, 1.5, -1.5*(GLfloat)h/(GLfloat)w,
         1.5*(GLfloat)h/(GLfloat)w, -10.0, 10.0);
   else
      glOrtho (-1.5*(GLfloat)w/(GLfloat)h,
         1.5*(GLfloat)w/(GLfloat)h, -1.5, 1.5, -10.0, 10.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}
void keyboard(unsigned char key, int x, int y)
{
   switch (key) {
      case 27:
         exit(0);
         break;
      case 'O':
      case 'o':
         if(specularLightstatus == 1)
         {
            glDisable(GL_LIGHTING);
            glutPostRedisplay();
            specularLightstatus = 0;
         }
         else
         {
             glEnable(GL_LIGHTING);
             glutPostRedisplay();
             specularLightstatus = 1;
         }
         break;
      case 'p':
      case     'P':
           if(ambientLightStatus == true)
         {
            glDisable(GL_LIGHT0);
            glutPostRedisplay();
            ambientLightStatus = false;
         }
         else
         {
             glEnable(GL_LIGHT0);
             glutPostRedisplay();
             ambientLightStatus = true;
         }
         break;
   }
} 

void main()
{
   vec4 pos = vec4(in_position.x, in_position.y, in_position.z, 1.0);
   out_world_pos = Model * pos;
   gl_Position = Projection * View * out_world_pos;
 
   [...]
 
   out_light_space_pos = LightViewProjection * out_world_pos;
} 

float
compute_shadow_factor(vec4 light_space_pos, sampler2D shadow_map)
{
   // Convert light space position to NDC
   vec3 light_space_ndc = light_space_pos.xyz /= light_space_pos.w;
 
   // If the fragment is outside the light's projection then it is outside
   // the light's influence, which means it is in the shadow (notice that
   // such sample would be outside the shadow map image)
   if (abs(light_space_ndc.x) > 1.0 ||
       abs(light_space_ndc.y) > 1.0 ||
       abs(light_space_ndc.z) > 1.0)
      return 0.0;
 
   // Translate from NDC to shadow map space (Vulkan's Z is already in [0..1])
   vec2 shadow_map_coord = light_space_ndc.xy * 0.5 + 0.5;
 
   // Check if the sample is in the light or in the shadow
   if (light_space_ndc.z > texture(shadow_map, shadow_map_coord.xy).x)
      return 0.0; // In the shadow
 
   // In the light
   return 1.0;
}  

VkSampler sampler;
VkSamplerCreateInfo sampler_info = {};
sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
sampler_info.anisotropyEnable = false;
sampler_info.maxAnisotropy = 1.0f;
sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
sampler_info.unnormalizedCoordinates = false;
sampler_info.compareEnable = false;
sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
sampler_info.magFilter = VK_FILTER_LINEAR;
sampler_info.minFilter = VK_FILTER_LINEAR;
sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
sampler_info.mipLodBias = 0.0f;
sampler_info.minLod = 0.0f;
sampler_info.maxLod = 100.0f;
 
VkResult result =
   vkCreateSampler(device, &sampler_info, NULL, &sampler);
   
   VkDescriptorImageInfo image_info;
image_info.sampler = sampler;
image_info.imageView = shadow_map_view;
image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
 
VkWriteDescriptorSet writes;
writes.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
writes.pNext = NULL;
writes.dstSet = image_descriptor_set;
writes.dstBinding = 0;
writes.dstArrayElement = 0;
writes.descriptorCount = 1;
writes.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
writes.pBufferInfo = NULL;
writes.pImageInfo = &image_info;
writes.pTexelBufferView = NULL;
 
vkUpdateDescriptorSets(ctx->device, 1, &writes, 0, NULL);