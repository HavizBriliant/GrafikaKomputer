#include <windows.h>
#include <GL/glut.h>
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
    glLoadIdentity();
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
    src = cv2.imread('table.c')
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