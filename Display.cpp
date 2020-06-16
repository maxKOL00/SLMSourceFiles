#include "Display.h"

static byte* SLM_IMAGE_PTR;

// Image dimensions
static size_t				SLM_PX_H;
static size_t				SLM_PX_V;

// 1/frame rate, the actual frame rate can be lower if the computations is expensive
static size_t               REFRESH_DELAY;

// Timer and termination condition
static StopWatchInterface*  TIMER = NULL;
static bool				    FINISHED;


static void				    display(void);
static void					timerEvent(int value);


// I did not spent too much time on this code I was just happy when it worked
// Drawing/displaying static 2D images is definitely not the main purpose of 
// glut/opengl and its probably overkill (which is why we can afford to do things
// less efficiently than possible)

void init_window(const Parameters& params) {
    std::cout << "Creating OpenGL window\n";

    REFRESH_DELAY = double(1000.0 / params.get_frame_rate() + 0.5);
    SLM_PX_H = params.get_slm_px_x();
    SLM_PX_V = params.get_slm_px_y();
    
    // Dummy variables because glut is super unflexible
    int argc = 0;
    char* argv[1] = { (char*)" " };
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    //glutInitWindowSize(width, height);
    glutCreateWindow("SLM Image");
    glutPositionWindow(2600, 0);

    glutDisplayFunc(display);
    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glutTimerFunc((unsigned int)REFRESH_DELAY, timerEvent, 0);
    sdkStartTimer(&TIMER);

    for (size_t i = 0; i < 5; i++) {
        Sleep(200);
        glutMainLoopEvent();
    }

}

void display_phasemap(byte* image_ptr) {
    SLM_IMAGE_PTR = image_ptr;
    glutMainLoopEvent();
}


void display(void) {

    glutFullScreen();
    sdkStartTimer(&TIMER);
    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
        
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_LUMINANCE,
        (GLsizei)SLM_PX_H,
        (GLsizei)SLM_PX_V,
        0,
        GL_LUMINANCE, // GL_LUMINANCE,
        GL_UNSIGNED_BYTE,
        &SLM_IMAGE_PTR[0]
    );

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0, 1.0);
    glEnd();

    sdkStopTimer(&TIMER);
    glutSwapBuffers();
    
}


void timerEvent(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc((unsigned int)REFRESH_DELAY, timerEvent, 0);
    }
}
