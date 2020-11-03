#!env python

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import cv2
from evaluation import checkMaxMin
from setVerts import setVertsFromImg, setVertsFromNpy

LeftButtonOn = False
RightButtonOn = False
Angle1 = 0
Angle2 = 0
Distance = 7.0
px, py = -1, -1


def mouse(button, state, x, y):
    global LeftButtonOn, RightButtonOn
    if button == GLUT_LEFT_BUTTON:
        if state == 1:
            LeftButtonOn = False
        elif state == 0:
            LeftButtonOn = True

    if button == GLUT_RIGHT_BUTTON:
        if state == GLUT_UP:
            RightButtonOn = False
        elif state == GLUT_DOWN:
            RightButtonOn = True


def motion(x, y):
    global RightButtonOn, LeftButtonOn, Angle1, Angle2, Distance, px, py
    if LeftButtonOn == True and RightButtonOn == True:
        Angle1 = 0
        Angle2 = 0
        Distance = 7.0
        gluLookAt(0, 0, 7.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    elif LeftButtonOn == True:
        if px >= 0 and py >= 0:
            Angle1 += float(-(x - px) / 100)
            Angle2 += float((y - py) / 100)
        px = x
        py = y
    elif RightButtonOn == True:
        if px >= 0 and py >= 0:
            Distance += float(y - py) / 20
        px = x
        py = y
    else:
        px = -1
        py = -1

    # print(LeftButtonOn, RightButtonOn, Angle1, Angle2, Distance, px, py, x, y)
    glutPostRedisplay()


def keyboard(key, x, y):
    global Angle2
    if key.decode() == "\033":  # Escape
        sys.exit()
    elif key.decode() == "q":
        sys.exit()
    elif key.decode() == "j":
        Angle2 += 1.0
    else:
        print(key.decode())


def resize_cb(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30.0, w / h, 1, 1000.0)  # 視体積を設定することができる
    glMatrixMode(GL_MODELVIEW)


def draw():
    global Angle1, Angle2, verts

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(
        Distance * np.cos(Angle2) * np.sin(Angle1),
        Distance * np.sin(Angle2),
        Distance * np.cos(Angle2) * np.cos(Angle1),
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
    )
    glPointSize(2)
    glBegin(GL_POINTS)
    for vert in verts:
        glColor3d(vert[3], vert[4], vert[5])
        glVertex3f(vert[0], vert[1], vert[2])
    glEnd()
    glFlush()
    glutSwapBuffers()


glutInit(sys.argv)
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutInitWindowSize(320, 320)
glutCreateWindow("PyOpenGL 11")
glutDisplayFunc(draw)
glutReshapeFunc(resize_cb)
glutKeyboardFunc(keyboard)
glutMouseFunc(mouse)
glutMotionFunc(motion)

glClearColor(0.0, 0.0, 1.0, 0.0)
glEnable(GL_DEPTH_TEST)
verts_np3d = setVertsFromNpy()
# np.savetxt(
#     "fromImg.txt",
#     np.reshape(verts_np3d, (verts_np3d.shape[0] * verts_np3d.shape[1], 6)),
# )
verts = list(np.reshape(verts_np3d, (verts_np3d.shape[0] * verts_np3d.shape[1], 6)))
checkMaxMin(verts)
print("len_verts:", len(verts))
glutMainLoop()
