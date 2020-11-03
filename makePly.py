#!env python

# plyを作成するコード
# xは全体
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import cv2

# from readFile import readCg
import os
import scipy.io as sio
import re


def readCg(cgPath):
    patternList = ["focal_length_mm", "sensor_size_mm", "baseline_mm"]
    paraDict = {}
    with open(cgPath) as f:
        s = f.read()
        sLines = s.split("\n")
        for sLine in sLines:
            for pattern in patternList:
                if re.match(pattern, sLine):
                    sList = sLine.split()
                    paraDict[pattern] = float(sList[2])
    print(paraDict)
    return paraDict


def matLoad(u, v):
    mat = sio.loadmat(
        "/home/takashi/Desktop/dataset/from_iwatsuki/mat_file/additional_disp_mat/%s.mat"
        % LFName
    )
    disp_gt = mat["depth"]
    # prin(disp_gt.shape)
    return disp_gt[u][v]


# basePath = "/home/takashi/Desktop/dataset/lf_dataset/additional"
u, v = 4, 4  # 0~8(uが→方向　vが下方向)
camNum = u * 9 + v

basePath = "/home/takashi/Desktop/dataset/lf_dataset/additional"
LFName = "antinous"
cfgName = "parameters.cfg"
imgName = "input_Cam{:03}.png".format(camNum)
cgPath = os.path.join(basePath, LFName, cfgName)
imgPath = os.path.join(basePath, LFName, imgName)
img = cv2.imread(imgPath)
dispImg = matLoad(u, v)
dMin = np.min(dispImg)


width = img.shape[1]
height = img.shape[0]
verts = []

paraDict = readCg(cgPath)
f_mm = paraDict["focal_length_mm"]
s_mm = paraDict["sensor_size_mm"]
b_mm = paraDict["baseline_mm"]
longerSide = max(dispImg.shape[0], dispImg.shape[1])
beta = b_mm * f_mm * longerSide
f_pix = (f_mm * longerSide) / s_mm


def setVerts(img, depthImg, paraDict):
    global verts
    print(img.shape, depthImg.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            colors = [
                float(img[x][y][2] / 255.0),
                float(img[x][y][1] / 255.0),
                float(img[x][y][0] / 255.0),
            ]
            X, Y, Z = pix2m_disp(x, y, paraDict)

            vert = np.array([X, Y, Z, colors[0], colors[1], colors[2]])
            verts.append(vert)
    # print(verts)
    return verts


def pix2m_disp(x, y, paraDict):
    if dispImg[x][y]:
        # Z = b_mm * f_pix / float(dispImg[x][y])
        Z = float(beta * f_mm) / float((dispImg[x][y] * f_mm * s_mm + beta))
        # Z=float(dispImg[x][y])+dMin
    else:
        print("zero!!")
        Z = 0
    X = float(x) * Z / f_pix
    Y = float(y) * Z / f_pix

    return X, Y, Z


if __name__ == "__main__":
    verts = setVerts(img, dispImg, paraDict)
    verts_np = np.array(verts)
    verts_reshape = np.reshape(verts_np, (512, 512, 6))
    np.save("verts_reshape", verts_reshape)
    # ここで出てくるplyはワールド座標、x,y,zともに単位はmm
    # OpenGLで描画するときはこれを正規化して、見やすくする
