import numpy as np
import cv2

# from normalization import checkMaxMin
# from matLoader import makeDepthImg
from sklearn import preprocessing
import scipy.io as sio

img = cv2.imread("tower/input_Cam000.png")
depthImg = cv2.imread("tower/depth_0_0.png", 0)

width = img.shape[1]
height = img.shape[0]
maxD = np.max(depthImg)
minD = np.min(depthImg)
ratio = 0.000003
verts = []
vert_point = []


def makeDepthImg():
    file_name = "tower"

    mat = sio.loadmat(
        "/home/takashi/Desktop/dataset/from_iwatsuki/mat_file/additional_disp_mat/%s.mat"
        % file_name
    )
    depth_gt = mat["depth"]
    mm = preprocessing.MinMaxScaler()
    min0_max1 = mm.fit_transform(depth_gt[0][0])
    # cv2.imwrite(
    #     "./tower/depth_%d_%d.png" % (0, 0), min0_max1 * 255,
    # )
    print("depth", min0_max1.shape)
    return min0_max1


def setVertsFromImg():
    global verts
    colors = []
    points = []
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            point = [float(x), float(y), float(depthImg[x][y])]
            color = [
                float(img[x][y][2] / 255.0),
                float(img[x][y][1] / 255.0),
                float(img[x][y][0] / 255.0),
            ]
            colors.append(color)
            points.append(point)
    # この時点でcolorsがlenは256*256,3ch
    # この時点でpointsがlenは256*256,3ch
    # pointを正規化して、全て0~1に変更
    points_np3d = np.reshape(np.array(points), img.shape)
    # points_np=pointsNormal(points_np3d)
    points_np = mmNormal(points_np3d)
    colors_np = np.reshape(np.array(colors), img.shape)
    verts = np.concatenate((points_np, colors_np), axis=2)
    return verts


def setVertsFromNpy():
    global verts
    npyVerts = np.load("verts_reshape.npy")
    colors = npyVerts[:, :, 3:6]
    points = npyVerts[:, :, 0:3]
    points_np3d = np.reshape(np.array(points), img.shape)
    points_np = mmNormal(points_np3d)
    colors_np = np.reshape(np.array(colors), img.shape)
    verts = np.concatenate((points_np, colors_np), axis=2)
    return verts


def pointsNormal(points_np3d):
    points_np3d_Normed = mmNormal(points_np3d)
    return points_np3d_Normed


def mmNormal(array):
    max, min = [], []
    dst_3d = np.zeros(array.shape)
    for i in range(3):
        max.append(np.max(array[:, :, i]))
        min.append(np.min(array[:, :, i]))
        for x in range(array.shape[1]):
            for y in range(array.shape[0]):
                dst_3d[x][y][i] = (
                    2 * float(array[x][y][i] - min[i]) / float(max[i] - min[i]) - 1
                )
    return dst_3d


if __name__ == "__main__":
    verts = setVertsFromNpy()
    print(verts.shape)
