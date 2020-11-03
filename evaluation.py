import numpy as np
import cv2

checkC_F = True


def normalization(array):
    max = np.max(array)
    min = np.min(array)
    dst = np.zeros(array.shape)
    for x in range(array.shape[1]):
        for y in range(array.shape[0]):
            dst[x][y] = float(array[x][y] - min) / float(max - min)
    return dst

def checkMaxMin(verts_list):
    max, min,ave = [], [],[]
    view={}
    # print(verts.ndim)
    verts = np.array(verts_list)
    if verts.ndim == 2:
        for i in range(6):
            max.append(np.max(verts[:, i]))
            min.append(np.min(verts[:, i]))
            ave.append(np.average(verts[:,i]))

    elif verts.ndim == 3:
        for i in range(6):
            max.append(np.max(verts[:, :, i]))
            min.append(np.min(verts[:, :, i]))
            ave.append(np.average(verts[:,:,i]))
    # view.append(max)
    # view.append(min)
    view["max"]=max
    view["min"]=min
    view["ave"]=ave
    for key,value in view.items():
        print(key,value)
    # print(view)


if __name__ == "__main__":
    pre_verts = np.load("verts_reshape.npy")
    verts = vertsNormalization(pre_verts)
    checkMaxMin(np.array(verts))

    if checkC_F and np.array(verts).ndim == 3:
        cv2.imwrite("./c1_c6/x.png", verts[:, :, 0])
        cv2.imwrite("./c1_c6/y.png", verts[:, :, 1])
        cv2.imwrite("./c1_c6/z.png", verts[:, :, 2])
        cv2.imwrite("./c1_c6/c1.png", verts[:, :, 3] * 255)
        cv2.imwrite("./c1_c6/c2.png", verts[:, :, 4] * 255)
        cv2.imwrite("./c1_c6/c3.png", verts[:, :, 5] * 255)
