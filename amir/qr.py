import cv2 as cv
import numpy as np
import math

def Check(metka):
    module_len = sum(el['len'] for el in metka) / 7
    if abs(metka[0]['len'] / module_len - 1) < 0.5 and \
            abs(metka[1]['len'] / module_len - 1) < 0.5 and \
            abs(metka[2]['len'] / module_len - 1 * 3) < 0.5 and \
            abs(metka[3]['len'] / module_len - 1) < 0.5 and \
            abs(metka[4]['len'] / module_len - 1) < 0.5:
        if False and abs(metka[2]['len'] / module_len - 1 * 3) > 0.5:
            print(abs(metka[2]['len'] / module_len - 1 * 3))
            print(module_len)
            print(metka)
        return True
    return False
def Found(img, y1, y2, x1, x2):
    if y1 == y2:
        img[y1, x1: x2] = (x2-x1)/3
    else:
        img[y1:y2, x1] = (y2-y1)/3

def GetBoxQR(img):
    metka = []
    img2 = np.zeros(img.shape,np.float64)
    img3 = np.zeros(img.shape,np.float64)
    # проверка по x
    for y in range(img.shape[0]):
        metka = [{'pos': 0, 'color': img[y, 0] > 127, 'len': 0}]
        for x in range(img.shape[1]):
            #if img[y,x] > 70 and img[y,x] < 290:
            #    continue
            if metka[-1]['color'] == (img[y,x] > 127):
                continue
            metka[-1]['len'] = x - metka[-1]['pos']
            if len(metka) > 4:
                if Check(metka):
                    Found(img2, y, y, metka[2]['pos'], metka[2]['pos'] + metka[2]['len'])
                metka.pop(0)
            metka.append({'pos': x, 'color': img[y, x] > 127, 'len': 0})
        # проверка на границе
        metka[-1]['len'] = img.shape[1] - metka[-1]['pos']
        if len(metka) > 4:
            if Check(metka):
                Found(img2, y, y, metka[2]['pos'], metka[2]['pos'] + metka[2]['len'])
    # проверка по y
    for x in range(img.shape[1]):
        metka = [{'pos': 0, 'color': img[0, x] > 127, 'len': 0}]
        for y in range(img.shape[0]):
            #if img[y, x] > 70 and img[y, x] < 290:
            #    continue
            if metka[-1]['color'] == (img[y,x] > 127):
                continue
            metka[-1]['len'] = y - metka[-1]['pos']
            if len(metka) > 4:
                if Check(metka):
                    Found(img3, metka[2]['pos'], metka[2]['pos'] + metka[2]['len'], x, x)
                metka.pop(0)
            metka.append({'pos': y, 'color': img[y, x] > 127, 'len': 0})
        # проверка на границе
        metka[-1]['len'] = img.shape[0] - metka[-1]['pos']
        if len(metka) > 4:
            if Check(metka):
                Found(img3, metka[2]['pos'], metka[2]['pos'] + metka[2]['len'], x, x)

    # проверка по y
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img2[y, x] = img2[y, x] * img3[y, x]


    img4 = np.zeros(img.shape,np.uint32)
    countGroup = 0
    Groups=[{'count':0}]
    for y in range(img2.shape[0]):
        for x in range(img2.shape[1]):
            if img2[y,x] != 0 and x != 0 and y != 0:
                id = 0
                if img4[y, x-1] != 0:
                    id = img4[y, x-1]
                elif img4[y-1, x] != 0:
                    id = img4[y-1, x]
                if id == 0:
                    Groups.append({'count': 1, 'cx': x, 'cy': y})
                    countGroup += 1
                    img4[y,x] = countGroup
                else:
                    img4[y,x] = id
                    Groups[id]['count'] += 1
                    Groups[id]['cx'] += x
                    Groups[id]['cy'] += y
    Groups.sort(key=lambda el:el['count'], reverse=True)
    #Groups = Groups[:2]
    for i in range(len(Groups)-1):
        Groups[i]['cx'] /= Groups[i]['count']
        Groups[i]['cy'] /= Groups[i]['count']
        Groups[i]['m_size'] = math.sqrt(img2[int(Groups[i]['cy']), int(Groups[i]['cx'])])
    if len(Groups)<4:
        img2 = cv.normalize(img2, None, 0, 1.0,
                            cv.NORM_MINMAX)
        return img2

    v12 = np.array([Groups[1]['cx'] - Groups[0]['cx'], Groups[1]['cy'] - Groups[0]['cy'], 0])
    v13 = np.array([Groups[2]['cx'] - Groups[0]['cx'], Groups[2]['cy'] - Groups[0]['cy'], 0])
    v23 = np.array([Groups[2]['cx'] - Groups[1]['cx'], Groups[2]['cy'] - Groups[1]['cy'], 0])
    Ver = []
    if np.linalg.norm(v23) > np.linalg.norm(v12) and np.linalg.norm(v23) > np.linalg.norm(v13):
        Ver.append(Groups[0])
        if np.cross(v13, v23)[2] > 0:
            Ver.append(Groups[1])
            Ver.append(Groups[2])
        else:
            Ver.append(Groups[2])
            Ver.append(Groups[1])
    elif np.linalg.norm(v13) > np.linalg.norm(v12) and np.linalg.norm(v13) > np.linalg.norm(v23):
        Ver.append(Groups[1])
        if np.cross(v13,v23)[2] > 0:
            Ver.append(Groups[0])
            Ver.append(Groups[2])
        else:
            Ver.append(Groups[2])
            Ver.append(Groups[0])
    else:
        Ver.append(Groups[2])
        if np.dot(v13,v23)[2] > 0:
            Ver.append(Groups[1])
            Ver.append(Groups[0])
        else:
            Ver.append(Groups[0])
            Ver.append(Groups[1])


    vx = np.array([Ver[1]['cx'] - Ver[0]['cx'], Ver[1]['cy'] - Ver[0]['cy'], 0])
    vx = vx / np.linalg.norm(vx)
    vy = np.array([Ver[2]['cx'] - Ver[0]['cx'], Ver[2]['cy'] - Ver[0]['cy'], 0])
    vy = vy / np.linalg.norm(vy)
    vp = np.array([-(Ver[0]['cx'] - 3.5 * (vx[0]+vy[0])*Ver[0]['m_size']), -(Ver[0]['cy'] - 3.5 * (vx[1]+vy[1])*Ver[0]['m_size'])])
    M = np.array([[vx[0], vx[1], vp[0]*vx[0]+vp[1]*vx[1]],[vy[0], vy[1], vp[0]*vy[0]+vp[1]*vy[1]]])
    img5 = cv.warpAffine(img, M, img.shape)
    cv.imshow(winname='qr', mat=img5)
    # cv.imshow(winname='test', mat=imgb)
    cv.waitKey(0)

    img2 = cv.normalize(img2, None, 0, 1.0,
                                   cv.NORM_MINMAX)
    return img2




img = cv.imread('qrr.jpg', cv.IMREAD_COLOR)
imgb = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow(winname='test', mat=img)
#cv.waitKey(0)
img2 = GetBoxQR(imgb)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        imgb[y, x] = (imgb[y, x] > 127) * 255
        img[y, x, 0] = max(imgb[y, x]/2, img2[y, x]*255)
img[:, :, 2] = imgb/2
img[:, :, 1] = imgb/2
cv.imshow(winname='test', mat=img)
#cv.imshow(winname='test', mat=imgb)
cv.waitKey(0)

