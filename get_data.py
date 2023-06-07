import constants as c
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

'''이미지 가져오기 /  resize'''
#png로 끝나는 이미지들 이름들 list로 저장

def getImageNameList(imgPath):
    file_list = os.listdir(imgPath)
    imageNameList = [file for file in file_list if file.endswith(".png")]

    return imageNameList


#이미지들을 가져와서 image라는 변수에 저장

def getImages(imgPath):
    images = []
    imageNameList = getImageNameList(imgPath)    
    for file in imageNameList:
        full_path = imgPath+'/'+file

        img_array = np.fromfile(full_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        images.append(img)

    return images    

#image 들을 (c.WIDTH,c.HEIGHT)크기 만큼 resize한 후 반환

def getResizedImages(images):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i],(c.WIDTH,c.HEIGHT))

    return images

        
'''좌표 가져오기 /  변환'''

#바퀴 좌표 전체를 받아옴

def getCord():
    xml = open(c.labelPath, "r")
    tree = ET.parse(xml)
    root = tree.getroot()
    images = root.findall("image")
    label_x = []
    label_y = []
    x = 0
    y = 0

    for image in images:

        points = image.find('points')
        point = points.get('points')

        x,y = map(float, point.split(','))
        label_x.append(x)
        label_y.append(y)


    return label_x,label_y

#바퀴 좌표 전체를 변환된 비율로 계산 

def getResizedCord(labelPath):
    xml = open(labelPath, "r",encoding='UTF8')
    tree = ET.parse(xml)
    root = tree.getroot()
    images = root.findall("image")
    label_x = []
    label_y = []
    x = 0
    y = 0

    for image in images:
        points = image.find('points')
        point = points.get('points')
        width = float(image.get('width'))
        height = float(image.get('height')) 

        x,y = map(float, point.split(','))

        label_x.append(x*c.WIDTH/width)
        label_y.append(y*c.HEIGHT/height)


    return label_x,label_y

''' 
cord/grid/onehot 사이의 변환

cord : 원래 이미지에서 256by256 크기로 변환된 이미지에 맞게 비율 변환된 앞바퀴 좌표
grid : 변환된 좌표가 속하는 grid의 왼쪽 위 꼭짓점 좌표 
onehot : 변환된 좌표가 속하는 grid의 index (8by8 grid 상에서는 64개 중 1)

ex) cord_to_grid : 변환된 좌표를 해당 grid의 꼭짓점 좌표로 바꿔줌

'''


'''cord->grid->onehot : 좌표를 grid index로 변환하여 y값으로 정의하는 과정임'''

def cord_to_grid(x,y):
    
    x_index = []
    y_index = []

    for i in range(len(x)): 

        w=int(x[i]//c.ONE_GRID_WIDTH)
        h=int(y[i]//c.ONE_GRID_HEIGHT)

        if w == c.GRID_WIDTH: w=c.GRID_WIDTH-1
        if h == c.GRID_HEIGHT: h=c.GRID_HEIGHT-1

        x_index.append(w)
        y_index.append(h)     

    return x_index,y_index


def grid_to_onehot(x,y):
    list = []

    for i in range(len(x)): 
        list.append((y[i]*c.GRID_WIDTH)+x[i])

    return list

def cord_to_onehot(x,y):
    
    x,y = cord_to_grid(x,y)
    list = grid_to_onehot(x,y)

    return list


''' 
예측 결과 출력을 위해
반대로 onehot->cord로 변환하는 부분
'''
def oneHot_to_grid(numList):
    
    x = []
    y = []
    for i in range(len(numList)): 

        x.append(int(numList[i]%c.GRID_WIDTH))
        y.append(int(numList[i]//c.GRID_HEIGHT))

    return x,y

def grid_to_cord(x,y):
    
    x1 = []
    y1 = []

    for i in range(len(x)): 

        x1.append(x[i]*c.ONE_GRID_WIDTH)
        y1.append(y[i]*c.ONE_GRID_HEIGHT)


    return x1,y1

def oneHot_to_cord(numList):

    x,y = oneHot_to_grid(numList)
    x,y = grid_to_cord(x,y)
    
    return x,y


