import constants as c
import calc
import os
import cv2
import numpy as np
import get_data as g

def makeDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

'''
point : img에 x,y 좌표에 점 찍는 함수. 
-> x,y는 해당 grid의 왼쪽 위 좌표 이므로 
그리드 칸 내 중앙 값으로 찍기 위해 
x,y에 그리드 한 칸 길이//2 값을 더해 줌 
'''
def point(img,x,y,color,thickness):
    
    recimg = cv2.line(img,(x+(c.ONE_GRID_WIDTH//2),y+(c.ONE_GRID_HEIGHT//2)),(x+(c.ONE_GRID_WIDTH//2),y+(c.ONE_GRID_HEIGHT//2)),color=color, thickness = thickness)

    return recimg

'''
y값과 pred값이 일치하지 않은 경우에 
각각의 점을 찍고 이미지로 출력함.
오차 거리(distance 마다) 디렉토리를 만들어 구분.
-> 오차 확인 용
'''
def genOutput(x,y,pred,path,batch_cnt):

    x = x.cpu()
    x = np.array(x.permute(0,2,3,1))
    x = x.astype(np.uint8)
    

    x_cor,y_cor = g.oneHot_to_cord(pred)
    cx_cor,cy_cor = g.oneHot_to_cord(y)

    

    for i in range(len(pred)):
        if y[i] != pred[i]:
            img =[]
            img=x[i].copy()
            distance = calc.two_point_distance_single(x_cor[i],y_cor[i],cx_cor[i],cy_cor[i])

            makeDirectory(path+f'/{distance}')
            new_img_name = path+f'/{distance}/batch{batch_cnt}_false{i}.jpg'

            recimg = point(img,cx_cor[i],cy_cor[i],(0,255,0),12)
            recimg = point(img,x_cor[i],y_cor[i],(255,0,0),12)

            extension = os.path.splitext(new_img_name)[1] # 이미지 확장자
            
            result, encoded_img = cv2.imencode(extension, recimg)
            
            if result:
                with open(new_img_name, mode='w+b') as f:
                    encoded_img.tofile(f)