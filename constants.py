import torch
import numpy as np

#이미지 크기
WIDTH = 64
HEIGHT = 64

#그리드 크기
GRID_WIDTH = 8
GRID_HEIGHT = 8

#그리드 한 칸의 길이 ex)256//16 = 16
ONE_GRID_WIDTH = WIDTH //GRID_WIDTH
ONE_GRID_HEIGHT = HEIGHT //GRID_HEIGHT

#대각선 길이, 최대 거리차. (정확도 측정에 이용)
#WIDTH/HEIGHT 크기 삼각형의 대각선이 아니라
#중앙점 기준이므로 (WIDTH-ONE_GRID_WIDTH)/(HEIGHT-ONE_GRID_HEIGHT) 크기의 삼각형의 대각선.
DIAGONOL= np.sqrt(np.square(WIDTH-ONE_GRID_WIDTH) + np.square(HEIGHT-ONE_GRID_HEIGHT))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#image/annotaion file 경로 
#1- 1000개 라벨링 한 경로
#2- 추가 라벨링한 2000개 set

imgPath = ".././data/1000sets/images" #이미지들 path
labelPath = ".././data/1000sets/annotations.xml" #annotations path

imgPath_2 = ".././data/2000sets/images" #이미지들 path
labelPath_2 = ".././data/2000sets/annotations.xml" #annotations path