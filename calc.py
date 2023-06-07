import numpy as np
import constants as c

#두 점 거리를 구하는 함수 - 리스트 입력
def two_point_distance(a_x,a_y,b_x,b_y):
    distance=[]
    for i in range(len(a_x)):
        distance.append(np.sqrt(np.square(a_x[i]-b_x[i]) + np.square(a_y[i]-b_y[i])))
    return distance


#두 점 거리를 구하는 함수 - 단수 입력
def two_point_distance_single(a_x,a_y,b_x,b_y):

    distance= np.sqrt(np.square(a_x-b_x) + np.square(a_y-b_y))

    return distance

#정확도 구하는 함수
def getAccuracy(y,pred):
    #cuda -> cpu
    
    a = y.cpu().numpy()
    b = pred.cpu().numpy()

    a_x = (a%c.GRID_WIDTH)*c.ONE_GRID_WIDTH+(c.ONE_GRID_WIDTH//2)
    a_y = (a//c.GRID_WIDTH)*c.ONE_GRID_HEIGHT+(c.ONE_GRID_HEIGHT//2)

    b_x = (b%c.GRID_WIDTH)*c.ONE_GRID_WIDTH+(c.ONE_GRID_WIDTH//2)
    b_y = (b//c.GRID_WIDTH)*c.ONE_GRID_HEIGHT+(c.ONE_GRID_HEIGHT//2)

    distance = two_point_distance(a_x,a_y,b_x,b_y)

    accuracy = np.mean(1-(distance/c.DIAGONOL))

    return accuracy

