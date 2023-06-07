'''데이터셋 정의'''
'''Airplanes - custom data set으로 x는 256/256 크기의 비행기 이미지, y는 앞바퀴가 포함된 그리드의 index 번호'''
'''FoldData - K-fold별 x,y,pred 정보, y와 pred의 거리차, 일치 여부들을 저장함'''

import get_data
import get_transformed_data as g
import torch
import calc

class Airplanes:
    def __init__(self) -> None:
        self.x,self.y = g.get_dataset()
        # self.x,self.y = g.get_grey_dataset()

    def __len__(self):
        return len(self.x) 
           
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]

        return x,y

class FoldData:
    def __init__(self,idx,y_index, pred_index):
        self.idx = idx
        self.y_index = torch.stack(y_index,0).cpu().numpy() #y값
        self.pred_index = torch.stack(pred_index,0).cpu().numpy() #pred값
        self.y_x, self.y_y = get_data.oneHot_to_cord(y_index) #y의 x,y좌표
        self.pred_x, self.pred_y = get_data.oneHot_to_cord(pred_index) #pred의 x,y좌표
        self.distance = calc.two_point_distance(self.y_x, self.y_y, self.pred_x, self.pred_y) #y와 pred의 거리
        self.is_correspond = [y == p for y, p in zip(self.y_index, self.pred_index)] #y와 pred의 일치 여부
    
    def __len__(self):
        return len(self.y_index) 
    
    def __getitem__(self,idx):
        y_index = self.y_index[idx]
        pred_index=self.pred_index[idx]
        y_x=self.y_x[idx]
        y_y=self.y_y[idx]
        pred_x=self.pred_x[idx]
        pred_y=self.pred_y[idx]
        distance=self.distance[idx]
        is_correspond=self.is_correspond[idx]
        return y_index,pred_index,y_x,y_y,pred_x,pred_y,distance,is_correspond
    
