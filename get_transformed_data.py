import random
import get_data as d
import albumentations as A
import torch
import numpy as np
import constants as c
import cv2
from torchvision import transforms

'''
albumentations 사용 시 
좌표가 이미지의 가장 자리에 위치하는 경우
transform사용 시 error 발생.

따라서 이런 경우 좌표 값에 1을 빼서 
이미지 위에 위치하도록 precheckKeypoints로 정의
'''
def precheckKeypoints(keypoints):
    checkedkeypoints = []
    for i in range(len(keypoints)):
        for (x, y) in keypoints[i]:
            if int(x) == c.WIDTH:
                keypoints[i][0][0] = x-1
            if int(y) == c.HEIGHT:
                keypoints[i][0][1] = y-1

            checkedkeypoints.append(keypoints[i])
                
    
    return checkedkeypoints

'''
데이터 변형후 zoom/crop등의 이유로 
라벨이 이미지 내에 포함 안돼는 경우
y값이 없는 데이터가 생김.

이러한 경우 해당 데이터를 제외하기 위해
checkData를 정의함
'''
def checkData(transformed_image,transformed_keypoints):
    checkedImage = []
    checkedkeypoints = []

    if len(transformed_image) != len(transformed_keypoints):
        print('image 갯수 != keypoints 갯수 error')

    else:
        for i in range(len(transformed_image)):
            if len(transformed_keypoints[i]) == 0:
                continue
            else:
                checkedImage.append(transformed_image[i])
                checkedkeypoints.append(transformed_keypoints[i])
    
    return checkedImage,checkedkeypoints

'''
albumentations 함수들 정의

증강 방법

-horizonalFlip : 좌우 반전
-VerticalFlip : 상하 반전
-RandomCrop : Random한 부분을 crop
-rotated : Random한 각도로 회전
-CenterCrop : 중심을 기준 150/150크기로 crop
-ShiftScaleRotate : Random하게 이미지를 shift/회전/scale함
-complex : 여러 요소들을 Random하게 적용
-ChannelShuffle
-PixelDropout : Pixel을 랜덤하게 정해진 비율만큼 dropout함

영상 전처리 방법

-CLAHE
-ToGray : 흑백 이미지 (단일 채널) 변환 시에 사용
'''

def horizonalFlip(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    transform = A.Compose(
        [A.HorizontalFlip(p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def VerticalFlip(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    transform = A.Compose(
        [A.VerticalFlip(p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    ) 

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def RandomCrop(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    random.seed(7)
    transform = A.Compose(
        [A.RandomCrop(width=150, height=150, p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def rotated(image,keypoints):

    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    random.seed(7)
    transform = A.Compose(
        [A.Rotate(p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i].copy(), keypoints=keypoints[i].copy())
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def CenterCrop(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    transform = A.Compose(
        [A.CenterCrop(height=150, width=150, p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def ShiftScaleRotate(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    random.seed(14)
    transform = A.Compose(
        [A.ShiftScaleRotate(p=1)], #p -> 데이터에 해당 transform 적용하는 비율.1로 설정하여 데이터전체 적용
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def complex(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    random.seed(7)
    transform = A.Compose([
            #A.RandomSizedCrop(min_max_height=(20, 256), height=150, width=150, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.HueSaturationValue(p=0.5), 
                A.RGBShift(p=0.7)
            ], p=1),                          
            A.RandomBrightnessContrast(p=0.5)
        ], 
        keypoint_params=A.KeypointParams(format='xy'),
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def ChannelShuffle(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    transform = A.Compose(
        [A.ChannelShuffle(p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def PixelDropout(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    transform = A.Compose(
        [A.PixelDropout(dropout_prob=0.01,p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints


def CLAHE(image,keypoints):
    precheckKeypoints(keypoints)
    transformed_image = []
    transformed_keypoints = []

    transform = A.Compose(
        [A.CLAHE(p=1)], 
        keypoint_params=A.KeypointParams(format='xy')
    )

    for i in range(len(image)):
        transformed = transform(image=image[i], keypoints=keypoints[i])
        transformed_image.append(transformed['image'])
        transformed_keypoints.append(transformed['keypoints'])

    checkedImage,checkedkeypoints = checkData(transformed_image,transformed_keypoints)

    return checkedImage,checkedkeypoints

def ToGray(image):
    transformed_image = []

    for i in range(len(image)):
        transformed_image.append(cv2.cvtColor(image[i],cv2.COLOR_BGR2GRAY))
    return transformed_image

'''데이터 셋 준비 함수'''
'''get_dataset - 컬러 (3 channel)'''
'''get_grey_dataset - 흑백 (1 channel)'''

def get_dataset():
    #이미지/좌표 준비
    image = d.getResizedImages(d.getImages(c.imgPath))
    image2 = d.getResizedImages(d.getImages(c.imgPath_2))

    x,y = d.getResizedCord(c.labelPath)
    x_2,y_2 = d.getResizedCord(c.labelPath_2)

    image = image+image2
    x = x+x_2
    y = y+y_2
    
    #좌표 keypoints 형식으로 준비(albumentation keypoints 형식)
    keypoints = [[[x,y]] for x,y in zip(x,y)]

    dataset_x = []
    dataset_y = []

    dataset_x = image.copy()
    dataset_y = keypoints.copy()

    '''데이터 증강'''
    tempImgs,tempKeys = horizonalFlip(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = rotated(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = ShiftScaleRotate(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = complex(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = ChannelShuffle(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = PixelDropout(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys



    '''이미지 전처리'''
    dataset_x,dataset_y = CLAHE(dataset_x,dataset_y)
    
    '''shape 변환 - x'''
    dataset_x = np.array(dataset_x)
    dataset_x = torch.FloatTensor(dataset_x)
    dataset_x = dataset_x.permute(0,3,1,2)

    '''shape 변환 - y'''
    dataset_y = torch.tensor(dataset_y.copy())
    dataset_y = dataset_y.reshape(len(dataset_y),2)
    dataset_y = np.array(dataset_y.permute(1,0))
    list = d.cord_to_onehot(dataset_y[0],dataset_y[1]) 
    list = torch.LongTensor(list)


    return dataset_x,list

def get_grey_dataset():
    #이미지/좌표 준비
    image = d.getResizedImages(d.getImages(c.imgPath))
    image2 = d.getResizedImages(d.getImages(c.imgPath_2))

    x,y = d.getResizedCord(c.labelPath)
    x_2,y_2 = d.getResizedCord(c.labelPath_2)

    image = image+image2
    x = x+x_2
    y = y+y_2
    
    #좌표 keypoints 형식으로 준비(albumentation keypoints 형식)
    keypoints = [[[x,y]] for x,y in zip(x,y)]

    dataset_x = []
    dataset_y = []

    dataset_x = image.copy()
    dataset_y = keypoints.copy()

    '''데이터 증강'''
    tempImgs,tempKeys = horizonalFlip(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = rotated(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = ShiftScaleRotate(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = complex(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = ChannelShuffle(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys

    tempImgs,tempKeys = PixelDropout(image,keypoints)
    dataset_x += tempImgs
    dataset_y += tempKeys



    '''이미지 전처리'''
    dataset_x = ToGray(dataset_x)
    dataset_x,dataset_y = CLAHE(dataset_x,dataset_y)

    '''shape 변환 - x'''
    dataset_x = np.array(dataset_x)
    dataset_x = torch.FloatTensor(dataset_x)
    dataset_x = dataset_x.permute(0,3,1,2)
    

    '''shape 변환 - y'''
    dataset_y = torch.tensor(dataset_y.copy())
    dataset_y = dataset_y.reshape(len(dataset_y),2)
    dataset_y = np.array(dataset_y.permute(1,0))
    list = d.cord_to_onehot(dataset_y[0],dataset_y[1]) 
    list = torch.LongTensor(list)


    return dataset_x,list
