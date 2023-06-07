import dataset
import model
import torch
import utils
import calc 
import torch.optim as optim
import torch.nn as nn
import constants as c
import evaluation as e
from sklearn.model_selection import KFold
import torchvision.models as models
import torch.nn as nn

print ('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'커스텀 데이터셋 사용'
airplanes = dataset.Airplanes()


learning_rate = 0.005
training_epochs = 200
k_folds = 10

train_batch_size = 16
test_batch_size = 8

foldResults={}

Kfold=KFold(n_splits=k_folds,shuffle=True,random_state=42)

'사전학습x resnet18사용'
mymodel = models.resnet18(weights = False).to(device) 

'''
모델의 마지막 dense층에 이어
그리드 칸 갯수(64개) 만큼의 노드를 가지는  dense층을 한 층 추가해 줌. 
-> 64개 칸 중 하나로 분류 하도록.
'''
num_classes = c.GRID_WIDTH*c.GRID_HEIGHT # 8*8 = 64
num_ftrs = mymodel.fc.in_features
mymodel.fc = nn.Linear(num_ftrs, num_classes)


mymodel.to(device)

'손실 함수 - 크로스 엔트로피'
loss_fn = nn.CrossEntropyLoss().to(device) 

'옵티마이저 - SGD'
optimizer = optim.SGD(mymodel.parameters(),lr = learning_rate) 

fold_y_index = []
fold_pred_index = []
count = 0
pastcost = 0

#res파일 내 모델 이름과 같은 폴더 내에 결과 저장
modelname = 'resnet18_0606'
saveAt = f'.././res/{c.GRID_WIDTH}by{c.GRID_HEIGHT}Grid/{c.WIDTH}by{c.HEIGHT}/{modelname}'
utils.makeDirectory(saveAt)


for fold, (train_index, test_index) in enumerate(Kfold.split(airplanes)):

    utils.makeDirectory(f"{saveAt}/fold{fold}")

    #fold별 loss정보를 저장하기 위해 log남김

    f=open(f'{saveAt}/fold{fold}/fold{fold}_log.txt','w')

    print('--------------------------------')
    print(f'FOLD : {fold}')
    print('--------------------------------')

    f.writelines('--------------------------------')
    f.writelines(f'FOLD : {fold}')
    f.writelines('--------------------------------')

    y_index = []
    pred_index = []

    train_loss = []
    val_loss = []


    train_dataloader = torch.utils.data.DataLoader(airplanes, batch_size=train_batch_size, sampler=train_index)
    test_dataloader = torch.utils.data.DataLoader(airplanes, batch_size=test_batch_size, sampler=test_index)
    

    #각fold에서 학습 시작 전 weights를 초기화 함
    mymodel.apply(model.reset_weights)

    'epoch 수 만큼 train함'
    for epoch in range(training_epochs):
        avg_cost = 0

        for x,y in train_dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.

            x = x.to(device)
            y = y.to(device)   
            prediction = mymodel(x)

            # loss 계산
            loss = loss_fn(prediction, y)            
            
            #파라미터 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #에포크 당 평균 손실 계산
            avg_cost += loss / len(train_dataloader)

        print(f'[Epoch: {epoch+1}] train loss = {avg_cost}  ')
        f.writelines(f'[Epoch: {epoch+1}] train loss = {avg_cost}  ')

        #best_loss보다 loss가 작다면 모델 저장.
        with torch.no_grad(): #기울기 계산x 옵션
            if epoch == 0:
                best_loss = avg_cost
                torch.save(mymodel, f"{saveAt}/fold{fold}_bestmodel.pt")
                print(f'----------first model saved successfully--------------------')
                f.writelines(f'----------first model saved successfully--------------------')
            else :
                if avg_cost <best_loss: 
                    best_loss = avg_cost
                    torch.save(mymodel, f"{saveAt}/fold{fold}_bestmodel.pt")
                    print(f'----------saved at epoch: {epoch+1}--------------------')
                    f.writelines(f'----------saved at epoch: {epoch+1}--------------------')
    f.close()

    '학습 종료 후 trainset/testset에 대한 예측 및 결과, 정확도 분석'

    with torch.no_grad(): 

        '''trainset에 대해'''
        train_y = []
        train_pred = []
        train_accuracy = 0
        batch_cnt = 0

        for x,y in train_dataloader:
            x = x.to(device)
            y = y.to(device)   
            prediction = mymodel(x)
            loss = loss_fn(prediction,y)

            #예측 -> 64개 중 속할 확률이 가장 높은 class(그리드 칸의 인덱스)로 추출
            prediction = torch.argmax(prediction,dim=1)

            #사전 정의한 거리 기반 정확도 계산 방법으로 계산
            train_accuracy += calc.getAccuracy(y,prediction)/len(train_dataloader)

            #배치 마다 트레인 셋에 대한 예측결과/정답 이미지 파일로 출력
            utils.genOutput(x,y,prediction,f"{saveAt}/fold{fold}/trainResults",batch_cnt) 
            
            batch_cnt += 1

            #전체 trainset에 대한 y값,예측값을 저장
            train_y += y
            train_pred += prediction
            
        #각 fold별 trainset에 대한 y값,예측값을 FoldData에 저장하여 오차 데이터 분석에 활용    
        FoldData_train = dataset.FoldData(fold,train_y,train_pred)

        #train data에 대한 정확도 출력
        print('\n')
        print(f'Trainset - Accuracy: {train_accuracy.item()}')


        '''testset에 대해'''
        test_y = []
        test_pred = []     
        test_accuracy = 0
        batch_cnt = 0
        
        for x,y in test_dataloader:
            x = x.to(device)
            y = y.to(device)   
            prediction = mymodel(x)
            loss = loss_fn(prediction,y)

            #예측 -> 64개 중 속할 확률이 가장 높은 class(그리드 칸의 인덱스)로 추출
            prediction = torch.argmax(prediction,dim=1)

            #사전 정의한 거리 기반 정확도 계산 방법으로 계산
            test_accuracy += calc.getAccuracy(y,prediction)/len(test_dataloader)
            
            #배치 마다 테스트 셋에 대한 예측결과/정답 이미지 파일로 출력
            utils.genOutput(x,y,prediction,f"{saveAt}/fold{fold}/testResults",batch_cnt) 

            batch_cnt += 1

            #전체 testset에 대한 y값,예측값을 저장
            test_y += y
            test_pred += prediction

        #각 fold별 testset에 대한 y값,예측값을 FoldData에 저장하여 오차 데이터 분석에 활용
        FoldData_test = dataset.FoldData(fold,test_y,test_pred)

        #test data에 대한 정확도 출력
        print('\n')
        print(f'Testset - Accuracy: {test_accuracy.item()}')

        #train/tests FoldData를 가지고 오차 데이터 분석.
        e.main(FoldData_train,FoldData_test,modelname,fold,train_accuracy.item(),test_accuracy.item())
