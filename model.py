import torch.nn as nn
import constants as c

#weight 초기화 함수
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
    
relu = nn.ReLU()
MaxPool2d = nn.MaxPool2d(2)
flatten = nn.Flatten()
relu = nn.ReLU()
MaxPool2d = nn.MaxPool2d(2)
flatten = nn.Flatten()

#초기 모델 -> 사용x
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,20,kernel_size=21,stride=1)
        self.conv2 = nn.Conv2d(20,40,kernel_size=13,stride=1)
        self.conv3 = nn.Conv2d(40,60,kernel_size=13,stride=1)
        self.conv4 = nn.Conv2d(60,80,kernel_size=13,stride=1)

        self.fc1 = nn.Linear(1280,640)
        self.fc2 = nn.Linear(640,320)
        self.fc3 = nn.Linear(320,c.GRID_WIDTH*c.GRID_HEIGHT) ## 출력을 그리드 세로*가로 칸 수 만큼 맞춰야 각 칸에 속할 확률을 구할 수 있음

    def forward(self,input):
        out = self.conv1(input)
        out = relu(out)
        out = MaxPool2d(out)

        out = self.conv2(out)
        out = relu(out)
        out = MaxPool2d(out)

        out = self.conv3(out)
        out = relu(out)
        out = MaxPool2d(out)

        out = self.conv4(out)
        out = relu(out)
        out = MaxPool2d(out)

        out = flatten(out)
        
        out = self.fc1(out)
        out = relu(out)
        out = self.fc2(out)
        out = relu(out)
        out = self.fc3(out)
        
        return out
