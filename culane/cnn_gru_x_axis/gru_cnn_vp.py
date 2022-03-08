import torch
import torch.nn as nn
from torch.autograd import Variable
from parameters import Parameters


class GRU_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU_CNN, self).__init__()
        self.p = Parameters()
        # self.num_class = num_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru_1 = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.gru_2 = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.conv_1 = nn.Conv2d(1, 16, 3,(2,4)) # > h,w = [127,128]
        self.conv_2 = nn.Conv2d(16, 32, 3, 2) # > 63
        self.conv_3 = nn.Conv2d(32, 64, 3, 2) # >31
        self.conv_4 = nn.Conv2d(64, 128, 3, 2) # >15
        self.conv_5 = nn.Conv2d(128, 256, 3, 2) # >7
        self.conv_6 = nn.Conv2d(256, 360, 3, 2) # >3
        self.conv_7 = nn.Conv2d(360, 360, 3, 2) # >1

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(360)

        self.linear_1 = nn.Linear(360, 360, bias=True)
        self.linear_2 = nn.Linear(720, 360, bias=True)
        self.linear_3 = nn.Linear(360, 180, bias=True)
        self.linear_4 = nn.Linear(180, 90, bias=True)
        self.linear_5 = nn.Linear(90, 45, bias=True)
        self.linear_6 = nn.Linear(45, 2, bias=True)

        self.relu = nn.ReLU()

    def forward(self, in_1, in_2, in_3, in_4, img):

        ## GRU ##
        h_1_0 = Variable(torch.zeros(self.num_layers, in_1.shape[0], self.hidden_size).float()).cuda()
        h_2_0 = Variable(torch.zeros(self.num_layers, in_2.shape[0], self.hidden_size).float()).cuda()
        h_3_0 = Variable(torch.zeros(self.num_layers, in_3.shape[0], self.hidden_size).float()).cuda()
        h_4_0 = Variable(torch.zeros(self.num_layers, in_4.shape[0], self.hidden_size).float()).cuda()

        out_1, hidden_1 = self.gru_1(in_1, h_1_0) # y_axis left
        out_2, hidden_2 = self.gru_2(in_2, h_2_0) # y_axis right
        out_3, hidden_3 = self.gru_1(in_3, h_3_0) # x_axis left
        out_4, hidden_4 = self.gru_2(in_4, h_4_0) # x_axis right
        # 현재는 4개 그냥 다 더함....
        # concat 등 다른 방식 생각 필요
        # 1,2와 3,4는 연관이 깊으므로 add가 적절할지도?
        out_1 = out_1[:, -15:, :].reshape(out_1.shape[0], -1)
        out_2 = out_2[:, -15:, :].reshape(out_2.shape[0], -1)
        out_3 = out_3[:, -15:, :].reshape(out_3.shape[0], -1)
        out_4 = out_4[:, -15:, :].reshape(out_4.shape[0], -1)
        # out = out_1 + out_2 + out_3 + out_4

        ## LEFT끼리, RIGHT끼리 각각 LINEAR 한번씩 거친후 더함
        # 복잡해 지는것에 비해 큰 이득이 있을까?? 의문...
        # 한번에 4개 다 더하는 것에 비해 left끼리, right끼리 먼저 묶어줘서 덜 헷갈릴 수 있다고 생각
        out_left = out_1 + out_3
        out_right = out_2 + out_4
        out_left = self.linear_1(out_left)
        out_right = self.linear_1(out_right)
        out = out_left + out_right

        ## CNN ##
        img_out = self.relu(self.bn1(self.conv_1(img)))
        img_out = self.relu(self.bn2(self.conv_2(img_out)))
        img_out = self.relu(self.bn3(self.conv_3(img_out)))
        img_out = self.relu(self.bn4(self.conv_4(img_out)))
        img_out = self.relu(self.bn5(self.conv_5(img_out)))
        img_out = self.relu(self.bn6(self.conv_6(img_out)))
        img_out = self.conv_7(img_out)
        img_out = img_out.squeeze(2)
        img_out = img_out.squeeze(2) # batch까지 squeeze하는 문제 해결위함

        # out += img_out
        out = torch.cat((out, img_out), dim=1) # cnn과 gru사이는 +가 아니라 concat



        ## FC Layer ##
        # out = self.linear_1(out)
        # out = self.batch_norm_1(out)
        # out = self.relu(out)

        out = self.linear_2(out)
        out = self.relu(out)

        out = self.linear_3(out)
        out = self.relu(out)

        out = self.linear_4(out)
        out = self.relu(out)

        out = self.linear_5(out)
        out = self.relu(out)

        out = self.linear_6(out)
        # print('out.shape:',out.shape)

        return out
