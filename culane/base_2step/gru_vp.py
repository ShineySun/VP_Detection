import torch
import torch.nn as nn
from torch.autograd import Variable
from parameters import Parameters


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
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

        # self.linear_2 = nn.Linear(480, 360, bias=True)

        self.linear_3 = nn.Linear(360, 180, bias=True)

        self.linear_4 = nn.Linear(180, 90, bias=True)

        self.linear_5 = nn.Linear(90, 45, bias=True)

        self.linear_6 = nn.Linear(45, 2, bias=True)

        # self.linear_1 = nn.Linear(720, 360, bias=True)
        #
        # self.linear_2 = nn.Linear(360, 180, bias=True)
        #
        # self.linear_3 = nn.Linear(180, 90, bias=True)
        #
        # self.linear_4 = nn.Linear(90, 45, bias=True)
        #
        # self.linear_5 = nn.Linear(45, 22, bias=True)
        #
        # self.linear_6 = nn.Linear(22, 2, bias=True)
        self.relu = nn.ReLU()

    def forward(self, in_1, in_2):
        h_1_0 = Variable(torch.zeros(self.num_layers, in_1.shape[0], self.hidden_size).float()).cuda()
        h_2_0 = Variable(torch.zeros(self.num_layers, in_2.shape[0], self.hidden_size).float()).cuda()

        out_1, hidden_1 = self.gru_1(in_1, h_1_0)
        #print("out shape : ", out_1.shape)
        # print("hidden shape : ", hidden_1.shape)
        out_2, hidden_2 = self.gru_2(in_2, h_2_0)

        # out_1 = out_1[:, -30:, :].view(1, -1)
        # out_2 = out_2[:, -30:, :].view(1, -1)

        out_1 = out_1[:, -15:, :].reshape(out_1.shape[0], -1)
        out_2 = out_2[:, -15:, :].reshape(out_2.shape[0], -1)
        out = out_1 + out_2

        # out = self.linear_1(out)
        # out = self.batch_norm_1(out)
        # out = self.relu(out)

        # out = self.linear_2(out)
        # #out = self.batch_norm_2(out)
        # out = self.relu(out)

        out = self.linear_3(out)
        #out = self.batch_norm_3(out)
        out = self.relu(out)

        out = self.linear_4(out)
        out = self.relu(out)

        out = self.linear_5(out)
        out = self.relu(out)

        out = self.linear_6(out)
        # print('out.shape:',out.shape)

        return out
