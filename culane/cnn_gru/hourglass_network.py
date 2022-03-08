#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from util_hourglass import *
from bridge import bridge
from gru_cnn_vp import GRU_CNN

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()

        self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)
        self.layer3 = hourglass_block(128, 128)
        self.layer4 = hourglass_block(128, 128)
        self.gru_cnn = GRU_CNN(input_size=2, hidden_size=24, num_layers=2, output_size=2).cuda()


    def forward(self, inputs, vp_gt):
        #feature extraction
        out = self.resizing(inputs)
        result1, out, feature1 = self.layer1(out)
        result2, out, feature2 = self.layer2(out)   
        result3, out, feature3 = self.layer3(out)
        result4, out, feature4 = self.layer4(out)
        
        # vp detect
        left, right, vp_gt_used, vp_batch_idx, img = bridge(result4, vp_gt)

        if left is not None:
            pred_vp = self.gru_cnn(left.cuda(), right.cuda(), img.cuda())
        else:
            pred_vp = None



        return [result1, result2, result3, result4], [feature1, feature2, feature3, feature4], [pred_vp, vp_gt_used, vp_batch_idx]
        #return [result2], [feature2]
