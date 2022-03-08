from turtle import right
import time
from sklearn.utils import check_array
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import torch.nn as nn
import torch
from util_hourglass import *
from copy import deepcopy
import numpy as np

from gru_vp import GRU
from torch.autograd import Function as F
from parameters import Parameters
import util
import matplotlib.pyplot as plt

def bridge(result, vp_gt):
    confidance, offset, feature = result
    # lane_keypoints = []
    ego_left_pts = []
    ego_right_pts = []
    vp_gt_used = []
    vp_batch_idx = []

    for i in range(p.batch_size):
        try: confidance[i]
        except: break

        # lane key point 생성
        confidence_ = confidance[i].view(p.grid_y, p.grid_x).cpu().data.numpy()
        # confidence.shape <- [32, 64]
        # confidence map을 시각화한것이 히트맵

        offset_ = offset[i].cpu().data.numpy()
        offset_ = np.rollaxis(offset_, axis=2, start=0)
        offset_ = np.rollaxis(offset_, axis=2, start=0)
        # offset.shape < - (32, 64, 2)

        instance_ = feature[i].cpu().data.numpy()
        instance_ = np.rollaxis(instance_, axis=2, start=0)
        instance_ = np.rollaxis(instance_, axis=2, start=0)
        # instance.shape <- (32, 64, 4)

        x, y = generate_result(
            confidence_, offset_, instance_, p.threshold_point)
        x, y = eliminate_fewer_points(x, y)
        left_pt, right_pt = util.ego_lane(x, y)
        # print("left----------------")
        left_pt = spline_lane(left_pt)
        # print("right----------------")
        right_pt = spline_lane(right_pt)
        if left_pt is not None and right_pt is not None:
            vp_gt_used.append([vp_gt[i][0]/p.x_size, vp_gt[i][1]/p.y_size])
            ego_left_pts.append(left_pt)
            ego_right_pts.append(right_pt)
            vp_batch_idx.append(i)

    # if ego_left_pts != []:
    #         vp_gt_used = torch.from_numpy(np.array(vp_gt_used, dtype='float32')).cuda()
    #         pred_vp = gru(torch.from_numpy(np.array(ego_left_pts, dtype='float32')).cuda(), torch.from_numpy(np.array(ego_right_pts, dtype='float32')).cuda())
    if ego_left_pts != []:
        return torch.from_numpy(np.array(ego_left_pts, dtype='float32')), torch.from_numpy(np.array(ego_right_pts, dtype='float32')), torch.from_numpy(np.array(vp_gt_used, dtype='float32')).cuda(), vp_batch_idx
    else:
        return None, None, None, None



def spline_lane(pt):
    if pt != None:
        ys = np.array(pt).T[0]
        xs = np.array(pt).T[1] 
        # xs = -xs ## vp에 가까운 부분먼저 spline에 넣기위함. spline이후 원래대로 돌려줌

        s = xs.argsort()
        xs = xs[s]
        ys = ys[s]
        xs, unique_idx = np.unique(xs, return_index=True)
        ys = ys[unique_idx]
        # interpolation scheme : Cubic Spline
        try:
            cs_intrp = interp1d(xs, ys)
            # cs_intrp2 = interp1d(xs, ys, kind='quadratic')
            # cs_intrp = CubicSpline(xs, ys)
        except: 
            print('cubic spline error') 
            print('xs:', xs)
            print('ys:', ys)
            return None
        # x_intrp = np.linspace(int(xs.min()), int(xs.max()), int(xs.max())-int(xs.min())+1)
        x_intrp = np.linspace(int(xs.min()), int(xs.max()), 40)
        y_intrp = cs_intrp(x_intrp)


        x_intrp /= p.y_size
        y_intrp /= p.x_size
        intrp_lane = np.array(list(zip(y_intrp, x_intrp)), dtype='float32')
        return intrp_lane
    else: 
        # print("there is no lane")
        return None

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i) > 2:
            out_x.append(i)
            out_y.append(j)
    return out_x, out_y

def generate_result(confidance, offsets,instance, thresh):
    
    mask = confidance > thresh
    # print(confidance.shape)
    # print(offsets.shape)
    # print(instance.shape)
    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y
