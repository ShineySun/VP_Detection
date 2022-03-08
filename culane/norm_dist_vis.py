import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('./end_to_end/norm_dist/norm_dist.pkl', 'rb') as f:
    norm_dist_end_to_end = pickle.load(f)

with open('./cnn_gru/norm_dist/norm_dist.pkl', 'rb') as f:
    norm_dist_cnn_gru = pickle.load(f)

with open('./cnn_gru_x_axis/norm_dist/norm_dist.pkl', 'rb') as f:
    norm_dist_cnn_gru_x_axis = pickle.load(f)
# print(norm_dist)


others_end_to_end = 0
others_cnn_gru = 0
others_cnn_gru_x_axis = 0

lst_end_to_end = [[0, 0], [0.01, 0], [0.02, 0], [0.03, 0], [0.04, 0],
       [0.05, 0], [0.06, 0], [0.07, 0], [0.08, 0], [0.09, 0]]

lst_cnn_gru = [[0, 0], [0.01, 0], [0.02, 0], [0.03, 0], [0.04, 0],
       [0.05, 0], [0.06, 0], [0.07, 0], [0.08, 0], [0.09, 0]]

lst_cnn_gru_x_axis = [[0, 0], [0.01, 0], [0.02, 0], [0.03, 0], [0.04, 0],
                  [0.05, 0], [0.06, 0], [0.07, 0], [0.08, 0], [0.09, 0]]


for i in norm_dist_end_to_end:
    if i >=0 and i <0.01: lst_end_to_end[0][1] += 1
    elif i < 0.02: lst_end_to_end[1][1] += 1
    elif i < 0.03: lst_end_to_end[2][1] += 1
    elif i < 0.04: lst_end_to_end[3][1] += 1
    elif i < 0.05: lst_end_to_end[4][1] += 1
    elif i < 0.06: lst_end_to_end[5][1] += 1
    elif i < 0.07: lst_end_to_end[6][1] += 1
    elif i < 0.08: lst_end_to_end[7][1] += 1
    elif i < 0.09: lst_end_to_end[8][1] += 1
    elif i < 0.1: lst_end_to_end[9][1] += 1
    else:
        others_end_to_end += 1

for i in norm_dist_cnn_gru:
    if i >=0 and i <0.01: lst_cnn_gru[0][1] += 1
    elif i < 0.02: lst_cnn_gru[1][1] += 1
    elif i < 0.03: lst_cnn_gru[2][1] += 1
    elif i < 0.04: lst_cnn_gru[3][1] += 1
    elif i < 0.05: lst_cnn_gru[4][1] += 1
    elif i < 0.06: lst_cnn_gru[5][1] += 1
    elif i < 0.07: lst_cnn_gru[6][1] += 1
    elif i < 0.08: lst_cnn_gru[7][1] += 1
    elif i < 0.09: lst_cnn_gru[8][1] += 1
    elif i < 0.1: lst_cnn_gru[9][1] += 1
    else:
        others_cnn_gru += 1

for i in norm_dist_cnn_gru_x_axis:
    if i >=0 and i <0.01: lst_cnn_gru_x_axis[0][1] += 1
    elif i < 0.02: lst_cnn_gru_x_axis[1][1] += 1
    elif i < 0.03: lst_cnn_gru_x_axis[2][1] += 1
    elif i < 0.04: lst_cnn_gru_x_axis[3][1] += 1
    elif i < 0.05: lst_cnn_gru_x_axis[4][1] += 1
    elif i < 0.06: lst_cnn_gru_x_axis[5][1] += 1
    elif i < 0.07: lst_cnn_gru_x_axis[6][1] += 1
    elif i < 0.08: lst_cnn_gru_x_axis[7][1] += 1
    elif i < 0.09: lst_cnn_gru_x_axis[8][1] += 1
    elif i < 0.1: lst_cnn_gru_x_axis[9][1] += 1
    else:
        others_cnn_gru_x_axis += 1

for i in range(len(lst_end_to_end)):
    lst_end_to_end[i][1] /= len(norm_dist_end_to_end)
    if i>0:
        lst_end_to_end[i][1] += lst_end_to_end[i-1][1]

for i in range(len(lst_cnn_gru)):
    lst_cnn_gru[i][1] /= len(norm_dist_cnn_gru)
    if i>0:
        lst_cnn_gru[i][1] += lst_cnn_gru[i-1][1]

for i in range(len(lst_cnn_gru_x_axis)):
    lst_cnn_gru_x_axis[i][1] /= len(norm_dist_cnn_gru_x_axis)
    if i > 0:
        lst_cnn_gru_x_axis[i][1] += lst_cnn_gru_x_axis[i-1][1]

lst_end_to_end = np.array(lst_end_to_end).T.tolist()
lst_cnn_gru = np.array(lst_cnn_gru).T.tolist()
lst_cnn_gru_x_axis = np.array(lst_cnn_gru_x_axis).T.tolist()

plt.scatter(lst_end_to_end[0], lst_end_to_end[1])
plt.plot(lst_end_to_end[0], lst_end_to_end[1], label = 'end_to_end')

plt.scatter(lst_cnn_gru[0], lst_cnn_gru[1])
plt.plot(lst_cnn_gru[0], lst_cnn_gru[1], label='cnn_gru')

plt.scatter(lst_cnn_gru_x_axis[0], lst_cnn_gru_x_axis[1])
plt.plot(lst_cnn_gru_x_axis[0], lst_cnn_gru_x_axis[1], label='cnn_gru_x_axis')

plt.title("NormDist")
plt.xlabel("normdist error")
plt.ylabel("percentage of the whole dataset(%)")
plt.xticks(np.arange(0, 0.1, 0.01))
plt.legend()
plt.show()
