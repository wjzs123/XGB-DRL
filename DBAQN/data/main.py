import os
import time
import shutil
import numpy as np
import pandas as pd
from evaluation import Tool
from dqn import DQN
from ddpg import DDPG
from prediction import Prediction
from picture import DrawPicture
# from create_samples import CreateSample
# from state_classify import StateClassify
from state_classify import WOA_XGBoost
from DBAQN import WXGB_DBAQN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 数据处理
dir = os.getcwd()
dir_data = dir + "/data/"
filename_data = "shanghai_4.csv"
data = pd.read_csv(dir_data + filename_data)
energy_column = "Carbon_emission"

#
# time_data = data[time_column]
energy_data = data[energy_column]
# # 将数据转换为NumPy数组
data_array = np.array(energy_data)
# data_array = np.array(data)
data_array = data_array.reshape(-1, 1)

clf = IsolationForest(contamination=0.05, random_state=42)
outliers = clf.fit_predict(data_array)

# count = []
# for i in range(1, len(outliers)):
#     if outliers[i] == -1:
#         count.append(1)
# num = len(count)

# 指数平滑法处理异常点
def exponential_smoothing(series, alpha):
    result = [series[0]] # 初始值
    for i in range(1, len(series)):
        if outliers[i] == -1:  # 仅处理异常值点
            # result.append(alpha * series[i] + (1 - alpha) * result[i-1])
            smoothed_value = alpha * series[i] + (1 - alpha) * ((result[i - 1]) + series[i - 1]) / 2
            result.append(np.round(smoothed_value))
        else:
            result.append(series[i])
    return result

alpha = 0.2
# 8760
smoothed_energy_values = exponential_smoothing(data_array, alpha)

# 24, 7
features = 24
data, label = CreateSample().createSample(data=smoothed_energy_values, shape="vector", features=features)

data_train = np.array(data[0:len(data) - 7877])
label_train = np.array(label[0:len(label) - 7877])
data_test = np.array(data[len(data) - 7877:])
label_test = np.array(label[len(label) - 7877:])

# normalization
filename_log = ""
data_train_scale, data_test_scale = Tool(filename_log).normalization(data_train=data_train, data_test=data_test)

# data_train_scale = data_train
# data_test_scale = data_test



# WOA-XGBoost Double BiLSTM-Attention Q-nework

MAX_EPISODES = 100
MAX_STEPS = 1000
CLASS = 3
# 结束
N_CLASS = 26
iteration = 10

data_train_min = np.min(data_train)
data_train_max = np.max(data_train)
date_data = pd.read_csv(dir_data + 'shanghai_4.csv', header=0, skiprows=24,nrows=18379)
date_data = date_data.iloc[:, [2, 3, 4, 5, 6,7,8]]
date_data2 = pd.read_csv(dir_data + 'shanghai_4.csv', header=0, skiprows=18403,nrows=7877)
date_data2 = date_data2.iloc[:, [2, 3, 4, 5, 6,7,8]]


data_train_scale = np.hstack((data_train_scale, date_data))
data_test_scale = np.hstack((data_test_scale, date_data2))


METHOD_STR = "DBAQN"  # DBAQN
# dir_dqn = dir + '\\experiments\\China_shanghai_Carbon_data_dbaqn'
# dir_ddpg = dir + '\\experiments\\China_shanghai_Carbon_data_ddpg'

while True:
    CLASS = CLASS + 1
    dir_dbaqn = dir + '\\experiments\\shanghai_4.1_data' + '\\N_Class=' + str(CLASS) + '\\index='

    for index in np.arange(0, iteration):
        start_first = time.perf_counter()
        # determine the path of methods
        if METHOD_STR == "DBAQN":
            dir_choose = dir_dbaqn + str(index)

        # elif METHOD_STR == "DQN":
        #     dir_choose = dir_dqn + str(index)
        #
        # elif METHOD_STR == "DDPG":
        #     dir_choose = dir_ddpg + str(index)

        if os.path.exists(dir_choose):
            shutil.rmtree(dir_choose)
        os.makedirs(dir_choose)
        filename_log = "log.txt"
        file_log = open(dir_choose + "\\" + filename_log, 'w')

        # prediction
        if METHOD_STR == "DBAQN":

            gap = np.ceil((data_train_max - data_train_min + 1) / CLASS)  # size of sub-action space

            print("\nclass: ", str(CLASS), " iterations: ", str(index))
            print("actions: ", str(gap))

            file_log.write("class: " + str(CLASS) + " iterations: " + str(index) + "\n")
            file_log.write("actions: " + str(gap) + " \n\n")
            # 定义标签类型
            class_train_true = ((label_train - data_train_min) / gap).astype(int)
            class_test_true = ((label_test - data_train_min) / gap).astype(int)

            # data_train_scale
            state_train_scale, state_test_scale, class_train_pre, class_test_pre = WOA_XGBoost().constructStateXGBoost(
                data_train_scale=data_train_scale, data_test_scale=data_test_scale,
                class_train_true=class_train_true, class_test_true=class_test_true, file_log=file_log)
            start_second = time.perf_counter()

            # hyper-parameters
            N_FEATURES = features + CLASS + 7

            ACTION_START = data_train_min
            ACTION_END = data_train_min + gap
            N_ACTIONS = int(gap)
            N_HIDDEN = 64
            LEARNING_RATE = 0.01
            GAMMA = 0.9
            EPSILON = 0.5
            EPSILON_DECAY = 0.995
            EPSILON_MIN = 0.01
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            dbaqn = WXGB_DBAQN(n_features=N_FEATURES, n_class=CLASS, action_start=ACTION_START,
                            action_end=ACTION_END,
                            n_actions=N_ACTIONS, n_hidden=N_HIDDEN, learning_rate=LEARNING_RATE, gamma=GAMMA,
                            epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                            memory_size=MEMORY_SIZE,
                            batch_size=BATCH_SIZE)

            dbaqn_train, mae_train, predict_train, actual_train, reward_train = Prediction(). \
                train(method_str=METHOD_STR, method=dbaqn, state=state_train_scale, action=label_train,
                      max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log, state_kinds=class_train_pre)


            predict_test, actual_test = Prediction().prediction(
                method_str="DBAQN", method=dbaqn_train, state=state_test_scale, action=label_test,
                state_kinds=class_test_pre)

        # save data

        save_data_list = [mae_train, predict_train, actual_train, reward_train, predict_test, actual_test]
        save_data_filename = ["mae_train.csv", "predict_train.csv", "actual_train.csv", "reward_train.csv",
                              "predict_test.csv", "actual_test.csv"]

        for j in range(len(save_data_list)):

            data_temp = pd.DataFrame(save_data_list[j])

            data_temp_filename = "\\" + save_data_filename[j]

            if os.path.exists(dir_choose + data_temp_filename):
                os.remove(dir_choose + data_temp_filename)

            data_temp.to_csv(dir_choose + data_temp_filename, index=False, header=None)



        # Accuracy
        print("training set: ")
        file_log.write("\ntraining set: \n")
        Tool(file_log).mae(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).mape(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).rmse(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).r2(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).cv(action_predict=predict_train, action_true=actual_train)

        print("====================================================================")
        print("test set: ")
        file_log.write("===============================================================\n")
        file_log.write("test set: \n")
        Tool(file_log).mae(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).mape(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).rmse(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).r2(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).cv(action_predict=predict_test, action_true=actual_test)


        # 关闭日志
        file_log.close()


    if METHOD_STR == "DBAQN":
        if CLASS == N_CLASS:
            break
