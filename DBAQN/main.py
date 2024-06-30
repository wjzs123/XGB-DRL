import os
import time
import shutil
import numpy as np
import pandas as pd

from tool import Tool
from dqn_agent import DQN
from ddpg_agent import DDPG
from prediction import Prediction
from draw_picture import DrawPicture
from create_samples import CreateSample
from state_classify import StateClassify
from state_classify import WOA_XGBoost
from df_dqn_agent import DF_DQN

from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt

"""
    1. data_preprocessing
"""
dir = os.getcwd()
dir_data = dir + "/data/"

filename_data = "shanghai_4.csv"



data = pd.read_csv(dir_data + filename_data)
energy_column = "Carbon_emission"  # 用实际的列名替换 "能耗值"
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
# 8736
data, label = CreateSample().createSample(data=smoothed_energy_values, shape="vector", features=features)

# shanghai_4.csv的数据 7:3
#
data_train = np.array(data[0:len(data) - 7877])
label_train = np.array(label[0:len(label) - 7877])
#
data_test = np.array(data[len(data) - 7877:])
label_test = np.array(label[len(label) - 7877:])

# 1.3 normalization
filename_log = ""
data_train_scale, data_test_scale = Tool(filename_log).normalization(data_train=data_train, data_test=data_test)

# data_train_scale = data_train
# data_test_scale = data_test

"""
    2. DRL methods for energy consumption prediction
"""
MAX_EPISODES = 20


MAX_STEPS = 1000

CLASS = 24
# 终止条件
N_CLASS = 40
iteration = 1


data_train_min = np.min(data_train)
data_train_max = np.max(data_train)



# 24,6989
# 7013,1747
date_data = pd.read_csv(dir_data + 'shanghai_4.csv', header=0, skiprows=24,nrows=18379)
date_data = date_data.iloc[:, [2, 3, 4, 5, 6,7,8]]
date_data2 = pd.read_csv(dir_data + 'shanghai_4.csv', header=0, skiprows=18403,nrows=7877)
date_data2 = date_data2.iloc[:, [2, 3, 4, 5, 6,7,8]]


# date_data = pd.read_csv(dir_data + 'shanghai_5.csv', header=0, skiprows=24,nrows=12247)
# date_data = date_data.iloc[:, [2, 3, 4, 5, 6,7]]
# date_data2 = pd.read_csv(dir_data + 'shanghai_5.csv', header=0, skiprows=12271,nrows=5249)
# date_data2 = date_data2.iloc[:, [2, 3, 4, 5, 6,7]]

# date_data = pd.read_csv(dir_data + 'shanghai_3.csv', header=0, skiprows=24,nrows=12826)
# date_data = date_data.iloc[:, [2, 3, 4, 5, 6,7]]
# date_data2 = pd.read_csv(dir_data + 'shanghai_3.csv', header=0, skiprows=12850,nrows=3206)
# date_data2 = date_data2.iloc[:, [2, 3, 4, 5, 6,7]]


# date_data = pd.read_csv(dir_data + 'new_carbon_emission_dataset.csv', header=0, skiprows=24,nrows=6989)
# date_data = date_data.iloc[:, [2, 3, 4, 5, 6,7,8]]
# date_data2 = pd.read_csv(dir_data + 'new_carbon_emission_dataset.csv', header=0, skiprows=7013,nrows=1747)
# date_data2 = date_data2.iloc[:, [2, 3, 4, 5, 6,7,8]]




# 将 date_data 数据集与 data_train 数据集合并。
data_train_scale = np.hstack((data_train_scale, date_data))
data_test_scale = np.hstack((data_test_scale, date_data2))





METHOD_STR = "DBAQN"  # DBAQN

dir_dqn = dir + '\\experiments\\China_shanghai_Carbon_data_dbaqn' + '\\DDQN_index='
dir_ddpg = dir + '\\experiments\\China_shanghai_Carbon_data_ddpg' + '\\DDPG_index='

while True:

    CLASS = CLASS + 1
    #
    dir_dfdqn = dir + '\\experiments\\shanghai_4.1_data' + '\\N_Class=' + str(CLASS) + '\\DF-DQN_index='

    for index in np.arange(0, iteration):

        start_first = time.perf_counter()

        # determine the path of methods
        if METHOD_STR == "DF-DQN":
            #     dir_dfdqn = dir + '\\experiments\\DF-DQN' + '\\N_Class=' + str(CLASS) + '\\DF-DQN_index='
            dir_choose = dir_dfdqn + str(index)

        elif METHOD_STR == "DQN":
            dir_choose = dir_dqn + str(index)

        elif METHOD_STR == "DDPG":
            dir_choose = dir_ddpg + str(index)

        if os.path.exists(dir_choose):
            shutil.rmtree(dir_choose)
        os.makedirs(dir_choose)

        filename_log = "log.txt"
        file_log = open(dir_choose + "\\" + filename_log, 'w')

        # 2.1 DF-DQN
        if METHOD_STR == "DF-DQN":

            gap = np.ceil((data_train_max - data_train_min + 1) / CLASS)  # size of sub-action space

            print("\nclass: ", str(CLASS), " iterations: ", str(index))
            print("the number of actions: ", str(gap))

            file_log.write("class: " + str(CLASS) + " iterations: " + str(index) + "\n")
            file_log.write("the number of actions: " + str(gap) + " \n\n")
            # 定义标签类型
            class_train_true = ((label_train - data_train_min) / gap).astype(int)
            class_test_true = ((label_test - data_train_min) / gap).astype(int)




            # data_train_scale：24个归一化的数据+7个时间戳特征
            state_train_scale, state_test_scale, class_train_pre, class_test_pre = StateClassify().constructStateXGBoost(
                data_train_scale=data_train_scale, data_test_scale=data_test_scale,
                class_train_true=class_train_true, class_test_true=class_test_true, file_log=file_log)


            # # WOA-XGBoost
            # state_train_scale, state_test_scale, class_train_pre, class_test_pre = WOA_XGBoost().constructStateXGBoost(
            #     data_train_scale=data_train_scale, data_test_scale=data_test_scale,
            #     class_train_true=class_train_true, class_test_true=class_test_true, file_log=file_log)




            start_second = time.perf_counter()

            # hyper-parameters
            N_FEATURES = features + CLASS +7

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

            df_dqn = DF_DQN(n_features=N_FEATURES, n_class=CLASS, action_start=ACTION_START,
                            action_end=ACTION_END,
                            n_actions=N_ACTIONS, n_hidden=N_HIDDEN, learning_rate=LEARNING_RATE, gamma=GAMMA,
                            epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
                            memory_size=MEMORY_SIZE,
                            batch_size=BATCH_SIZE)

            dqn_train, mae_train, predict_train, actual_train, reward_train = Prediction(). \
                train(method_str=METHOD_STR, method=df_dqn, state=state_train_scale, action=label_train,
                      max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log, state_kinds=class_train_pre)


            predict_test, actual_test = Prediction().prediction(
                method_str="DF-DQN", method=dqn_train, state=state_test_scale, action=label_test,
                state_kinds=class_test_pre)



        elif METHOD_STR == "DQN":

            print("iterations: ", str(index))
            file_log.write("iterations: " + str(index) + "\n")

            N_FEATURES = features + 6
            N_ACTIONS = int(data_train_max - data_train_min + 1)
            ACTION_LOW = data_train_min
            ACTION_HIGH = data_train_max
            N_HIDDEN = 32
            LEARNING_RATE = 0.01
            GAMMA = 0.9
            EPSILON = 0.5
            EPSILON_DECAY = 0.995
            EPSILON_MIN = 0.01
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            dqn = DQN(n_features=N_FEATURES, n_actions=N_ACTIONS, n_hidden=N_HIDDEN, action_low=ACTION_LOW,
                      action_high=ACTION_HIGH,
                      learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY,
                      epsilon_min=EPSILON_MIN, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)

            dqn_train, mae_train, predict_train, actual_train, reward_train = Prediction().train(
                method_str=METHOD_STR, method=dqn, state=data_train_scale,
                action=label_train, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log)

            predict_test, actual_test = Prediction().prediction(
                method_str="DQN", method=dqn_train, state=data_test_scale, action=label_test)

        elif METHOD_STR == "DDPG":

            print("iterations: ", str(index))
            file_log.write("iterations: " + str(index) + "\n")

            N_FEATURES = features + 6
            ACTION_LOW = data_train_min
            ACTION_HIGH = data_train_max
            CLIP_MIN = data_train_min
            CLIP_MAX = data_train_max
            N_HIDDEN = 32
            LEARNING_RATE_ACTOR = 0.001
            LEARNING_RATE_CRITIC = 0.001
            GAMMA = 0.9
            TAU = 0.1
            VAR = 40
            MEMORY_SIZE = 2000
            BATCH_SIZE = 64

            ddpg = DDPG(n_features=N_FEATURES, action_low=ACTION_LOW, action_high=ACTION_HIGH, n_hidden=N_HIDDEN,
                        learning_rate_actor=LEARNING_RATE_ACTOR, learning_rate_critic=LEARNING_RATE_CRITIC,
                        gamma=GAMMA, tau=TAU, var=VAR, clip_min=CLIP_MIN, clip_max=CLIP_MAX, memory_size=MEMORY_SIZE,
                        batch_size=BATCH_SIZE)

            ddpg_train, mae_train, predict_train, actual_train, reward_train = Prediction().train(
                method_str=METHOD_STR, method=ddpg, state=data_train_scale, action=label_train,
                max_episodes=MAX_EPISODES, max_steps=MAX_STEPS, file_log=file_log)

            predict_test, actual_test = Prediction().prediction(
                method_str=METHOD_STR, method=ddpg_train, state=data_test_scale, action=label_test)


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



        # prediction accuracy
        print("training set: ")
        file_log.write("\ntraining set: \n")
        Tool(file_log).mae(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).mape(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).rmse(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).r2(action_predict=predict_train, action_true=actual_train)
        Tool(file_log).cv(action_predict=predict_train, action_true=actual_train)

        print("====================================================================")
        print("test set: ")
        file_log.write("====================================================================\n")
        file_log.write("test set: \n")
        Tool(file_log).mae(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).mape(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).rmse(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).r2(action_predict=predict_test, action_true=actual_test)
        Tool(file_log).cv(action_predict=predict_test, action_true=actual_test)




        # plot
        DrawPicture().Xrange_Y(dir=dir_choose, figName="mae_train", Yname="mae", Y=mae_train)

        DrawPicture().Xrange_Ypredicted_Yactual(dir=dir_choose, figName="predict and actual in test set",
                                                action_predict=predict_test, action_true=actual_test)
        DrawPicture().Xpredicted_Yactual(dir=dir_choose, figName="trend", action_predict=predict_test, action_true=actual_test)

        # 关闭日志
        file_log.close()




    if METHOD_STR == "DQN" or METHOD_STR == "DDPG":
        break
    elif METHOD_STR == "DF-DQN":
        if CLASS == N_CLASS:
            break
