import math
import numpy as np

class Tool:

    def __init__(self, file_log):
        self.file_log = file_log

    def normalization(self, data_train, data_test):
        mean_train = np.mean(data_train, axis=0)
        std_train = np.std(data_train, axis=0)
        data_train_scale = (data_train - mean_train) / std_train
        data_test_scale = (data_test - mean_train) / std_train
        return data_train_scale, data_test_scale

    def mae(self, action_predict, action_true):
        N = len(action_true)
        sum = 0
        for i in range(N):
            sum = sum + abs(action_true[i] - action_predict[i])
        MAE = sum / N
        print("MAE: ", str(MAE))
        self.file_log.write("MAE: " + str(MAE) + '\n')

    def mape(self, action_predict, action_true):
        N = len(action_true)
        sum = 0
        for i in range(N):
            sum = sum + abs((action_true[i] - action_predict[i]) / action_true[i])
        MAPE = sum / N

        print("MAPE: ", str(MAPE))
        self.file_log.write("MAPE: " + str(MAPE) + '\n')


    def rmse(self, action_predict, action_true):
        sum = 0
        N = len(action_true)  # 测试集样本数
        for i in range(N):
            sum = sum + math.pow(action_true[i] - action_predict[i], 2)
        RMSE = math.sqrt(sum / N)

        print("RMSE: ", str(RMSE))
        self.file_log.write("RMSE: " + str(RMSE) + '\n')


    def r2(self, action_predict, action_true):
        sum_squares = 0
        mean_true = sum(action_true) / len(action_true)  # 实际值的均值
        for i in range(len(action_true)):
            sum_squares += (action_predict[i] - action_true[i]) ** 2
        residual_sum_squares = sum_squares
        total_sum_squares = sum((val - mean_true) ** 2 for val in action_predict)
        r_squared = 1 - (residual_sum_squares / total_sum_squares)
        print("R2:", r_squared)
        self.file_log.write("R2: " + str(r_squared) + '\n')

    def cv(self, action_predict, action_true):
        sum = 0
        N = len(action_true)
        for i in range(N):
            sum = sum + math.pow(action_true[i] - action_predict[i], 2)
        RMSE = math.sqrt(sum / N)
        sum = 0
        for i in range(N):
            sum = sum + action_true[i]
        actual_mean = sum / N
        CV = RMSE / actual_mean
        print("CV: ", str(CV))
        self.file_log.write("CV: " + str(CV) + '\n')