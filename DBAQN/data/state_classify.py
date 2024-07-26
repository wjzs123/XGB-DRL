import numpy as np

from deepforest import CascadeForestClassifier



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense





from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class WOA_XGBoost:
    def __init__(self):
        pass
    def constructStateXGBoost(self, data_train_scale, data_test_scale, class_train_true, class_test_true, file_log):
        # 初始化 WOA 参数
        max_iter = 20  # 最大迭代次数
        population_size = 10
        dim = len(data_train_scale[0])

        # 机参数值初始化种群
        population = []
        for _ in range(population_size):
            params = {'learning_rate': max(np.random.uniform(0.001, 0.5), 0),  # Ensure learning rate is not less than 0
                      'max_depth': np.random.randint(3, 15),  # Random integer value for max_depth
                      'min_child_weight': np.random.randint(1, 20)}  # Random integer value for min_child_weight
            population.append(params)
        best_accuracy = 0.0
        best_params = None


        # WOA优化
        for iter in range(max_iter):
            for whale in population:
                # 训练XGBoost
                model_xgb = XGBClassifier(**whale)
                model_xgb.fit(data_train_scale, class_train_true)

                # 计算准确率
                class_test_pre = model_xgb.predict(data_test_scale)
                acc_test = accuracy_score(class_test_true, class_test_pre) * 100

                if acc_test > best_accuracy:
                    best_accuracy = acc_test
                    best_params = whale
                # 更新参数
                for param in whale:
                    if param == 'max_depth' or param == 'min_child_weight':
                        whale[param] = int(np.random.randint(3, 15))  # Random integer value
                    elif param == 'learning_rate':
                        whale[param] = max(whale[param] + np.random.uniform(-0.1, 0.1), 0)  # Ensure learning rate is not less than 0
                    else:
                        whale[param] += np.random.uniform(-0.1, 0.1)  # Example: Random perturbation

                print("Iteration:", iter+1, "Accuracy:", acc_test)

        # 利用最佳参数评估
        print("Best parameters:", best_params)
        # best_params = population[np.argmax(acc_test)]
        model_xgb = XGBClassifier(**best_params)
        model_xgb.fit(data_train_scale, class_train_true)

        # 概率估计
        class_train_pre = model_xgb.predict(data_train_scale)
        class_test_pre = model_xgb.predict(data_test_scale)
        class_train_proba = model_xgb.predict_proba(data_train_scale)
        class_test_proba = model_xgb.predict_proba(data_test_scale)

        # 计算测试集准确率
        acc_test = accuracy_score(class_test_true, class_test_pre) * 100

        print("Final accuracy:", acc_test)
        file_log.write("Final accuracy: {:.3f} %\n\n".format(acc_test))

        # 构建状态空间
        state_train_scale = np.hstack((data_train_scale, class_train_proba))
        state_test_scale = np.hstack((data_test_scale, class_test_proba))

        return state_train_scale, state_test_scale, class_train_pre, class_test_pre











