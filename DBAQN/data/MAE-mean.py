# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 将每一次迭代的mae对应相加并取平均，获得对应分类的mae-mean
import pandas as pd

# 读取十个数据集并存储在列表中
data_sets = []
for i in range(10):
    data = pd.read_csv(f"D:\\桌面\\WXGB-DBAQN\\experiments\\mae_train.csv")  # 替换成你的数据集路径
    data_sets.append(data)
# 初始化一个列表来存储累积的和
accumulated_sums = [0] * len(data_sets[0])  # 假设数据集的列数一致

# 逐行相加并累积
for data in data_sets:
    for idx, row in enumerate(data.values):
        accumulated_sums[idx] += row

# 计算平均值
num_datasets = len(data_sets)
average_values = [sums / num_datasets for sums in accumulated_sums]

# 将结果保存到文件
result_df = pd.DataFrame(average_values, columns=data_sets[0].columns)
result_df.to_csv("path_to_save_result.csv", index=False)  # 替换成你希望保存的路径
