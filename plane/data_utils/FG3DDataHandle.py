# -*- ecoding: utf-8 -*-
# @ModuleName: FG3DDataHandle
# @Author: Kandy
# @Time: 2023-11-21 10:05
import os
import numpy as np
import torch
from data_utils.ModelNetDataLoader import farthest_point_sample,pc_normalize

def get_subfolder_names(folder_path):
    subfolders_list = []

    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subfolders_list.append(dir)
    return subfolders_list
def dataNormalization(dataPath):
    if "\\" in dataPath:
        filename = dataPath.split("\\")[-1]  # 使用split方法将路径按照反斜杠进行分割，并取最后一个元素
    elif "/" in dataPath:
        filename = dataPath.split("/")[-1]
    name = filename.split("_")[0]
    fn = (name, dataPath)
    # list = ["jingangji","nimiziji","tiwanjideji"]
    # list = ['airliner','biplane','deltawing','fighter','helicopter','light','propeller','shuttle']
    list = ["airliner","awcas","cargoShip","deltawing","fighter","helicopter","jingangji","nimiziji","tiwanjideji"]
    classes = {}
    for index,class_ in enumerate(list):
        classes[class_] = index
    # cls = classes[fn[0]]
    # cls = np.array([cls]).astype(np.int32)
    point_set = np.loadtxt(fn[1], delimiter=' ').astype(np.float32)
    # 最远点采样，采1024个点
    point_set = farthest_point_sample(point_set, 1024)
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 相当于3列值，做标准化
    point_set = torch.Tensor(point_set).reshape(1, 1024, 6)
    # cls = torch.Tensor(cls).unsqueeze(0)
    return point_set,classes

if __name__ == '__main__':
    txtPaht = r"C:\Users\Administrator\Desktop\项目相关\点云项目\plane\test_data\deltawing_000006.txt"
    point_set,classes = dataNormalization(txtPaht)
    print(classes)