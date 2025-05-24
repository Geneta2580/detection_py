#!/usr/bin/env python3
#encoding=utf-8

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
import os
import sys
import numpy as np
import pandas as pd
from collections import deque
sys.path.append(f'/home/geneta/anaconda3/envs/pytorch/lib/python3.8/site-packages/') 
import torch
import onnxruntime as ort

model_path = "/home/geneta/project/model/model.onnx"
session = ort.InferenceSession(model_path)
# data = np.genfromtxt(f'/home/geneta/dataset/1408.csv', delimiter=',', skip_header=0) # 读入为double(float64)
# data_3d = data.reshape((1, 50, 1)).astype(np.float32) # 调整为3D输入，强制转为float32
# print(data_3d.size)

# 初始化滑动窗口，最大长度为50
data_window = deque(maxlen=50)

class DataProcessor:
    def __init__(self):
        self.last_timestamp = None  # 初始化时间戳数据
        rospy.Subscriber("/gnss_err_topic", Float32MultiArray, self.callback)
        global pub 
        pub = rospy.Publisher('detection', Float32MultiArray, queue_size=10)

    def callback(self, msg):
        # 提取时间戳
        current_timestamp = msg.data[1]

        # 检查时间戳是否相同,相同直接等待下一个数据
        if self.last_timestamp is not None and current_timestamp == self.last_timestamp:
            rospy.loginfo("时间戳相同，丢弃数据")
            return

        # 更新并处理新数据
        self.last_timestamp = current_timestamp
        rospy.loginfo(f"接收新数据，时间戳: {current_timestamp}")
        
        # 在这里处理有效数据（例如调用模型推理）
        self.process_data(msg.data[0], msg.data[1])

    def process_data(self, new_data, new_data_stamps):
        # new_data = float(new_data)  # 转换为数值
        new_data = (new_data - 0.018) / (7.815 - 0.018) # 数据归一化
        data_window.append(new_data)  # 添加到窗口（自动丢弃旧数据）

        # 更新输入数据（转换为3D数组）
        data_3d = np.array(data_window).reshape((1, len(data_window), 1)).astype(np.float32)
        rospy.loginfo(f"更新窗口数据，当前窗口大小: {len(data_window)}")

        if len(data_window) == 50: # 窗口为50 
            input_name = session.get_inputs()[0].name  # 获取输入节点名称
            output_name = session.get_outputs()[0].name  # 获取输出节点名称
            outputs = session.run([output_name], {input_name: data_3d})
            outputs_tensor = torch.from_numpy(outputs[0]) 
            _, predicted = torch.max(outputs_tensor, dim=1) # 输出概率最大之索引
            predicted = predicted.float()  # 转换为 float32
            predicted_np = predicted.numpy()

            msg = Float32MultiArray(data=[new_data_stamps, predicted])
            pub.publish(msg)
            rospy.loginfo(predicted)


if __name__ == '__main__':
    rospy.init_node('detection')
    rate = rospy.Rate(1)  # 1 Hz
    processor = DataProcessor()
    rospy.spin()