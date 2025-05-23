#!/usr/bin/env python3
#encoding=utf-8

import rospy
from std_msgs.msg import String
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(f'/home/geneta/anaconda3/envs/pytorch/lib/python3.8/site-packages/') 
import torch
import onnxruntime as ort

model_path = "/home/geneta/project/model/model.onnx"
session = ort.InferenceSession(model_path)
data = np.genfromtxt(f'/home/geneta/dataset/1408.csv', delimiter=',', skip_header=0) # 读入为double(float64)
data_3d = data.reshape((1, 50, 1)).astype(np.float32) # 调整为3D输入，强制转为float32
print(data_3d.size)

def detection():
    input_name = session.get_inputs()[0].name  # 获取输入节点名称
    output_name = session.get_outputs()[0].name  # 获取输出节点名称
    outputs = session.run([output_name], {input_name: data_3d})
    outputs_tensor = torch.from_numpy(outputs[0]) 
    _, predicted = torch.max(outputs_tensor, dim=1)

    pub = rospy.Publisher('detection', String, queue_size=10)
    rospy.init_node('detection', anonymous=True)
    rate = rospy.Rate(1)  # 1 Hz
    count = 0

    while not rospy.is_shutdown():
        hello_str = "输入名称:%s, 输出名称:%s" % (input_name, output_name)
        rospy.loginfo(predicted)
        # pub.publish(hello_str)
        rate.sleep()
        count += 1

if __name__ == '__main__':
    try:
        detection()
    except rospy.ROSInterruptException:
        pass
