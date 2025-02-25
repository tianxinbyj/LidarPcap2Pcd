#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bu Yujun
# @Time     : 11/2/22 9:29 AM
# @File     : Main.py
# @Project  : Pcap2pcd
import time

from Pcap2pcd import Pcap2pcd

if __name__ == '__main__':

    # 1. 选择pcap文件
    # pcap_file = '/home/buyujun/Downloads/n00015-2022-11-21-19-38-26-21_Pandar128.pcap'
    pcap_file = '/home/zhangliwei01/ZONE/Lidar/Pandar128/n000003_2023-06-02-14-57-55-120_Pandar128.pcap'
    lidar = Pcap2pcd(pcap_file)

    # 2. 自动加载角度文件
    lidar.get_calibration()
    lidar.get_firetime()

    # 3. 开始转换
    cur_pos = 0
    t0 = time.time()
    frame_index = 0
    while True:
        dataBytes, cur_pos, field = lidar.slice_pcap_into_frame(cur_pos)
        if cur_pos == -1:
            break
        pcd_name = lidar.parse_frame_data_pcd(dataBytes)
        frame_index += 1
        print(cur_pos, frame_index, pcd_name)
        break
        # if frame_index == 10:
        #     break

    print("time cost {:f} s to get {:d}".format(time.time() - t0, frame_index))