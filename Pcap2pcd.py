#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bu Yujun
# @Time     : 10/31/22 2:07 PM
# @File     : Pcap2pcd.py
# @Project  : Pcap2pcd

import os
import time

import numpy as np
import pandas as pd
# import open3d as o3d
from scipy.interpolate import interp1d

from UDP_class import UDP_class


class Pcap2pcd(UDP_class):

    def __init__(self, pcap_file):

        self.pcap_file = pcap_file

        UDP_class.__init__(self, pcap_file, '192.168.1.201', 2368)
        self.init_parameters()
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        _, pcap_name = os.path.split(pcap_file)
        self.pcd_folder, _ = os.path.splitext(pcap_file)
        self.tag = pcap_name.split('-')[0]
        if not os.path.exists(self.pcd_folder):
            os.mkdir(self.pcd_folder)

    def get_calibration(self, calib_file=None):
        """
        :param calib_file: 角度文件，仅在需要离线上传的时候需要
        :return: 解析完角度文件的参数
        """

        # 原始角度文件中_adjust为2°一个间隔，将至差值为0.01°一个间隔
        def interpolate_for_correction_AT(raw_adjust, interval=1):
            a, b = raw_adjust.shape
            new_adjust = np.zeros((a, int(36000 / interval)))
            for i, line in enumerate(raw_adjust):
                raw_x = np.linspace(0, 36000, b + 1)
                raw_y = np.append(line, 0)
                f = interp1d(raw_x, raw_y)
                new_x = np.arange(0, 36000, interval)
                new_adjust[i, :] = f(new_x)
            return new_adjust

        if not calib_file:
            if self.lidar_type == 'AT128':
                calib_file = os.path.join(self.root_dir, 'params/correction/AT128.dat')
            else:
                calib_file = os.path.join(self.root_dir, 'params/correction/Pandar128.csv')

        self.get_firetime()

        if self.lidar_type == 'AT128':
            with open(calib_file, 'rb') as file_handle:
                calib_raw = file_handle.read(72572)
                calib_data = np.frombuffer(calib_raw, dtype=self.AT128_UDP_Corrections_15, count=1,
                                           offset=0)
            self.calib_data = pd.DataFrame()
            self.calib_data['Azimuth'] = - calib_data['azimuth_offset'][0] / 256 / 100
            self.calib_data['Elevation'] = calib_data['elevation'][0] / 256 / 100
            self.start_frame = calib_data['start_frame'].ravel()
            azimuth_adjust = calib_data['azimuth_adjust'][0].reshape(128, 180)
            elevation_adjust = calib_data['elevation_adjust'][0].reshape(128, 180)
            self.azimuth_adjust = interpolate_for_correction_AT(azimuth_adjust)
            self.elevation_adjust = interpolate_for_correction_AT(elevation_adjust)
            return self.calib_data, self.azimuth_adjust, self.elevation_adjust
        else:
            self.calib_data = pd.read_csv(calib_file, index_col=False)
            self.azi_correction = np.array(self.calib_data, dtype='float64')[:, 2]  # (128, )
            self.ele_correction = np.array(self.calib_data, dtype='float64')[:, 1]
            self.azi_correction_rad = np.deg2rad(np.array(self.calib_data, dtype='float64')[:, 2])  # (128, )
            self.ele_correction_rad = np.deg2rad(np.array(self.calib_data, dtype='float64')[:, 1])
            return self.calib_data, None, None

    def get_firetime(self, firetime_file=None):
        """
        :param firetime_file: 发光时序文件要求
        :return:
        """
        if self.udp_key == '1.4.128':
            if firetime_file is None:
                firetime_file = os.path.join(self.root_dir, 'params/firetime/P128_4.2_4.5_firetime.xlsx')
            df = pd.read_excel(firetime_file)
            self.speed = self.motor_speed / 60 * 360 / 1e9
            self.fire_time = np.array(np.array(df)[2:, 1:] * self.speed, dtype='float64')
            self.near_mode = float(df.columns.values[0])

            self.op_sidx = [0, 8, 8, 12][self.operation_mode]
            self.op_eidx = [8, 0, 12, 16][self.operation_mode]
            self.firetime_aft_op = np.deg2rad(self.fire_time[:, self.op_sidx:self.op_eidx])
            self.base_ft_idx_array = np.arange(128, dtype='uint8').reshape(-1, 1) * self.firetime_aft_op.shape[1]
            self.base_idx_array = np.arange(128, dtype='uint8').reshape(-1, 1)

            self.base_ft_idx_vector = np.arange(128, dtype='uint8') * self.firetime_aft_op.shape[1]
            self.base_idx_vector = np.arange(128, dtype='uint8')

            self.a = -1
            self.b = 0.012
            self.h = 0.04

    def init_parameters(self):
        """
        :return: 定义其他地方会用到的参数值
        """
        self.ratio1 = self.block_num.astype('uint32') * self.laser_num.astype('uint32')
        self.dist_min = 0.01
        self.dist_max = 300
        self.azi_min = 0
        self.azi_max = 36000
        self.id_min = 0
        self.id_max = self.laser_num

    def slice_pcap_into_frame(self, start_pos):
        """
        :param start_pos: 将pcap切割的位置，用于迭代，每次输出一帧
        :return: 一帧的数据
        """
        dataBytes = []
        field = None
        break_flag = 0
        last_cur = 0
        last_azi = 0
        # 切帧角度
        end_angle = [angle * 100 for angle in self.end_angle]

        if not self.tmp_data_list:
            with open(self.pcap_file, 'rb') as file_handle:
                for data_type, data in self.get_packet_from_pcap(file_handle, start_pos=start_pos):
                    # 前后两包角度，判断是否切割
                    # 前一个角度位于范围内，且当前角度在范围内，则认为此时应当切割
                    if len(data) == self.data_size:
                        udp_i = np.frombuffer(data, dtype=self.udp_dtype, count=1, offset=0)
                        azm_i = udp_i['body']['block']['azimuth'][0][0]
                        cur_pos = file_handle.tell()
                        for idx, e_angle in enumerate(end_angle):
                            low_limit = e_angle - self.azm_one_udp
                            high_limit = e_angle + 0.5 * self.azm_one_udp
                            if low_limit < last_azi < high_limit and (high_limit <= azm_i or azm_i <= low_limit):
                                field = idx
                                break_flag = 1
                                break
                        if break_flag:
                            break
                        last_azi = azm_i
                        last_cur = cur_pos
                        dataBytes.append(data)

            cur_pos = last_cur
            if not cur_pos:
                cur_pos = -1

            udps = np.frombuffer(b''.join(dataBytes), dtype=self.udp_dtype)
            seqs = udps['tail']['udp_sequence']
            azms = udps['body']['block']['azimuth'][:, 0]
            data_len = len(dataBytes)

            # 处理意外情况
            # 1.丢包
            # 2.角度发生跳跃，即中间有不发光的角度
            # 角度跳跃的限值定为 10个UDP包
            low_limit = - self.azm_one_udp * 10
            high_limit = - low_limit
            special_ids = np.array([idx + 1 for idx in range(data_len - 1)
                                    if seqs[idx + 1] - seqs[idx] > 1
                                    or float(azms[idx + 1]) - float(azms[idx]) > high_limit
                                    or float(azms[idx + 1]) - float(azms[idx]) < low_limit])

            if not len(special_ids):
                return dataBytes, cur_pos, field
            else:
                delta_azms = np.array([float(azms[i]) - float(azms[i - 1]) for i in special_ids])
                delta_seqs = np.array([seqs[i] - seqs[i - 1] for i in special_ids])
                as_ratio = delta_azms / delta_seqs
                sep_frm = np.logical_or(self.azm_one_udp * 1.1 < as_ratio, as_ratio < self.azm_one_udp * 0.9)
                if any(sep_frm):
                    tmp = special_ids[np.nonzero(sep_frm * special_ids)]
                    print('pcap会被切成{:d}块, 接下来{:d}个位置相同'.format(len(tmp) + 1, len(tmp)))
                    frm_idx = [None] + list(tmp) + [None]
                    tmp = [(frm_idx[i], frm_idx[i + 1]) for i in range(len(frm_idx) - 1)]
                    self.tmp_data_list = [dataBytes[x:y] for x, y in tmp]
                    dataBytes = self.tmp_data_list[0]
                    self.tmp_data_list = self.tmp_data_list[1:]
                    self.field = field
                    return dataBytes, cur_pos, None
                else:
                    return dataBytes, cur_pos, field
        else:
            dataBytes = self.tmp_data_list[0]
            self.tmp_data_list = self.tmp_data_list[1:]
            return dataBytes, start_pos, self.field

    def parse_frame_data_pcd(self, dataBytes):

        data_lens = len(dataBytes)
        databytes = b''.join(dataBytes)
        udps = np.frombuffer(databytes, dtype=self.udp_dtype)

        utc_time = self.transfer_utc_time(udps['tail']['utc_year'][0], udps['tail']['utc_month'][0],
                                          udps['tail']['utc_day'][0], udps['tail']['utc_hour'][0],
                                          udps['tail']['utc_min'][0],
                                          udps['tail']['utc_sec'][0])
        date_time = '{:s}.{:0>6d}'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(utc_time + 28800)), udps['tail']['timestamp'][0])

        body_lens = data_lens * self.ratio1
        ratio2 = self.block_num.astype('uint32') * data_lens
        body_datas = np.zeros(body_lens, dtype=self.body_array_dtype)

        azm_tmp = udps['body']['block']['azimuth']
        if self.udp_key == '4.3.128':
            azm_array = azm_tmp + udps['body']['block']['fine_azimuth'] / 256  # AT
        else:
            azm_array = azm_tmp

        body_datas['azimuth'] = np.repeat(azm_array, self.laser_num)
        body_datas['distance'] = udps['body']['block']['dis_ref']['distance'].flatten() * self.distance_unit
        body_datas['intensity'] = udps['body']['block']['dis_ref']['reflectivity'].flatten()
        body_datas['laser_id'] = np.tile(np.arange(self.laser_num), ratio2)
        ele_correction = self.calib_data['Elevation'].values

        # 以下为修正函数
        if self.udp_key == '4.3.128':
            frame_idx = (body_datas['azimuth'] < 12000) \
                        + np.logical_and(body_datas['azimuth'] >= 12000, body_datas['azimuth'] < 24000) * 2 \
                        + (body_datas['azimuth'] >= 24000) * 3 - 1
            d_azi1 = (body_datas['azimuth'] - self.start_frame[frame_idx] / 256) * 2
            d_azi2 = self.calib_data['Azimuth'].values[body_datas['laser_id']]
            d_azi3 = self.azimuth_adjust[:, azm_tmp].ravel('F')
            body_datas['calibrated_azi'] = (d_azi1 + d_azi3) / 100 + d_azi2
            delta_ele = self.elevation_adjust[:, azm_tmp].ravel('F') / 100
            body_datas['elevation'] = ele_correction[body_datas['laser_id']] + delta_ele

        elif self.udp_key == '1.4.128':
            azi_flag_array = np.zeros(shape=(data_lens * int(self.block_num),), dtype='uint8')
            azi_flag_array[::2, ] = udps['tail']['azimuth_flag'] >> 6
            azi_flag_array[1::2, ] = ((udps['tail']['azimuth_flag'] >> 4) & 0b0011)

            near_flag_array = body_datas['distance'] < self.near_mode
            idx_array = np.tile(self.base_ft_idx_vector, data_lens * 2) \
                        + azi_flag_array.repeat(128) * 2 + near_flag_array
            azi = np.take(self.firetime_aft_op, idx_array)  # (921600, 1)

            delta_azi = azi + np.tile(self.azi_correction, data_lens * 2)
            # delta_ele = np.tile(self.ele_correction, data_lens * 2)
            body_datas['calibrated_azi'] = delta_azi + body_datas['azimuth'] / 100
            body_datas['elevation'] = ele_correction[body_datas['laser_id']]

        pcd_datas = np.zeros((body_lens, 4), dtype=np.float)
        pcd_datas[:, 2] = np.sin(np.deg2rad(body_datas['elevation'])) * body_datas['distance']
        pcd_datas[:, 0] = np.cos(np.deg2rad(body_datas['elevation'])) * body_datas['distance'] * np.sin(
            np.deg2rad(body_datas['calibrated_azi']))
        pcd_datas[:, 1] = np.cos(np.deg2rad(body_datas['elevation'])) * body_datas['distance'] * np.cos(
            np.deg2rad(body_datas['calibrated_azi']))
        # pcd_datas[:, 4] = body_datas['intensity'] / 255
        # pcd_datas[:, 5] = 1
        pcd_datas[:, 3] = body_datas['intensity']

        pcd_header = '''# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {:d}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {:d}
DATA ascii
'''.format(pcd_datas.shape[0], pcd_datas.shape[0])

        pcd_name = '{:s}_{:s}_{:s}.pcd'.format(self.tag, date_time, self.lidar_type)
        path = os.path.join(self.pcd_folder, pcd_name)

        with open(path, 'w') as f:
            f.write(pcd_header)
            for line in pcd_datas:
                f.write('{:.4f} {:.4f} {:.4f} {:.0f}\n'.format(line[0], line[1], line[2], line[3]))

        return pcd_name

    def transfer_utc_time(self, *args):
        x = list(args) + [0, 0, 0]
        if x[0] < 100:
            x[0] = x[0] + 1970
        else:
            x[0] = x[0] + 1970 - 70
        return time.mktime(time.struct_time(x))
