#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/18 下午2:21
# @Author  : LYD Bu Yujun
# @Site    : 
# @File    : UDP_class.py
# @Software: PyCharm
import os
import socket
import struct
import time
import numpy as np
import pandas as pd
import timeout_decorator


class UDP_class:
    AT128_UDP_Struct = np.dtype([
        ('pre-header', [
            ('start_str', 'S2'),
            ('major_version', 'u1'),
            ('minor_version', 'u1'),
            ('header_reserved', 'u2')]
         ),

        ('header', [
            ('laser_num', 'u1'),
            ('block_num', 'u1'),
            ('first_block_return', 'u1'),
            ('distance_unit', 'u1'),
            ('return_number', 'u1'),
            ('flags', 'u1')]
         ),

        ('body', [
            ('block', [
                ('azimuth', 'u2'),
                ('fine_azimuth', 'u1'),
                ('dis_ref', [
                    ('distance', 'u2'),
                    ('reflectivity', 'u1'),
                    ('confidence', 'u1')],
                 128)],
             2),
            ('CRC1', 'u4')]
         ),

        ('tail', [
            ('reserved1', '<u2'),
            ('reserved1_id', 'u1'),
            ('reserved2', '<u2'),
            ('reserved2_id', 'u1'),
            ('high_t_shutdown', 'u1'),
            ('reserved3-1', 'u1'),
            ('reserved3-2', 'u1'),
            ('reserved3-3', 'u1'),
            ('reserved4', 'u1'),
            ('reserved5', 'u1'),
            ('reserved6', 'u1'),
            ('reserved7', 'u1'),
            ('reserved8', 'u1'),
            ('reserved9', 'u1'),
            ('reserved10', 'u1'),
            ('reserved11', 'u1'),
            ('motor_speed', '<i2'),
            ('timestamp', 'u4'),
            ('return_mode', 'u1'),
            ('factory_info', 'u1'),
            ('utc_year', 'u1'),
            ('utc_month', 'u1'),
            ('utc_day', 'u1'),
            ('utc_hour', 'u1'),
            ('utc_min', 'u1'),
            ('utc_sec', 'u1'),
            ('udp_sequence', 'u4'),
            ('crc', 'u4'),
            ('signature', 'S32')]
         )
    ])

    AT128_UDP_Corrections_15 = np.dtype([
        ('eeff', 'u2'),
        ('protocol_version_major', 'u1'),
        ('protocol_version_minor', 'u1'),
        ('laser_num', 'u1'),
        ('mirror_num', 'u1'),
        ('frame_num', 'u1'),
        ('frame_config_byte', 'u8'),
        ('resolution', 'u1'),
        ('start_frame', 'u4', 3),
        ('end_frame', 'u4', 3),
        ('azimuth_offset', 'i4', 128),
        ('elevation', 'i4', 128),
        ('azimuth_adjust', 'i1', 23040),
        ('elevation_adjust', 'i1', 23040),
        ('sha_256_value', 'u1', 32)
    ])

    Pandar128_UDP_Struct = np.dtype([
        ('pre-header', [
            ('start_str', 'S2'),
            ('major_version', 'u1'),
            ('minor_version', 'u1'),
            ('header_reserved', 'u2')]
         ),

        ('header', [
            ('laser_num', 'u1'),
            ('block_num', 'u1'),
            ('first_block_return', 'u1'),
            ('distance_unit', 'u1'),
            ('return_number', 'u1'),
            ('flags', 'u1')]
         ),

        ('body', [
            ('block', [
                ('azimuth', 'u2'),
                ('dis_ref', [
                    ('distance', 'u2'),
                    ('reflectivity', 'u1')],
                 128)],
             2),
            ('CRC1', 'u4')]
         ),

        ('function_safety', [
            ('fs_version', 'u1'),
            ('LFR', 'u1'),
            ('TF', 'u1'),
            ('fault_code_id', 'u2'),
            ('reserved', 'u8'),
            ('CRC2', 'u4')]
         ),

        ('tail', [
            ('reserved1', '<u2'),
            ('reserved1_id', 'u1'),
            ('reserved2', '<u2'),
            ('reserved2_id', 'u1'),
            ('reserved3', '<u2'),
            ('reserved3_id', 'u1'),
            ('azimuth_flag', '<u2'),
            ('high_t_shutdown', 'u1'),
            ('return_mode', 'u1'),
            ('motor_speed', '<u2'),
            ('utc_year', 'u1'),
            ('utc_month', 'u1'),
            ('utc_day', 'u1'),
            ('utc_hour', 'u1'),
            ('utc_min', 'u1'),
            ('utc_sec', 'u1'),
            ('timestamp', 'u4'),
            ('factory_info', 'u1'),
            ('udp_sequence', 'u4'),
            ('IMU_temp', '<i2'),
            ('IMU_acc_unit', 'u2'),
            ('IMU_ang_vel_unit', 'u2'),
            ('IMU_timestamp', 'u4'),
            ('IMU_x_acc', '<i2'),
            ('IMU_y_acc', '<i2'),
            ('IMU_z_acc', '<i2'),
            ('IMU_x_vel', '<i2'),
            ('IMU_y_vel', '<i2'),
            ('IMU_z_vel', '<i2'),
            ('crc3', 'u4')]
         ),
    ])

    UDP_family = pd.DataFrame([['AT128', AT128_UDP_Struct, {200: 5, 300: 8, 400: 10, 500: 13}, 6275],
                               ['Pandar128', Pandar128_UDP_Struct, {600: 10, 1200: 20}, 18000], ],
                              columns=['lidar_type', 'UDP_Struct', 'standard_resolution', 'udp_one_sec'],
                              index=['4.3.128', '1.4.128'])

    def __init__(self, pcap_file=None, host='192.168.1.201', port=2368):
        self.pcap_file = pcap_file
        self.host = host
        self.port = port
        self.data_size = None
        self.udp_key = None
        self.udp_dtype = None
        self.tmp_data_list = []

        self.get_basic_info()
        self.init_body_tail_dtype()

    def get_basic_info(self, packet_num=36000):
        """
        :param packet_num: 为初始化雷达需要的udp包数量
        :return: 获取和打印雷达的基本信息，后续的数据处理算法以此为基础
        """
        try:
            self.get_pretest_info(packet_num)
        except TimeoutError as e:
            print('-----------------------------------------')
            print(e)
            print('初始化时间超过了20秒，请检查FOV设置，可能启用了较小的FOV')
            return
        else:
            print('-----------------------------------------')
            print('恭喜你，初始化成功')
        print('-----------------属性如下-----------------')
        rows = [
            ['雷达类型', self.lidar_type],
            ['标准水平分辨率', '{:.2f}°'.format(self.standard_resolution / 100)],
            ['标准每包角度', '{:.2f}°'.format(self.azm_one_udp / 100)],
            ['包时间间隔', '{:.2f}μs'.format(self.udp_deltatime * 1e6)],
            ['标准每圈包数', '{:.0f}'.format(self.standard_udp_one_round)],
            ['标准每秒包数', '{:.0f}'.format(self.standard_udp_one_sec)],
            ['水平分辨率', '{:.2f}°'.format(self.resolution / 100)],
            ['每圈包数', '{:.0f}'.format(self.udp_one_round)],
            ['每秒包数', '{:.0f}'.format(self.udp_one_sec)],
            ['电机转速', '{:.0f} rpm'.format(self.motor_speed)],
            ['回波模式', '{:.0f}回波'.format(self.return_mode)],
            ['平均每帧包数', '{:.0f}'.format(self.udp_per_frame)],
            ['转动方向', '{:s}'.format(['顺时针', '逆时针'][self.rotation < 0])],
        ]
        for i, area in enumerate(self.firing_area):
            rows.append(['发光范围-{:d}'.format(i + 1), '{}°, {:.0f}包'.format(area, self.firing_udp_number[i])])
        print(pd.DataFrame(rows, columns=['属性', '值']))

    def init_body_tail_dtype(self):
        """
        :return: 初始化body和tail的数据结构，应当实时更新分时复用信息
        """
        self.body_array_dtype = [('udp_sequence', 'u4'), ('azimuth', 'float64'), ('distance', 'float64'),
                                 ('intensity', 'u2'), ('laser_id', 'u2'), ('elevation', 'float64'),
                                 ('calibrated_azi', 'float64')]

        tmp = [x for x, y in self.body_array_dtype]
        self.body_header = ','.join(tmp)
        self.body_fmt = ",".join(["%d"] + ["%.3f"] * 2 + ["%d"] * 2 + ["%.2f"] * 2)

        # ["reserved1", "reserved1_id", "reserved2", "mode_flag", "error_code", "error_code_id",
        #  "azimuth_flag", "high_t_shutdown", "return_mode", "motor_speed", "utc_time", "pcap_timestamp",
        #  "factory_info", "udp_sequence"]

        self.tail_array_dtype = [('azimuth_flag', 'u2'), ('high_t_shutdown', 'u1'), ('return_mode', 'u1'),
                                 ('motor_speed', 'float64'), ('timestamp', 'float64'), ('udp_sequence', 'u4')]
        tmp = [x for x, y in self.tail_array_dtype]
        self.tail_header = ','.join(tmp)
        self.tail_fmt = ",".join(["%d"] * 3 + ["%.2f"] + ["%.6f"] + ["%d"])

        self.pcd_dtype = [('x', 'float64'), ('y', 'float64'), ('z', 'float64'), ('intensity', 'u2')]
        self.pcd_header = ','.join([x for x, y in self.pcd_dtype])
        self.pcd_fmt = ",".join(["%.3f"] * 3 + ["%d"])

    def abandon_udp_in_cache(self):
        """
        :return: 删除未收包状态下，在网口中的缓存数据
        """
        for idx in range(1000):
            t0 = time.time()
            for _ in range(102):
                self.udpsock.recvfrom(int(self.data_size) + 42)
            t1 = time.time()
            if (t1 - t0) * 1e4 > self.udp_deltatime * 1e6:
                # print("丢弃{:d}个udp包，在缓存中".format(103 * idx))
                return 102 * idx
        print("超过101000udp在缓存中, 请确认缓存空间")
        return 101000

    @timeout_decorator.timeout(30)
    def get_one_packet(self):
        """
        :return: 取一个包获取雷达型号，并设定30秒为timeout
        """
        data = b''
        if not self.pcap_file:
            if not self.data_size:
                self.udpsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    self.udpsock.bind(('', self.port))
                except OSError as e1:
                    print(e1)
                    print('端口已被占用，检查是否有其他程序占用端口！')
                    command = "netstat -tunlp | grep 2368 | awk '{print $6}' | cut -f 1 -d '/'"
                    print('占用端口的进程号如下：')
                    os.system(command)
                    print('尝试使用【kill -9 进程号】关闭进程！')
                else:
                    self.udpsock.settimeout(30)

            for _ in range(10):
                data, addr = self.udpsock.recvfrom(1500)
                if len(data) > 600:
                    self.data_size = len(data)
                    break

        else:
            with open(self.pcap_file, 'rb') as file_handle:
                for data_type, data in self.get_packet_from_pcap(file_handle):
                    if data_type == 'u' and len(data) > 600:
                        break

        self.data_size = len(data)
        return data

    @timeout_decorator.timeout(30)
    def get_pretest_info(self, packet_num=36000):
        """
        :param packet_num: 取packet_num个包，获取雷达的基本信息
        :return:
        """
        dataByte = self.get_one_packet()
        protocol_version_major, protocol_version_minor = struct.unpack('<BB', dataByte[2:4])
        self.protocol_version = protocol_version_major + protocol_version_minor / 10
        self.laser_num = struct.unpack('<B', dataByte[6:7])[0]

        # 获得UDP结构体和雷达类型
        self.udp_key = '{:d}.{:d}.{:d}'.format(protocol_version_major, protocol_version_minor, self.laser_num)
        self.udp_dtype = self.UDP_family.loc[self.udp_key]['UDP_Struct']
        data = np.frombuffer(dataByte, dtype=self.udp_dtype, count=1, offset=0)
        self.block_num = data['header']['block_num'][0]
        self.laser_num = data['header']['laser_num'][0]
        self.distance_unit = data['header']['distance_unit'][0].astype('float64') * 0.001
        self.lidar_type = self.UDP_family.loc[self.udp_key]['lidar_type']
        self.udp_one_sec_unit = self.UDP_family.loc[self.udp_key]['udp_one_sec']
        self.resolution_family = self.UDP_family.loc[self.udp_key]['standard_resolution']

        # 收多个包，开始计算基本信息
        dataBytes = []
        if not self.pcap_file:
            while packet_num > 0:
                dataByte, addr = self.udpsock.recvfrom(int(self.data_size) + 42)
                if len(dataByte) > 600:
                    dataBytes.append(dataByte)
                    packet_num -= 1
        else:
            with open(self.pcap_file, 'rb') as file_handle:
                for data_type, dataByte in self.get_packet_from_pcap(file_handle, start_pos=0):
                    if data_type == 'u' and len(dataByte) > 600:
                        dataBytes.append(dataByte)
                        packet_num -= 1
                        if packet_num < 1:
                            break

        dataBytes = b''.join(dataBytes)
        datas = np.frombuffer(dataBytes, dtype=self.udp_dtype)
        azm_series = datas['body']['block']['azimuth'].ravel()
        one_azm_series = datas['body']['block']['azimuth'][:, 0]
        self.motor_speed = round(datas['tail']['motor_speed'][-1], -1)
        self.operation_mode = datas['tail']['high_t_shutdown'][-1]
        if self.udp_dtype == self.AT128_UDP_Struct:
            self.motor_speed = round(self.motor_speed / 10, -1)
        self.return_mode = 1 if datas['tail']['return_mode'][-1] < 57 else 2
        # 每秒包数，查表
        self.standard_udp_one_sec = self.udp_one_sec_unit * self.return_mode
        # 标准分辨率，查表
        self.standard_resolution = self.resolution_family[self.motor_speed]
        # 转动方向
        azm_diff = [int(one_azm_series[i + 1]) - int(one_azm_series[i]) for i in range(len(one_azm_series) - 1)]
        self.rotation = 1 if sum([diff > 0 for diff in azm_diff]) > 0 else -1
        # 出现最多的角度差
        max_count_value = max(azm_diff, key=azm_diff.count)
        if self.rotation == 1:
            norm_azm = [diff for i, diff in enumerate(azm_diff) if max_count_value * 3 > diff > 0]
        else:
            norm_azm = [diff for i, diff in enumerate(azm_diff) if max_count_value * 3 < diff < 0]
        # 每圈的UDP个数
        self.standard_udp_one_round = self.standard_udp_one_sec / (self.motor_speed / 60)
        # 每包的角度
        self.azm_one_udp = self.standard_resolution * (self.block_num / self.return_mode)
        # 每包的时间间隔
        self.udp_deltatime = self.azm_one_udp / 100 / (360 * self.motor_speed / 60)
        # 计算的分辨率
        self.resolution = np.round(np.mean(norm_azm), 1) / (self.block_num / self.return_mode)
        # 发光角度范围
        azm_list = list(set(azm_series))
        azm_list.sort()
        azm_set = [-510, *azm_list, 36510]
        end_angle = [azm_set[i] / 100 for i in range(len(azm_set) - 1) if azm_set[i + 1] - azm_set[i] > 500]
        start_angle = [azm_set[i + 1] / 100 for i in range(len(azm_set) - 1) if azm_set[i + 1] - azm_set[i] > 50]
        self.start_angle = start_angle[:-1]
        self.end_angle = end_angle[1:]
        self.firing_area = [[a, b] for a, b in zip(start_angle[:-1], end_angle[1:])]
        self.firing_udp_number = [(b - a) / self.azm_one_udp * 100 for a, b in zip(start_angle[:-1], end_angle[1:])]
        self.udp_per_frame = np.mean(self.firing_udp_number)
        self.udp_one_round = round(sum([(area[-1] - area[0] + self.standard_resolution / 100) / self.azm_one_udp * 100
                                        for area in self.firing_area]))
        self.udp_one_sec = self.udp_one_round * (self.motor_speed / 60)

    def get_packet_from_pcap(self, file_handle, start_pos=0):
        """
        :param file_handle: open with之后得到的pcap文件
        :param start_pos: 开始读的位置
        :return: 将pcap切片成udp包，按照生成器的方式输出udp包
        """
        if start_pos:
            first_flag = 0
            file_handle.seek(start_pos, 0)
        else:
            first_flag = 1

        chunk_data = 16 + 42

        while True:
            if first_flag:
                udp_info = file_handle.read(82)
                data_type = udp_info[40 + 23]
                if data_type == 17:
                    first_data_size = struct.unpack('>H', udp_info[78:80])[0] - 8
                    first_data = file_handle.read(first_data_size)
                    if first_data[0:2] == b'\xee\xff':
                        # print('first data is point cloud udp')
                        first_flag = 0
                        yield 'u', first_data
                    elif first_data[0:2] == b'\xcd\xdc':
                        print('first data is safety udp')
                        first_flag = 0
                        yield 'u', first_data
                    elif first_data[0:2] == b'\xff\xee':
                        print('first data is gps udp')
                        first_flag = 0
                        yield 'u', first_data
                elif data_type == 6:
                    first_data_size = struct.unpack('>H', udp_info[56:58])[0] - 28
                    # print(struct.unpack('>H', udp_info[56:58])[0])
                    first_data = udp_info[74:] + file_handle.read(first_data_size)
                    # print('first data is tcp, data size is {}'.format(first_data_size + 8))
                    first_flag = 0
                    yield 't', first_data
            else:
                udp_info = file_handle.read(chunk_data)
                try:
                    data_type = udp_info[16 + 23]
                except (struct.error, IndexError):
                    print(len(udp_info))
                    return
                if data_type == 17:
                    data_size = struct.unpack('>H', udp_info[chunk_data - 4:chunk_data - 2])[0] - 8
                    data = file_handle.read(data_size)
                    yield 'u', data
                elif data_type == 6:
                    data_size = struct.unpack('>H', udp_info[32:34])[0] - 28
                    data = udp_info[chunk_data - 8:] + file_handle.read(data_size)
                    yield 't', data
