import struct
import os
import sys
import open3d
import numpy as np
import cv2
import json


def get_files(path):
    file_list = []
    for (root, _, files) in os.walk(path):
        if len(files) > 0 and files[0] != '.DS_Store':
            for fn in files:
                if os.path.splitext(fn)[1] in ['.json', '.png', '.pcd']:
                    file_list.append(
                        os.path.join(root, fn)
                    )
    return file_list


def get_file_names(file_list):
    return [os.path.splitext(os.path.basename(fn))[0] for fn in file_list]


def read_class_names(path):
    try:
        cls_names = {}
        with open(path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                name = line.rstrip('\n')
                if name not in cls_names:
                    cls_names[name] = len(cls_names)

        reverse_cls_names = {v: k for k, v in cls_names.items()}
        return cls_names, reverse_cls_names
    except FileNotFoundError:
        return None, None


def read_pcd_data(path):
    return np.asarray(open3d.io.read_point_cloud(path).points)


def read_segmentation_file(path):
    with open(path, 'r') as f:
        seg_data = json.load(f)

    return seg_data['meta'], seg_data['objects']


def read_mask_file(path, mode='None'):
    mask_data = cv2.imread(path, mode)
    mask_values = np.unique(mask_data)
    return mask_data, mask_values


def read_camera_intrinsic_file(path):
    camera_matrix, rectification, projection = [], [], []

    with open(path, 'r') as f:
        while True:
            line = f.readline()

            if 'camera matrix' in line:
                fp = f.tell()
                f.seek(fp)
                for i in range(0, 3):
                    element = list(map(float, f.readline().rstrip('\n').split()))
                    camera_matrix.append(element)
                    f.seek(f.tell())
            elif 'rectification' in line:
                fp = f.tell()
                f.seek(fp)
                for i in range(0, 3):
                    element = list(map(float, f.readline().rstrip('\n').split()))
                    rectification.append(element)
                    f.seek(f.tell())
            elif 'projection' in line:
                fp = f.tell()
                f.seek(fp)
                for i in range(0, 3):
                    element = list(map(float, f.readline().rstrip('\n').split()))
                    projection.append(element)
                    f.seek(f.tell())

            if len(camera_matrix) != 0 and len(rectification) != 0 and len(projection) != 0:
                break

    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
    rectification = np.asarray(rectification, dtype=np.float32)
    projection = np.asarray(projection, dtype=np.float32)
    assert camera_matrix.shape[0] == 3 and camera_matrix.shape[1] == 3
    assert rectification.shape[0] == 3 and rectification.shape[1] == 3
    assert projection.shape[0] == 3 and projection.shape[1] == 4

    return camera_matrix, rectification, projection


def read_camera_extrinsic_file(path):
    camera_matrix = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elements = list(map(float, line.rstrip('\n').split()))
            if len(elements) != 0:
                camera_matrix.append(elements)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
    assert camera_matrix.shape[0] == 4 and camera_matrix.shape[1] == 4

    return camera_matrix[:3, :]


def read_calibration_file(path, is_intrinsic=True):
    if is_intrinsic:
        camera_matrix, rectification, projection = read_camera_intrinsic_file(path)
        return camera_matrix, rectification, projection
    else:
        camera_matrix = read_camera_extrinsic_file(path)
        return camera_matrix


def get_target_data(path):
    target_list = []
    for (root, _, files) in os.walk(path):
        if 'calibration' in root and len(files) > 0 and files[0] != '.DS_Store':
            target_list.append(
                root.replace(path, '').split('/')[:-1]
            )
    return target_list

