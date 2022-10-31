import struct
import os
import sys
import open3d
import numpy as np
import cv2
import json


def get_files(path):
    """
    인자로 받은 path 내의 파일을 리스트 형태로 가져 오는 메소드
        파일 확장자가 '.json', '.png', '.pcd' 인 파일을 가져옴

    Args:
        path       파일을 가져올 경로(directory path)
    Returns:
        3가지 타입의 확장자를 가진 파일들이 담긴 리스트
        인자로 받은 path를 포함한 경로를 포함한 파일이 담김
        ex) [/home/users/PycharmProjects/example1.png,
             home/users/PycharmProjects/example2.png,
             home/users/PycharmProjects/example3.png, ...]
    """
    file_list = []
    for (root, _, files) in os.walk(path):
        if len(files) > 0:
            for fn in files:
                if os.path.splitext(fn)[1] in ['.json', '.png', '.pcd']:
                    file_list.append(
                        os.path.join(root, fn)
                    )
    return file_list


def get_file_names(file_list, target='', is_existing_file=False):
    """
    파일 리스트에 담긴 파일(경로 포함)의 파일명만 가지고 옴.
        즉, 확장자 제외

    단, target 데이터가 'DOGU/CP/a/cloudy/220719/17-19' 이거나, 기존에 있는 파일을 가져오는 거라면,
        확장자만 제외하고 가져오고,
        나머지 target에 대해서는 '날짜_시간_ms.확장자' 라는 파일명에서 '날짜_시간'만 가져옴

    Args:
        file_list      파일 리스트
    Return:
        인자로 받은 파일 리스트 내의 확장자를 제외한 파일명 또는 ms 단위 파일을 제외한 파일이 담긴 리스트
    """
    if '/'.join(target) == 'DOGU/CP/a/cloudy/220719/17-19' or is_existing_file:
        return [os.path.splitext(os.path.basename(fn))[0] for fn in file_list]
    return ['_'.join(os.path.splitext(os.path.basename(fn))[0].split('_')[:-1]) for fn in file_list]


def read_class_names(path):
    """
    class name이 담긴 텍스트 파일을 읽는 메소드
    파일이 존재하면, 딕셔너리 형태의 class_name dict와 reverse 딕셔너리를 반환함
        ex) class_names = {'human': 0,
                           'tree': 1},
            reverse_class_names = {0: 'human',
                                   1: 'tree'}

    Args:
        path       class name 경로
    Return:
        파일이 존재하면, class_name_dict, reverse_class_name_dict 반환
        없으면 None, None 반환
    """
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
    """
    Point Cloud Data(.pcd) 파일을 읽고 (x, y, z)가 담긴 넘파이 array를 반환하는 메소드

    Args:
        path       pcd data 경로
    Return:
        numpy array 형태의 pcd point 데이터. (num of points, 3)
    """
    return np.asarray(open3d.io.read_point_cloud(path).points)


def read_segmentation_file(path):
    """
    segmentation 파일(.json)을 읽고, meta 데이터와 object 데이터를 반환하는 메소드
        meta 데이터에는 width, height, object 데이터에는 annotation, class name 데이터가 포함되어 있음

    Args:
        path       segmentation 파일 경로
    Return:
        (meta 데이터, objects 데이터)를 반환
    """
    with open(path, 'r') as f:
        seg_data = json.load(f)

    return seg_data['meta'], seg_data['objects']


def read_mask_file(path, mode='None'):
    """
    이미지 형태의 mask 데이터를 읽고, 해당 데이터와 포함되어 있는 value(unique)를 반환하는 메소드

    Args:
        path        mask 데이터 경로
        mode        cv2를 통한 이미지 읽기 떄 사용되는 mode. (cv2.IMREAD_GRAYSCALE, ...)
    Return:
        (mask data, mask values(unique))
    """
    mask_data = cv2.imread(path, mode)
    mask_values = np.unique(mask_data)
    return mask_data, mask_values


def read_camera_intrinsic_file(path):
    """
    내부 파라미터 파일을 읽고 camera matrix, rectification, projection 정보를 번환하는 메소드
        LiDAR 데이터를 이미지로 projection 할 때 필요한 값들
    파일 내에 camera matrix, rectification, projection 이 포함된 line을 읽고 찾은 파일 포인터의 다음 3줄을 읽어 값을 얻음

    Args:
        path        내부 파라미터 파일 경로
    Return:
        (camera matrix, rectification, projection)

    """
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
    """
    내부 파라미터 파일을 읽고 camera matrix, rectification, projection 정보를 번환하는 메소드
        LiDAR 데이터를 이미지로 projection 할 때 필요한 값
    extrinsic 파일은 모두 (4, 4)로 이루어져 있는데, 마지막 행은 제외하고 반환한다.

    Args:
        path        외부 파라미터 파일 경로
    Return:
        (3, 4) 크기의 파라미터 반환
    """
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
    """
    camera intrinsic, camera extrinsic 파일을 읽는 메소드.
        is_intrinsic 인자에 따라 반환하는 값이 달라짐

   #################
    # 사용 안함!!!!
    ##############

    Args:
        path                camera parameter 파일 경로
        is_intrinsic        내부 파라미터 파일을 읽을 것인지 외부 파라미터 파일을 읽을 것인지 결정
    Return:
        is_intrinsic인자에 따라 파라미터 값을 반환

    """
    if is_intrinsic:
        camera_matrix, rectification, projection = read_camera_intrinsic_file(path)
        return camera_matrix, rectification, projection
    else:
        camera_matrix = read_camera_extrinsic_file(path)
        return camera_matrix


def get_target_data(path):
    """
    데이터 전처리를 할 대상 파일들을 가져오는 메소드.
        대상 파일은 calibration 이 존재하는 데이터만 가져옴.

    구글 드라이브에 포함되어 있는 파일을 확인해보면, '.DS_Store'가 포함되어 있음. 이는 MacOS 환경에서 압축 또는 압축해제를 했기 때문에 생기는 파일.
        따라서, os.walk로 가져올 때 files에 ['.DS_Store'] 가 있는 경우가 생김. 이런 경우는 제외함

    Args:
        path        데이터 root 경로. (ex, ~/52_2/)
    Return:
        데이터 전처리를 할 대상 파일이 담긴 경로를 담은 리스트를 반환
        ex, [['DOGU', 'CP', 'a', 'cloudy', '220719', '17-19'],
             ['RASTECH', 'DCC', 'D', 'cloudy', '220901', '11-14'],
             ['RASTECH', 'DCC', 'D', 'cloudy', '220902', '14-19'],
             ...]

             ['DOGU', 'CP', 'a', 'cloudy', '220719', '17-19'] 아래에 ['seg', 'refine', 'calibration'] 디렉토리가 있음!!
            README의 Data Structure 확인!
    """
    target_list = []
    for (root, _, files) in os.walk(path):
        if 'calibration' in root and len(files) > 0 and files[0] != '.DS_Store':
            target_list.append(
                root.replace(path, '').split('/')[:-1]
            )
    return target_list

