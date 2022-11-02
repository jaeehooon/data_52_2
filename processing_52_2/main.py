import random
import sys
import os
import argparse
import shutil
from utils.utils import *
from lidar_to_depth.lidar_to_depth import LiDAR2CameraKITTI
from depth2hha.getHHA import getHHA
from depth2hha.utils.getCameraParam import get_camera_param
from config import predefined_class_names


def define_args():
    p = argparse.ArgumentParser()

    p.add_argument('--data_root', type=str, default="../dataset/52_2/",
                   help="the path of original data")
    p.add_argument('--output_path', type=str, default="../dataset/52_2/final/",
                   help="the path of output data")
    p.add_argument('--class_name_path', type=str, default='./class_names.txt',
                   help='the path of class names')
    p.add_argument('--save_class_name', type=bool, default=True,
                   help="whether you'll save the class name file or not")
    config = p.parse_args()
    return config


def main(config):
    class_names, reverse_cls_names = read_class_names(config.class_name_path)
    if class_names is None:
        class_names = {
            'unlabeled': 0
        }
    assert predefined_class_names == class_names, "!!"                #

    data_root = config.data_root
    output_path = config.output_path
    # print('current path: ', os.getcwd())
    # print("Data Root: ", data_root)
    # print("Output Path: ", output_path)
    # print(os.listdir(output_path))

    target_list = get_target_data(data_root)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_rgb = os.path.join(output_path, 'RGBFolder')
    output_label = os.path.join(output_path, 'LabelFolder')
    output_modal = os.path.join(output_path, 'ModalXFolder')

    # print(len(os.listdir(output_rgb)))
    # print(len(os.listdir(output_label)))
    # print(len(os.listdir(output_modal)))

    if not os.path.exists(output_rgb):
        os.makedirs(output_rgb)
    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_modal):
        os.makedirs(output_modal)

    existing_file_names = get_file_names(get_files(output_label))

    for idx, target in enumerate(target_list):
        print("\n\nIdx >>> ", idx + 1)
        print("  ", '/'.join(target))
        comp, loc, scene, weather, date, timezone = target

        label_dir = os.path.join(data_root, comp, loc, scene, weather, date, timezone, 'seg')
        refine_dir = os.path.join(data_root, comp, loc, scene, weather, date, timezone, 'refine')
        calib_dir = os.path.join(data_root, comp, loc, scene, weather, date, timezone, 'calibration')

        if len(os.listdir(calib_dir)) > 0:

            # de_id_dir = os.path.join(label_dir, "de-identification")
            # mask_dir = os.path.join(label_dir, "mask")
            # seg_dir = os.path.join(label_dir, "segmentation")
            # camera_dir = os.path.join(refine_dir, 'camera')
            # pcd_dir = os.path.join(refine_dir, 'pcd')

            # get common file names
            de_id_list = get_files(os.path.join(label_dir, "de-identification"))
            mask_list = get_files(os.path.join(label_dir, "mask"))
            seg_list = get_files(os.path.join(label_dir, "segmentation"))
            camera_list = get_files(os.path.join(refine_dir, 'camera'))
            pcd_list = get_files(os.path.join(refine_dir, 'pcd'))

            de_id_fn_list = get_file_names(de_id_list)
            mask_fn_list = get_file_names(mask_list)
            seg_fn_list = get_file_names(seg_list)
            camera_fn_list = get_file_names(camera_list)
            pcd_fn_list = get_file_names(pcd_list)
            print(pcd_fn_list[:4])

            common_file_names = get_common_files((de_id_fn_list, mask_fn_list, seg_fn_list), pcd_fn_list, target)
            print('#############')
            print("가공 데이터 공통 파일 수 : ", len(set(de_id_fn_list) & set(mask_fn_list) & set(seg_fn_list)))
            print("가공 데이터(seg)와 정제 데이터(refine/pcd)와 공통 파일 수: ", len(common_file_names))
            print("############")

            # 이미 final 에 존재하는 파일 개수, 처리 해야하는 파일 개수
            non_target, target = remove_existing_files(existing_file_names, common_file_names, target)
            print("# of the files already existed: \t", len(non_target))
            print("# of the target files (common file names): \t", len(target))

            print("# of file names: \t", len(target))
            print("  label")
            print("\t>>> de-identification:\t{:,d}/{:,d}".format(len(target), len(de_id_fn_list)))
            print("\t>>> mask:\t\t\t\t{:,d}/{:,d}".format(len(target), len(mask_fn_list)))
            print("\t>>> segmentation:\t\t{:,d}/{:,d}".format(len(target), len(seg_fn_list)))

            print("  Data")
            print("\t>>> camera:\t\t\t\t{:,d}/{:,d}".format(len(target), len(camera_fn_list)))
            print("\t>>> pcd:\t\t\t\t{:,d}/{:,d}".format(len(target), len(pcd_fn_list)))
            print()

            #############
            # labeling
            ###############
            de_id_list = sorted([fn for fn in de_id_list
                                 if os.path.splitext(os.path.basename(fn))[0] in target])
            mask_list = sorted([fn for fn in mask_list
                                if os.path.splitext(os.path.basename(fn))[0] in target])
            seg_list = sorted([fn for fn in seg_list
                               if os.path.splitext(os.path.basename(fn))[0] in target])
            pcd_list = sorted([fn for fn in pcd_list
                               if os.path.splitext(os.path.basename(fn))[0] in target])

            # print("de_id_list: ", de_id_list[:2], '...')
            # print("mask_list: ", mask_list[:2], '...')
            # print("seg_list: ", seg_list[:2], '...')
            # print("pcd_list: ", pcd_list[:2], '...')

            for idx, (de_id_path, mask_path, seg_path, pcd_path) in enumerate(zip(de_id_list, mask_list, seg_list, pcd_list)):
                fn = os.path.splitext(os.path.basename(de_id_path))[0]

                if idx % 4 == 0 or idx == len(de_id_list) - 1:
                    print('\r{:,d}/{:,d}'.format(idx + 1, len(de_id_list)), end='')

                mask_data, mask_values = read_mask_file(mask_path, mode=cv2.IMREAD_GRAYSCALE)
                seg_meta, seg_objects = read_segmentation_file(seg_path)
                np_points = read_pcd_data(pcd_path)

                # ---------------------------------
                # Get Mask as label
                # ---------------------------------
                height, width = seg_meta['size']['height'], seg_meta['size']['width']
                label_mask = np.zeros((height, width), dtype=np.uint8)

                for obj in seg_objects:
                    _label_mask = np.zeros((height, width), dtype=int)
                    class_ = obj['class']
                    annotation = obj['annotation'][0][0]

                    if reverse_cls_names is None:
                        if class_ not in class_names:
                            class_names[class_] = len(class_names)

                    for point in annotation:
                        x, y = point['x'], point['y']
                        _label_mask[y, x] = 1

                    max_val = -np.inf
                    for val in mask_values:
                        condition = np.where(mask_data == val, 1, 0)
                        sum_val = np.sum(condition * _label_mask)
                        if max_val < sum_val:
                            max_val = val
                    label_mask[mask_data == max_val] = class_names[class_]
                cv2.imwrite(os.path.join(output_label, "{}.png".format(fn)), label_mask)

                # ---------------------------------
                # Get Depth Image (HHA) as ModalX
                # ---------------------------------
                camera_intrinsic = read_camera_intrinsic_file(os.path.join(calib_dir, 'camera_intrinsic.txt'))
                camera_extrinsic = read_camera_extrinsic_file(os.path.join(calib_dir, 'camera_extrinsic.txt'))
                calib = (camera_intrinsic, camera_extrinsic)
                l2c = LiDAR2CameraKITTI(width=width, height=height, calib=calib)
                depthmap, _ = l2c.get_projected_image(None, np_points)
                depthmap = depthmap.astype(np.uint8) / 10000
                camera_matrix = get_camera_param(camera_intrinsic[0], 'color')
                hha = getHHA(camera_matrix, depthmap, depthmap)
                cv2.imwrite(os.path.join(output_modal, "{}.png".format(fn)), hha)

                # ---------------------------------
                # copy de-identification image as RGB Data
                # ---------------------------------
                shutil.copy2(de_id_path,
                             os.path.join(output_rgb, "{}.png".format(fn)))
    print()
    print(class_names)

    if config.save_class_name:
        with open(config.class_name_path, 'w') as f:
            f.write('\n'.join(list(class_names.keys())))

        assert predefined_class_names == class_names, "predefined class name과 처리된 class name이 다름! 확인 요망"


if __name__ == '__main__':
    cfg = define_args()
    main(cfg)
