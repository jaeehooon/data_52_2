# Prepare Dataset

1. Dataset Structure
   
   ```
   52_2
   ├── DOGU
   │   ├── CP
   │   │   ├── a
   │   │   │   └── cloudy
   │   │   │       └── 220719
   │   │   │           └── 17-19
   │   │   │               ├── calibration
   │   │   │               │   ├── camera_intrinsic.txt
   │   │   │               │   └── camera_extrinsic.txt
   │   │   │               ├── refine
   │   │   │               │   ├── camera
   │   │   │               │   │   ├── 01
   │   │   │               │   │   ├── 02
   │   │   │               │   │   ├── ...
   │   │   │               │   └── pcd
   │   │   │               │       ├── 01
   │   │   │               │       ├── 02
   │   │   │               │       ├── ...
   │   │   │               └── seg
   │   │   │                   ├── de-identification
   │   │   │                   ├── mask
   │   │   │                   └── segmentation
   │   │   ├── b
   │   │   │   └── cloudy
   │   │   │       └── 220719
   │   │   │           └── 16-17
   │   │   │               ├── calibration
   │   │   │               │   ├── camera_intrinsic.txt
   │   │   │               │   └── camera_extrinsic.txt
   │   │   │               ├── refine
   │   │   │               │   ├── camera
   │   │   │               │   │   ├── 01
   │   │   │               │   │   ├── 02
   │   │   │               │   │   ├── ...
   │   │   │               ├── ...
   │
   ├── RASTECH
   │   ├── ...
   │   └──
   ├── ATECH
   │   ├── ...
   │   └── 
   │ ...
   ```
   
2. run
   ```
   python main.py --data_root [DATA ROOT] --output_path [FINAL DATA ROOT]
   ```
   
3. Results
   ```
   52_2
   ├── final
   │   ├── RGBFolder
   │   ├── ModalXFolder
   │   ├── LabelFolder
   │   ├── class_names.txt
   │   ├── train.txt (You should run 'train_test_split.ipynb')
   │   └── test.txt (You should run 'train_test_split.ipynb')
   ```