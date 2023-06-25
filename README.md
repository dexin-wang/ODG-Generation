# ODG-Generation


1. Generate data sets: Load models, randomly build scenes, render image data, compute grab labels, save data, data enhancement.

operating steps:

(1) Download `obj_models.zip` from `https://drive.google.com/file/d/1QLAubfa6nCkmRGeCZuy44lFfwKWobxVa/view?usp=drive_link` and unzip.

(2) Open `scripts\dataset\render_grasp_data.py`, modify `model_path` to path of obj_models, and modify `save_path` to dataset path.

(3) run `python scripts\dataset\render_grasp_data.py`.

(4) The dataset is divided into two parts: training set and test set.

(5) Open `scripts\dataset\create_dataset.py`, modify `src_path` to path of training set or test set, and modify `dst_path` for training network.

(6)  run `python scripts\dataset\create_dataset.py`.



2. Visualize mask and bbox labels ：

(1) Open `scripts\dataset\visual_mask_bbox.py`, modify `dataset_path` to `dataset path`.

(2) run `python scripts\dataset\visual_mask_bbox.py`



3. benchmark

(1) Download `backbones.zip` from `` and place files in `grasp_methods\ckpt`.

(2) Open `scripts\grasp_experiments\test_acc.py`, modify `obj_model_path` to path of obj_models.

(3) run `scripts\grasp_experiments\test_acc.py`

(4) Open `scripts\grasp_experiments\test_AP.py`, modify `obj_model_path` to path of obj_models.

(5) run `scripts\grasp_experiments\test_AP.py`


4. Future work

Translate comments into English.

