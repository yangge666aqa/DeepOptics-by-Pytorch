from torch.utils.data import Dataset
from torch.utils.data import IterableDataset, get_worker_info
from deepoptics.data.data_utils import safe_crop_to_bounding_box
import os
import h5py
import cv2
import random
from pathlib import Path


def mat_files(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {path}')
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.mat')
    if not files:
        raise ValueError(f'No MAT files found in {path}')
    return files


def read_cube(path):
    with h5py.File(path, 'r') as handle:
        cube = handle['rad'][:]
    if cube.ndim != 3 or cube.shape[0] != 31:
        raise ValueError(f'Expected rad with shape (31, H, W), got {cube.shape}: {path}')
    return cube.transpose(1, 2, 0).astype('float32') / 4095.0


DATASET_PATH = {
    "ICVL512-MAT": "./datasets/ICVL",
}


class ICVL_512_MAT_Dataset_map(Dataset):
    """使用map迭代方法难以完成每次返回一张裁剪好的图片"""
    def __init__(self, path, verbose=False):
        super().__init__()
        self._mat_path = mat_files(path)
        self.verbose = verbose

    def __getitem__(self, index):
        hyper = read_cube(self._mat_path[index])
        if self.verbose:
            print("Decoding ICVL MAT: <shape=", hyper.shape, ">@", "index: ", index)
        return hyper

    def __len__(self):
        return len(self._mat_path)


class ICVL_512_MAT_Dataset_iter(IterableDataset):
    """iter方法适用于每次返回一张裁剪好的图片，其能通过迭代器自由返回图片"""
    def __init__(self, path, verbose=True, shuffle=True):
        super().__init__()
        self._mat_list = mat_files(path)
        if shuffle:
            random.shuffle(self._mat_list)
        self.file = [p.name for p in self._mat_list]
        self.verbose = verbose

    @staticmethod
    def overlapped_patches_from_ICVL_512_MAT(_img):
        if min(_img.shape[:2]) < 512:
            raise ValueError('ICVL images must be at least 512 x 512 pixels.')
        overlapped_operation_list = [cv2.resize(_img, (512, 512), interpolation=cv2.INTER_LINEAR)]
        # print(cv2.resize(_img, (512, 512), interpolation=cv2.INTER_LINEAR).shape)
        third_height = 464
        third_width = 434
        for i in range(0, 1392, third_height):
            for j in range(0, 1300, third_width):
                # 防止crop到图片外部区域
                overlapped_operation_list.append(safe_crop_to_bounding_box(_img, i, j, 512, 512))
        return overlapped_operation_list

    def __iter__(self):
        worker = get_worker_info()
        paths = self._mat_list if worker is None else self._mat_list[worker.id::worker.num_workers]
        for path in paths:
            # 刚读入时为(31, 1392, 1300) --> 转置为(1392, 1300, 31)以便resize
            hyper = read_cube(path)
            if self.verbose:
                print("Decoding ICVL MAT: <shape=", hyper.shape, ">@", "file_name:", path.name)
            # 用于数据增强，返回 len = 10 的列表，里面储存 (512, 512, 31) 的元素
            overlaps = self.overlapped_patches_from_ICVL_512_MAT(hyper)
            for overlap in overlaps:
                yield overlap
