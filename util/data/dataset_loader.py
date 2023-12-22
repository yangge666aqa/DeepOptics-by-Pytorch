from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from util.data.data_utils import safe_crop_to_bounding_box
import os
import h5py
import cv2
import random

DATASET_PATH = {
    "ICVL512-MAT": "./datasets/ICVL",
}


class ICVL_512_MAT_Dataset_map(Dataset):
    """使用map迭代方法难以完成每次返回一张裁剪好的图片"""
    def __init__(self, path, verbose=False):
        self._mat_path = [os.path.join(path, file) for file in os.listdir(path)]
        self.verbose = verbose

    def __getitem__(self, index):
        hyper = h5py.File(self._mat_path[index])['rad'][:].transpose() / 4095.0
        if self.verbose:
            print("Decoding ICVL MAT: <shape=", hyper.shape, ">@", "index: ", index)
        return hyper

    def __len__(self):
        return len(self._mat_path)


class ICVL_512_MAT_Dataset_iter(IterableDataset):
    """iter方法适用于每次返回一张裁剪好的图片，其能通过迭代器自由返回图片"""
    def __init__(self, path, verbose=True):
        super(ICVL_512_MAT_Dataset_iter).__init__()
        self.file = os.listdir(path)
        random.shuffle(self.file)
        self._mat_list = [os.path.join(path, file) for file in self.file]
        self.verbose = verbose

    @staticmethod
    def overlapped_patches_from_ICVL_512_MAT(_img):
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
        import h5py  # import io
        for index, _ in enumerate(self._mat_list):
            # 刚读入时为(31, 1392, 1300) --> 转置为(1392, 1300, 31)以便resize
            hyper = h5py.File(self._mat_list[index])['rad'][:].transpose(1, 2, 0) / 4095.0
            if self.verbose:
                print("Decoding ICVL MAT: <shape=", hyper.shape, ">@", "file_name:", self.file[index])
            # 用于数据增强，返回 len = 10 的列表，里面储存 (512, 512, 31) 的元素
            overlaps = self.overlapped_patches_from_ICVL_512_MAT(hyper)
            for overlap in overlaps:
                yield overlap
