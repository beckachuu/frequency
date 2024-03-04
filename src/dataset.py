from pathlib import Path

import torch
import torch.utils.data as udata

try:
    import cupy as np
except ImportError:
    import numpy as np

from utility.bbox_util import get_yolo_bbox
from utility.format_utils import (preprocess_image_from_url_to_255HWC,
                                  preprocess_image_from_url_to_torch_input)
from utility.path_utils import (get_filepaths_list, get_last_path_element, get_last_element_name,
                                read_dict_from_json)


class ImageDataset(udata.Dataset):
    def __init__(self, input_dir, image_extensions):
        super(ImageDataset, self).__init__()

        self.input_dir = input_dir
        self.filepaths = get_filepaths_list(input_dir, image_extensions)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int):
        img, h, w = preprocess_image_from_url_to_255HWC(self.filepaths[index])
        image_name = get_last_path_element(self.filepaths[index])

        return img, image_name, (h, w)


class PathDataset(udata.Dataset):
    def __init__(self, input_dir, image_extensions):
        super(PathDataset, self).__init__()

        self.input_dir = input_dir
        self.filepaths = get_filepaths_list(input_dir, image_extensions)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index: int):
        return self.filepaths[index]

class YoloHyperparameters:
    def __init__(self, box=7.5, cls=0.5, dfl=1.5):
        self.box = box # (float) box loss gain
        self.cls = cls # (float) cls loss gain (scale with pixels)
        self.dfl = dfl # (float) dfl loss gain

class TrainDataset(udata.Dataset):
    def __init__(self, input_dir, img_extensions, split, is_train_set, gt_json_dir, labels_dir):
        super(TrainDataset, self).__init__()

        self.gt_data = read_dict_from_json(gt_json_dir)
        self.labels_dir = labels_dir

        self.image_paths = get_filepaths_list(input_dir, img_extensions)
        if split >= 0:
            count = int(split * len(self.image_paths))
            if is_train_set:
                self.image_paths = self.image_paths[:count]
            else:
                self.image_paths = self.image_paths[-count:]

        self.image_paths = self.filter_empty_label()

    def filter_empty_label(self):
        filtered_ids = {}
        for data in self.gt_data["annotations"]:
            filtered_ids[data["image_id"]] = None

        image_paths = []
        for path in self.image_paths:
            image_id = int(get_last_element_name(path))
            if image_id in filtered_ids:
                image_paths.append(path)

        return image_paths


    def __len__(self):
        return len(self.image_paths)

    
    def get_labels(self, labels: np.ndarray) -> dict:
        '''
            Obtained from ultralytics.data.datset.YOLODataset.get_labels()
            Returns dictionary of labels for YOLO training.
        '''

        labels_dict = dict(cls=torch.from_numpy(labels[:, 0:1]),  # n, 1
                           bboxes=torch.from_numpy(labels[:, 1:]),  # n, 4
                           normalized=True, bbox_format="xywh")

        return labels_dict


    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        img, height0, width0 = preprocess_image_from_url_to_torch_input(image_path)

        image_name = get_last_element_name(image_path)
        image_id = int(image_name)

        labels_file = Path(self.labels_dir, f'{image_name}.txt')
        gt_bbox = get_yolo_bbox(labels_file, self.gt_data, image_id, height0, width0)
        labels = self.get_labels(gt_bbox)

        item = labels
        item["img"] = img
        item["batch_idx"] = torch.zeros(len(labels["cls"]))
        
        return item
    
    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["bboxes", "cls"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


