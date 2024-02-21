import numpy as np

from const.constants import coco91_to_coco80, yolo_classes


def verify_yolo_labels(labels, num_cls=yolo_classes):
    '''
        Obtained from ultralytics.data.utils.verify_image_label()
    '''

    lb = np.array(labels, dtype=np.float32)
    nl = len(lb)

    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
    points = lb[:, 1:]

    assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
    assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

    # All labels
    max_cls = lb[:, 0].max()  # max label count
    assert max_cls <= num_cls, (
        f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
        f"Possible class labels are 0-{num_cls - 1}"
    )
    _, i = np.unique(lb, axis=0, return_index=True)
    if len(i) < nl:  # duplicate row check
        lb = lb[i]  # remove duplicates

    lb = lb[:, :5]
    return lb


def save_yolo_txt_labels(yolo_bboxes: np.ndarray, labels_file: str):
    with open(labels_file, "w+") as f:
        for box in yolo_bboxes:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

def read_yolo_txt_labels(labels_file):
    try:
        with open(labels_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)

        nl = len(lb)
        assert nl > 0

        lb = lb[:, :5]
        return lb
    except:
        return None


def get_yolo_bbox(labels_file, groundtruth, image_id, height0, width0) -> np.ndarray:
    '''
        Read YOLO labels txt files.
        If YOLO labels files don't exist, convert COCO annotation format to YOLO txt format:
        - COCO label format (JSON):
            annotation{
                "id": int,
                "image_id": int, 
                "category_id": int, (91 classes)
                "bbox": [x, y, width, height],
                ...
            }
        - YOLO label format (txt): class (80 classes) x_center y_center width height (normalized xywh format (from 0 to 1))
    '''
    yolo_bboxes = read_yolo_txt_labels(labels_file)
    if yolo_bboxes is None:
        yolo_bboxes = []
        for anno in groundtruth["annotations"]:
            if anno["image_id"] == image_id:
                yolo_bbox = np.zeros((5))
                coco_bbox = anno["bbox"].copy()

                # get class
                yolo_bbox[0] = coco91_to_coco80[int(anno["category_id"])]

                # get x_center y_center width height
                yolo_bbox[1] = coco_bbox[0] + coco_bbox[2]/2
                yolo_bbox[2] = coco_bbox[1] + coco_bbox[3]/2
                yolo_bbox[3] = coco_bbox[2]
                yolo_bbox[4] = coco_bbox[3]

                # normalize
                yolo_bbox[1] /= width0
                yolo_bbox[2] /= height0
                yolo_bbox[3] /= width0
                yolo_bbox[4] /= height0

                yolo_bboxes.append(yolo_bbox)
        save_yolo_txt_labels(yolo_bboxes, labels_file)
                
    try:
        return verify_yolo_labels(yolo_bboxes)
    except:
        yolo_bboxes = np.zeros((0, 5))
        print(f"Error loading labels for: {labels_file}")
        save_yolo_txt_labels(yolo_bboxes, labels_file)
        return np.zeros((0, 5))


def scale_bboxes(bboxes: np.ndarray, source_size: tuple, dest_size: tuple):
    bboxes = np.array(bboxes)
    if len(bboxes.shape) == 1:
        bboxes = bboxes[np.newaxis, :]

    height_scale = dest_size[0] / source_size[0]
    width_scale = dest_size[1] / source_size[1]

    for bbox in bboxes:
        bbox[0] = bbox[0] * width_scale   # x_min
        bbox[1] = bbox[1] * height_scale  # y_min
        bbox[2] = bbox[2] * width_scale   # x_max
        bbox[3] = bbox[3] * height_scale  # y_max

    return bboxes

