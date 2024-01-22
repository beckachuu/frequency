"""
Adapted from https://github.com/Cartucho/mAP
"""

try:
    import cupy as np
except ImportError:
    import numpy as np

from const.constants import coco_91_classes
from utility.path_utils import get_filepaths_list, get_last_path_element


class _ImageDetection:
    def __init__(self, score, label, boxes, used=False):
        self.boxes = boxes
        self.label = label
        self.score = score
        self.used = used


def _voc_ap(rec, prec):
    """
     Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
             precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """

    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """

    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def _check_dicts_for_content_and_size(ground_truth_dict: dict, result_dict: dict):
    """

    Checks if the content and the size of the arrays adds up.
    Raises and exception if not, does nothing if everything is ok.

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :return:
    """
    if 'boxes' not in ground_truth_dict.keys():
        raise ValueError("ground_truth_dict expects the keys 'boxes' and 'labels'.")
    if 'labels' not in ground_truth_dict.keys():
        raise ValueError("ground_truth_dict expects the keys 'boxes' and 'labels'.")
    if 'boxes' not in result_dict.keys():
        raise ValueError("result_dict expects the keys 'boxes' and 'labels' and optionally 'scores'.")
    if 'labels' not in result_dict.keys():
        raise ValueError("result_dict expects the keys 'boxes' and 'labels' and optionally 'scores'.")

    if 'scores' not in result_dict.keys():
        result_dict['scores'] = [1] * len(result_dict['boxes'])

    if len(ground_truth_dict['boxes']) != len(ground_truth_dict['labels']):
        raise ValueError("The number of boxes and labels differ in the ground_truth_dict.")

    if not len(result_dict['boxes']) == len(result_dict['labels']) == len(result_dict['scores']):
        raise ValueError("The number of boxes, labels and scores differ in the result_dict.")


def load_gt_data(groundtruth: dict, image_id: int):
    '''
    COCO JSON format: {
        'image_id': ...,
        'bbox': [x, y, w, h],
        'category_id': ...
    }

    Format we need: {
        'boxes': [x, y, x, y],
        'labels': []
    }
    '''

    gt_data = {
        'boxes': [],
        'labels': []
    }

    for gt_dict in groundtruth:
        if gt_dict["image_id"] == image_id:
            bbox = gt_dict["bbox"].copy()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            gt_data['boxes'].append(bbox)
            gt_data['labels'].append(int(gt_dict["category_id"]))

    gt_data['boxes'] = np.array(gt_data['boxes'])
    # gt_data['labels'] = np.array(gt_data['labels'])

    return gt_data

def get_gt_bbox(groundtruth: dict, image_id: int):
    gt_data = []
    
    for gt_dict in groundtruth:
        if gt_dict["image_id"] == image_id:
            bbox = gt_dict["bbox"].copy()
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            gt_data.append(bbox)
            gt_data[-1].append(1)
            gt_data[-1].append(int(gt_dict["category_id"]))

    return np.array(gt_data)



def load_detect_data(detect_list):
    if len(detect_list) == 0:
        detect_data = {
            'boxes': [],
            'labels': [],
            'scores': []
        }
        return detect_data

    if not isinstance(detect_list, np.ndarray):
        detect_list = np.array(detect_list)

    detect_data = {
        'boxes': detect_list[:, 0:4],
        'labels': list(map(np.int32, detect_list[:, 5])),
        'scores': detect_list[:, 4]
    }
    return detect_data

def get_detect_bbox(detect_path):
    with open(detect_path, 'r') as file:
        detect_data = file.readlines()
        detect_data = [line.rstrip('\n') for line in detect_data]
        detect_data = [list(map(float, line.split())) for line in detect_data]
    return np.array(detect_data)



def calculate_tp_fp_for_class(gt_data: dict, detect_data: dict, iou_threshold: float, class_name: int):
    """
    mAP@[iou_threshold]

    :param ground_truth_dict: dict with {boxes:, labels:}
    e.g.
    {
    'boxes':
        [[60., 80., 66., 92.],
         [59., 94., 68., 97.],
         [70., 87., 81., 94.],
         [8., 34., 10., 36.]],

    'labels':
        [2, 2, 3, 4]}
    :param result_dict: dict with {boxes:, labels:, scores:}
    e.g.
    {
    'boxes':
        [[57., 87., 66., 94.],
         [58., 94., 68., 95.],
         [70., 88., 81., 93.],
         [10., 37., 17., 40.]],

    'labels':
        [2, 3, 3, 4],

    'scores':
        [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
    :param iou_threshold: minimum iou for which the detection counts as successful
    :return: mean average precision (mAP)
    """

    # checking if the variables have the correct keys
    _check_dicts_for_content_and_size(gt_data, detect_data)

    detections_with_certain_class = list()
    for idx in range(len(detect_data['labels'])):
        if detect_data['labels'][idx] == class_name:
            detections_with_certain_class.append(_ImageDetection(score=detect_data['scores'][idx],
                                                                    label=detect_data['labels'][idx],
                                                                    boxes=detect_data['boxes'][idx]))
    ground_truth_list = list()
    for idx in range(len(gt_data['labels'])):
        ground_truth_list.append(_ImageDetection(score=1,
                                                    label=gt_data['labels'][idx],
                                                    boxes=gt_data['boxes'][idx]))

    tp = [0] * len(detections_with_certain_class)
    fp = [0] * len(detections_with_certain_class)

    for i, elem in enumerate(detections_with_certain_class):
        ovmax = -1
        gt_match = -1

        bb = elem.boxes
        for j, elem in enumerate(ground_truth_list):
            if ground_truth_list[j].label == class_name:
                bbgt = ground_truth_list[j].boxes
                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = elem

        if ovmax >= iou_threshold:
            if not gt_match.used:
                # true positive
                tp[i] = 1
                gt_match.used = True
            else:
                # false positive (multiple detection)
                fp[i] = 1
        else:
            # false positive
            fp[i] = 1

    return tp, fp


def calculate_tp_fp_range_for_class(gt_data: dict, detect_data: dict, class_name: int, iou_begin: float = 0.5, iou_end: float = 0.95, iou_step: float = 0.05):
    iou_list = np.arange(iou_begin, iou_end + iou_step, iou_step)

    tp = [[] for _ in range(len(iou_list))]
    fp = tp[:]
    for i, iou in enumerate(iou_list):
        tp[i], fp[i] = calculate_tp_fp_for_class(gt_data, detect_data, iou, class_name)

    return tp, fp



def calculate_AP_for_class(tp, fp, gt_count):
    # compute precision/recall
    cumsum = 0
    for i, val in enumerate(fp):
        fp[i] += cumsum
        cumsum += val

    cumsum = 0
    for i, val in enumerate(tp):
        tp[i] += cumsum
        cumsum += val

    rec = tp[:]
    prec = tp[:]
    for i, val in enumerate(tp):
        rec[i] = float(tp[i]) / gt_count
        prec[i] = float(tp[i]) / (fp[i] + tp[i])

    average_precision, mean_recall, mean_precision = _voc_ap(rec[:], prec[:])
    return average_precision


def calculate_tp_fp_range_from_path(path, groundtruth, iou_list, classes = coco_91_classes):
    tp = [[[] for _ in range(len(iou_list))] for _ in range(len(classes))]
    fp = [[[] for _ in range(len(iou_list))] for _ in range(len(classes))]

    detect_files_paths = get_filepaths_list(path, ['txt'])
    for file_path in detect_files_paths:
        image_id = int(get_last_path_element(file_path).split('.')[0])
        gt_data = load_gt_data(groundtruth, image_id)
        detect_data = load_detect_data(get_detect_bbox(file_path))

        # accumulate_tp_fp for each class
        for class_ind in gt_data['labels']:
            tp_range, fp_range = calculate_tp_fp_range_for_class(gt_data, detect_data, class_ind)
            for i in range(len(iou_list)):
                tp[class_ind-1][i] += tp_range[i]
                fp[class_ind-1][i] += fp_range[i]
    return tp, fp
