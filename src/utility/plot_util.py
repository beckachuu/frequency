# ================================================================
#
#   File name   : plot_util.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : additional yolov3 and yolov4 functions
#   Modified by : beckachuu
#
# ================================================================

import colorsys
import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from const.constants import coco_91_classes

plt.ioff()


def draw_bbox(image, bboxes, classes=coco_91_classes, show_label=True, show_confidence=True,
              Text_colors=(0, 0, 0), rectangle_colors='', tracking=False):
    image = image.copy()
    num_classes = len(classes)
    image_h, image_w, _ = image.shape if len(image.shape) == 3 else image.shape[1:]
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes + 1)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""
            if tracking: score_str = " " + str(score)
            try:
                label = "{}".format(classes[class_ind]) + score_str
            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                          thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image


def plot_images_with_bbox(images: list, bboxes: list, titles: list, fig_save_path: str, classes=coco_91_classes):

    drawed_images = [draw_bbox(image_i, bboxes=bboxes_i, classes=classes) for image_i, bboxes_i in zip(images, bboxes)]
    images = drawed_images

    plot_images(images, titles, fig_save_path)



def plot_images(images: list, titles: list, fig_save_path: str, cols: int = None, dpi=100):
    n = len(images)
    cols = cols if cols else math.ceil(math.sqrt(n))
    rows = math.ceil(len(images) / cols)

    # Assume the images all have the same size
    height, width = images[0].shape

    # Calculate the size of the figure in inches
    fig_width = width * cols / dpi
    fig_height = height * rows / dpi

    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)

    for i, ax in enumerate(axs.flat):
        ax.axis('off')
        if i < n:
            ax.imshow(images[i])
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=dpi)
    plt.close(fig)


