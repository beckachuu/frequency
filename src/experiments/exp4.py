import csv
import os
import time
from logging import Logger
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from dataset import YoloHyperparameters, TrainDataset
from model.freq_filter import FrequencyDomainFilter
from utility.format_utils import (preprocess_image_from_url_to_torch_input,
                                  resize_auto_interpolation)
from utility.path_utils import (create_path_if_not_exists, get_filepaths_list,
                                get_last_path_element)
from utility.train_util import find_last_checkpoint


class FrequencyExp():
    def __init__(self, logger:Logger, input_dir, image_extensions, batch_size, exp_dir: str, exp_values: list, 
                 force_exp: bool, plot_analyze: bool):
        self.logger = logger

        self.input_dir = input_dir
        self.image_extensions = image_extensions
        self.batch_size = batch_size
        self.exp_dir = exp_dir
        self.exp_values = exp_values
        self.force_exp = force_exp
        self.plot_analyze = plot_analyze

        # Training stuff
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoints = Path(self.exp_dir, f"checkpoints")
        create_path_if_not_exists(self.checkpoints)
        self.log_dir = Path(self.exp_dir, f"logs.csv")


    def run_experiment(self, train_dir, train_split, train_annos, val_dir, val_split, val_annos,
                       save_labels_dir, detect_model_type):

        # Exp values
        epochs = int(self.exp_values[0])
        filter_size = (int(self.exp_values[1]), int(self.exp_values[1]))
        patience = int(self.exp_values[2])
        save_freq = int(self.exp_values[3])
        init_lr = self.exp_values[4]
        lr_patience = int(self.exp_values[5])
        min_lr = self.exp_values[6]
        box_gain = self.exp_values[7]
        cls_gain = self.exp_values[8]
        dfl_gain = self.exp_values[9]

        patience_counter = 0
        best_val_loss = float("inf")

        if self.force_exp:
            initial_epoch = 0
        else:
            initial_epoch = find_last_checkpoint(save_dir=self.checkpoints)

        # Data
        train_loader, val_loader = self.load_dataset(train_dir, train_split, train_annos, val_dir, val_split, val_annos,
                                                     self.image_extensions, save_labels_dir, self.batch_size)

        # Load models
        filter_model = self.load_filter_model(initial_epoch, filter_size)
        detect_model = self.load_detect_model(detect_model_type, box_gain, cls_gain, dfl_gain)

        # Loss
        criterion = v8DetectionLoss(detect_model)

        # Optimizer, learning rate scheduler and grad scaler
        optimizer = torch.optim.Adam(filter_model.parameters(), lr=init_lr)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=lr_patience, min_lr=min_lr, verbose=True)
        scaler = GradScaler()

        self.write_results("epoch", "train box_loss", "train cls_loss", "train dfl_loss", 
                           "val box_loss", "val cls_loss", "val dfl_loss", 
                           "train time (sec)", "val time (sec)")

        for epoch in range(initial_epoch+1, epochs + 1):
            train_start_time = time.time()
            losses = {"train": {"box": 0, "cls": 0, "dfl": 0}, 
                    "val": {"box": 0, "cls": 0, "dfl": 0}}

            for batch_ind, batch in enumerate(train_loader):
                filter_model.train()
                optimizer.zero_grad()

                input_img, gt_bbox = batch["img"].to(self.device), batch

                filtered = filter_model(input_img)
                preds = detect_model(filtered)
                loss, loss_items = criterion(preds, gt_bbox)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                box, cls, dfl = [float(x) for x in loss_items]
                losses["train"]["box"] = (losses["train"]["box"] * batch_ind + box) / (batch_ind + 1)
                losses["train"]["cls"] = (losses["train"]["cls"] * batch_ind + cls) / (batch_ind + 1)
                losses["train"]["dfl"] = (losses["train"]["dfl"] * batch_ind + dfl) / (batch_ind + 1)
                self.logger.info(f"[Epoch {epoch}][{batch_ind}/{len(train_loader)}]: " + 
                                 f"box_loss = {box:.3f}, cls_loss = {cls:.3f}, dfl_loss = {dfl:.3f}")
            
            self.logger.info(f"[TRAIN LOSS - epoch {epoch}]: box_loss = {losses['train']['box']:.3f}, " +
                             f"cls_loss = {losses['train']['cls']:.3f}, dfl_loss = {losses['train']['dfl']:.3f}")

            train_time = time.time() - train_start_time

            # Validation
            val_start_time = time.time()
            with torch.no_grad():
                filter_model.eval()
                for batch_ind, batch in enumerate(val_loader):
                    input_img, gt_bbox = batch["img"].to(self.device), batch

                    filtered = filter_model(input_img)
                    preds = detect_model(filtered)
                    val_loss, loss_items = criterion(preds, gt_bbox)

                    box, cls, dfl = [float(x) for x in loss_items]
                    losses["val"]["box"] = (losses["val"]["box"] * batch_ind + box) / (batch_ind + 1)
                    losses["val"]["cls"] = (losses["val"]["cls"] * batch_ind + cls) / (batch_ind + 1)
                    losses["val"]["dfl"] = (losses["val"]["dfl"] * batch_ind + dfl) / (batch_ind + 1)
            
            scheduler.step(val_loss)

            val_time = time.time() - val_start_time
            self.logger.info(f"[VAL LOSS - epoch {epoch}]: box_loss = {losses['val']['box']:.3f}, " +
                             f"cls_loss = {losses['val']['cls']:.3f}, dfl_loss = {losses['val']['dfl']:.3f}")

            self.write_results(epoch, losses["train"]["box"], losses["train"]["cls"], losses["train"]["dfl"],
                               losses["val"]["box"], losses["val"]["cls"], losses["val"]["dfl"],
                               train_time, val_time)
            patience_counter, best_val_loss = self.save_filter_model(filter_model, epoch, patience_counter, save_freq, 
                                   losses["val"]["box"]+losses["val"]["cls"]+losses["val"]["dfl"], best_val_loss)

            if patience_counter > patience and patience >= 0:
                print("Early stopping.")
                break
        
        self.test_filter_model(filter_model)



    def load_dataset(self, train_dir, train_split, train_annos, val_dir, val_split, val_annos,
                     image_extensions, save_labels_dir, batch_size):
        if train_split == -1:
            train_split = 1
        if val_split == -1:
            val_split = 1

        if train_dir == val_dir and train_split + val_split > 1:
            self.logger.warn(f'Train and val dataset are overlapping (train_split = {train_split}, val_split = {val_split}) => ' + 
                             f'val_split is set to {1.0 - train_split}')
            val_split = 1.0 - train_split
        if save_labels_dir == '':
            save_labels_dir = os.path.join(os.path.dirname(train_dir), 'labels')
            self.logger.warn(f'labels_dir is not set. Creating new directory in: {save_labels_dir}')
        create_path_if_not_exists(save_labels_dir)

        train_dataset = TrainDataset(train_dir, image_extensions, train_split, True, train_annos, save_labels_dir)
        self.logger.info(f'Loaded train data from: {train_dir}. Images count: {len(train_dataset)}')
        train_loader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True, collate_fn=TrainDataset.collate_fn)
        val_dataset = TrainDataset(val_dir, image_extensions, val_split, False, val_annos, save_labels_dir)
        self.logger.info(f'Loaded validation data from: {val_dir}. Images count: {len(val_dataset)}')
        val_loader = DataLoader(dataset=val_dataset, num_workers=2, batch_size=batch_size, shuffle=True, collate_fn=TrainDataset.collate_fn)
        return train_loader, val_loader
    

    def load_filter_model(self, initial_epoch, filter_size):
        filter_model = FrequencyDomainFilter(filter_size)
        filter_model.to(self.device)
        
        if initial_epoch > 0:
            self.logger.info(f"Resuming by loading epoch {initial_epoch}")
            filter_model.load_state_dict(torch.load(Path(self.checkpoints, f"epoch{initial_epoch}.pth"), map_location=self.device))
        return filter_model

    def load_detect_model(self, model_type, box_gain, cls_gain, dfl_gain):
        version = model_type[0]
        if version != "8":
            self.logger.error("Only YOLOv8 versions can be used for this experiments.")

        # Download model weights
        detector = YOLO(f"yolov{model_type}.pt") # TODO: extract model download function only from this

        # Create model
        detector = DetectionModel(cfg=f"yolo_cfg/v{version}/yolov{model_type}.yaml", nc=80, verbose=False)
        detector.args = YoloHyperparameters(box_gain, cls_gain, dfl_gain)
        detector.eval()
        detector.to(self.device)

        detector.load(torch.load(f'yolov{model_type}.pt', map_location=self.device))

        # not training this detect model -> no grads required
        for param in detector.parameters():
            param.requires_grad = False

        return detector



    def save_filter_model(self, filter_model, epoch, patience_counter, save_freq, total_val_loss, best_val_loss):
        # save best trained model
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            patience_counter = 0
            torch.save(filter_model.state_dict(), Path(self.checkpoints, f'val_best.pth'))
        else:
            patience_counter += 1

        # save model frequently
        if save_freq > 0 and epoch % save_freq == 0:
            filter_model.save_filter_img(Path(self.exp_dir, f'filter_epoch{epoch}.png'))
            torch.save(filter_model.state_dict(), Path(self.checkpoints, 'epoch%d.pth' % (epoch)))

        return patience_counter, best_val_loss


    def write_results(self, epoch, train_box_loss: float, train_cls_loss: float, train_dfl_loss: float, 
                      val_box_loss: float, val_cls_loss: float, val_dfl_loss: float, train_time, val_time):
        f = open(self.log_dir, "a")
        writer = csv.writer(f)
        if isinstance(epoch, str):
            writer.writerow([epoch, train_box_loss, train_cls_loss, train_dfl_loss, 
                             val_box_loss, val_cls_loss, val_dfl_loss, train_time, val_time])
        else:
            writer.writerow([epoch, train_box_loss, train_cls_loss, train_dfl_loss,
                             val_box_loss, val_cls_loss, val_dfl_loss, train_time, val_time])
        f.close()


    def test_filter_model(self, filter_model):
        self.logger.info('Generating test outputs...')

        images_files = get_filepaths_list(self.input_dir, self.image_extensions)
        pth_files = get_filepaths_list(self.checkpoints, ['pth'])

        for pth_file in pth_files:
            self.logger.info(f'Testing with weights file: {pth_file}')

            filter_model.load_state_dict(torch.load(pth_file, map_location=self.device))
            filter_model.eval()

            save_folder = get_last_path_element(pth_file).split('.')[0]
            save_dir = Path(self.exp_dir, save_folder)
            create_path_if_not_exists(save_dir)
        
            for image_file in images_files:
                image, height0, width0 = preprocess_image_from_url_to_torch_input(image_file)
                image = image.unsqueeze(0).to(self.device)
                image_name = get_last_path_element(image_file)

                output = filter_model(image)[0]
                output = torch.clamp(output, 0, 1)
                output = resize_auto_interpolation(output, height0, width0)
                plt.imsave(Path(save_dir, image_name), output)


