import csv
import os
import time
from logging import Logger
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from dataset import Hyperparameters, TrainDataset
from model.freq_filter import FrequencyDomainFilter
from utility.path_utils import create_path_if_not_exists
from utility.train_util import find_last_checkpoint


class FrequencyExp():
    def __init__(self, logger:Logger, exp_dir: str, exp_values: list, force_exp: bool, plot_analyze: bool):
        self.logger = logger

        self.exp_dir = exp_dir
        self.checkpoints = Path(self.exp_dir, f"checkpoints")
        create_path_if_not_exists(self.checkpoints)
        self.log_dir = Path(self.exp_dir, f"logs.csv")

        self.exp_values = exp_values

        self.force_exp = force_exp
        self.plot_analyze = plot_analyze

        # Training stuff
        self.epochs = self.exp_values[0]
        self.filter_size = (self.exp_values[1], self.exp_values[1])
        self.patience_counter = 0
        self.patience = self.exp_values[2]
        self.lr = 1e-3
        self.save_freq = self.exp_values[3]

        self.filter_model = None


    def run_experiment(self, train_dir, train_split, train_annos, val_dir, val_split, val_annos,
                       image_extensions, save_labels_dir, batch_size, detect_model_type):
        # square_h, square_w = images[0].shape[:2]
        # big_img = np.zeros((square_h * 3, square_w * 3))
        # big_h, big_w = big_img.shape[:2]

        train_loader, val_loader = self.load_dataset(train_dir, train_split, train_annos, val_dir, val_split, val_annos,
                                                     image_extensions, save_labels_dir, batch_size)

        if self.force_exp:
            initial_epoch = 0
        else:
            initial_epoch = find_last_checkpoint(save_dir=self.checkpoints)
        self.filter_model = self.load_filter_model(initial_epoch)
        
        detect_model = self.load_detect_model(detect_model_type)

        criterion = v8DetectionLoss(detect_model)
        # criterion.cuda()

        # Optimizer
        optimizer = torch.optim.Adam(self.filter_model.parameters(), lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)
        scaler = GradScaler()

        best_val_loss = float("inf")

        self.write_results("epoch", "train_loss", "val_loss", "time (sec)")

        for epoch in range(initial_epoch, self.epochs + 1):
            current_time = time.time()
            total_train_loss = 0
            total_val_loss = 0

            for batch_ind, batch in enumerate(train_loader):
                self.filter_model.train()
                self.filter_model.zero_grad()
                optimizer.zero_grad()

                input_img, gt_bbox = batch["img"].cuda(), batch

                # with autocast():
                filtered = self.filter_model(input_img)
                preds = detect_model(filtered)

                loss, loss_items = criterion(preds, gt_bbox)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_sum = loss_items.sum()
                total_train_loss += (total_train_loss * batch_ind + loss_sum) / (batch_ind + 1)
                self.logger.info(f"[Epoch {epoch}][{batch_ind}/{len(train_loader)}] train_loss: {loss_sum:.2f}")

            scheduler.step()

            # Validation
            with torch.no_grad():
                self.filter_model.eval()
                for batch_ind, batch in enumerate(val_loader):
                    input_img, gt_bbox = batch["img"].cuda(), batch

                    filtered = self.filter_model(input_img)
                    preds = detect_model(filtered)
                    _, loss_items = criterion(preds, gt_bbox)
                    total_val_loss += (total_val_loss * batch_ind + loss_items.sum()) / (batch_ind + 1)

            self.write_results(epoch, round(float(total_train_loss), 2), round(float(total_val_loss), 2), round(current_time, 2))
            current_time = time.time()

            self.logger.info(f"[Validation] val_loss: {total_val_loss}")

            self.save_model(self.filter_model, epoch, self.save_freq, total_val_loss, best_val_loss)

            if self.patience_counter > self.patience:
                print("Early stopping.")
                break


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
    

    def load_filter_model(self, initial_epoch):
        filter_model = FrequencyDomainFilter(filter_size=(self.filter_size[0], self.filter_size[1]))
        filter_model.cuda()
        if initial_epoch > 0:
            self.logger.info("Resuming by loading epoch %d" % initial_epoch)
            filter_model.load_state_dict(torch.load(Path(self.checkpoints, "net_epoch%d.pth" % initial_epoch)))
        return filter_model

    def load_detect_model(self, model_type):
        version = model_type[0]

        detector = YOLO(f"yolov{model_type}.pt") # TODO: extract model download function only from this
        detector = DetectionModel(cfg=f"yolo_cfg/v{version}/yolov{model_type}.yaml", nc=80, verbose=False)
        detector.args = Hyperparameters(7.5, 0.5, 1.5)
        detector.eval()
        detector.cuda()

        if os.path.exists(f'yolov{model_type}.pt'):
            detector.load(torch.load(f'yolov{model_type}.pt'))
        else:
            detector.load(torch.load(f'yolov{model_type}u.pt'))

        # detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        for param in detector.parameters():
            param.requires_grad = False

        return detector


    def save_model(self, model, epoch, save_freq, total_val_loss, best_val_loss):
        # save best trained model
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            self.patience_counter = 0
            torch.save(model.state_dict(), Path(self.checkpoints, 'best_val.pth'))
        else:
            self.patience_counter += 1

        # save model frequently
        if save_freq > 0 and epoch % save_freq == 0:
            self.filter_model.save_filter_img(Path(self.exp_dir, f'filter_epoch{epoch}.png'))
            torch.save(model.state_dict(), Path(self.checkpoints, 'net_epoch%d.pth' % (epoch)))


    def write_results(self, epoch, epoch_train_loss, epoch_val_loss, current_time):
        f = open(self.log_dir, "a")
        writer = csv.writer(f)
        if isinstance(current_time, str):
            writer.writerow([epoch, epoch_train_loss, epoch_val_loss, current_time])
        else:
            writer.writerow([epoch, epoch_train_loss, epoch_val_loss, time.time() - current_time])
        f.close()

