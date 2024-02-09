# This is part of the tutorial materials in the UCL Module MPHY0041: Machine Learning in Medical Imaging
import os
import numpy as np

from openpyxl import Workbook

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from config import TRAIN, TRANSFORM, DATA, POOL

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
folder_name = 'data'
RESULT_PATH = './result'

## network class
class UNet(torch.nn.Module):

    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(UNet, self).__init__()

        n_feat = init_n_feat
        self.encoder1 = UNet._block(ch_in, n_feat)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(n_feat, n_feat*2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(n_feat*2, n_feat*4)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(n_feat*4, n_feat*8)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(n_feat*8, n_feat*16) # 8 for less pool

        self.upconv4 = torch.nn.ConvTranspose2d(n_feat*16, n_feat*8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((n_feat*8)*2, n_feat*8)
        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((n_feat*4)*2, n_feat*4)
        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((n_feat*2)*2, n_feat*2)
        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(n_feat*2, n_feat)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)

    def forward(self, x):
        if POOL == 4:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))

            bottleneck = self.bottleneck(self.pool4(enc4))

            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            return torch.sigmoid(self.conv(dec1))
        if POOL == 3:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))

            bottleneck = self.bottleneck(enc4)

            dec4 = torch.cat((bottleneck, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            return torch.sigmoid(self.conv(dec1))
        if POOL == 2:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(enc3)

            bottleneck = self.bottleneck(enc4)

            dec4 = torch.cat((bottleneck, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = torch.cat((dec4, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            return torch.sigmoid(self.conv(dec1))
        if POOL == 1:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(enc2)
            enc4 = self.encoder4(enc3)

            bottleneck = self.bottleneck(enc4)

            dec4 = torch.cat((bottleneck, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = torch.cat((dec4, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = torch.cat((dec3, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            return torch.sigmoid(self.conv(dec1))
        if POOL == 0:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(enc1)
            enc3 = self.encoder3(enc2)
            enc4 = self.encoder4(enc3)

            bottleneck = self.bottleneck(enc4)

            dec4 = torch.cat((bottleneck, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = torch.cat((dec4, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = torch.cat((dec3, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = torch.cat((dec2, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))

def calculate_2d_metrics(gt_slice, pred_slice):
    intersection = np.logical_and(gt_slice, pred_slice)
    union = np.logical_or(gt_slice, pred_slice)
   
    iou = np.sum(intersection) / np.sum(union)
    dice_coefficient = 2.0 * np.sum(intersection) / (np.sum(gt_slice) + np.sum(pred_slice))

    return iou, dice_coefficient

def calculate_average_metrics(gt_volume, pred_volume):
    num_slices = gt_volume.shape[0]
    iou_scores = []
    dice_scores = []

    for slice_idx in range(num_slices):
        gt_slice = gt_volume[slice_idx]
        skip=np.sum(gt_slice)
        if skip>0:
            pred_slice = pred_volume[slice_idx]
            iou, dice_coefficient = calculate_2d_metrics(gt_slice, pred_slice)
            iou_scores.append(iou)
            dice_scores.append(dice_coefficient)

    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)

    return avg_iou, avg_dice

## loss function
def loss_2Ddice(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2
    denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps
    return torch.mean(1. - (numerator / denominator))

def update_excel(epoch, epoch_train_loss, epoch_val_loss, val_iou, model_saved, worksheet):
    # Update values in respective rows and columns
    worksheet.cell(row=epoch+1, column=1, value=f'Epoch {epoch}')
    worksheet.cell(row=epoch+1, column=2, value=epoch_train_loss)
    worksheet.cell(row=epoch+1, column=3, value=epoch_val_loss)
    worksheet.cell(row=epoch+1, column=4, value=val_iou)

    if model_saved:
        worksheet.cell(row=epoch+1, column=5, value="Saved")
    else:
        worksheet.cell(row=epoch+1, column=5, value="")


## data loader
class NPyDataset(Dataset):
    def __init__(self, folder_name, is_train=True):
        self.folder_name = folder_name
        self.is_train = is_train
        self.transform = self._get_transform()
        self.Phantom = True if DATA == 'P' or DATA == 'Phantom' else False

    def __len__(self):
        if self.Phantom:
            return (1400 if self.is_train else 600)
        return (845 if self.is_train else 362)

    def __getitem__(self, idx):
        if self.is_train:
            if self.Phantom:
                image = self._load_npy("PTrain/frame_%04d.npy" % idx)
                label = self._load_npy("PTrain_label/frame_%04d.npy" % idx)   
            else:
                image = self._load_npy("TTrain/frame_%04d.npy" % idx)
                label = self._load_npy("TTrain_label/frame_%04d.npy" % idx)
        else:
            if self.Phantom:
                image = self._load_npy("PTest/frame_%04d.npy" % idx)
                label = self._load_npy("PTest_label/frame_%04d.npy" % idx) 
            else: 
                image = self._load_npy("TTest/frame_%04d.npy" % idx)
                label = self._load_npy("TTest_label/frame_%04d.npy" % idx)

        if self.transform and TRANSFORM:
            image,label = self.transform(image,label)
        
        return image, label

    def _load_npy(self, filename):
        filename = os.path.join(self.folder_name, filename)
        # if self.is_train:
        return torch.unsqueeze(torch.tensor(np.float32(np.load(filename))),dim=0)
        # return torch.unsqueeze(torch.tensor(np.float32(np.load(filename)[::2,::2])),dim=0)

    def _get_transform(self):
        if self.is_train:
            return v2.Compose([
                # Add your desired transformations here
                # v2.RandomHorizontalFlip(),
                v2.RandomHorizontalFlip(p=0.5),
                # v2.RandomAffine(0, shear=15),
                # v2.GaussianBlur(kernel_size=3),
                # v2.ColorJitter(brightness=(0.5, 2.0)),
                # v2.RandomAffine(0, translate=(0.2, 0.2))
                # v2.Resize(size=(128, 128)),
                # v2.RandomResizedCrop(size=(128, 128), antialias=True),
                # v2.Normalize(mean=[0.5], std=[0.5]),
                # v2.RandomRotation(degrees=10),
            ])
        else:
            return None


## training
def train():
    model = UNet(1,1)  # input 1-channel 3d volume and output 1-channel segmentation (a probability map)
    if use_cuda:
        model.cuda()

    # training data loader
    train_set = NPyDataset(folder_name)
    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=0)
    # test/validation data loader
    test_set = NPyDataset(folder_name, is_train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0)

    # optimisation loop
    num_epochs=int(500)
    best_eval_loss = 9999
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    if TRAIN:
        os.makedirs("result", exist_ok=True)
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(['Epoch', 'Train Loss', 'Val Loss', 'Val IoU', 'Model Saved'])

        for epoch in range(num_epochs):
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0
            train_iou = 0.0
            
            for (images, labels) in enumerate(train_loader):
                # step += 1
                if use_cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(images)

                loss = loss_2Ddice(outputs, labels)

                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                iou,_ = calculate_average_metrics(np.squeeze(outputs.detach().cpu().numpy()), np.squeeze(labels.detach().cpu().numpy()))
                pred_flat = outputs.view(-1)>0.5
                target_flat = labels.view(-1)
                total_train += target_flat.size(0)
                train_iou +=iou
                correct_train += (pred_flat == target_flat).sum().item()
                
            ## validation
            y_pred_test=np.array([])
            ytest=np.array([])
            for i, (images, labels)  in enumerate(test_loader):
                
                images = images.cuda()
                output = model(images)
                # predicted = np.argmax(np.squeeze(output.detach().cpu().numpy()),axis=0)
                # predicted = np.float32(np.squeeze(output.detach().cpu().numpy()))
                predicted = np.float32(np.squeeze(output.detach().cpu().numpy(),axis=0)>0.5)
                # filepath_to_save = os.path.join(RESULT_PATH,"label_test-pt.npy")
                # np.save(filepath_to_save, predicted)
                gt = np.squeeze(labels.detach().cpu().numpy(),axis=0)
                if y_pred_test.size == 0:
                    y_pred_test = predicted
                    ytest = gt
                else:
                    y_pred_test = np.concatenate((y_pred_test, predicted),axis=0)
                    ytest = np.concatenate((ytest, gt),axis=0)
                
            val_iou,dice = calculate_average_metrics(ytest,y_pred_test)
            train_iou = train_iou/len(train_loader)
            epoch_train_loss = running_train_loss / len(train_loader)
            epoch_val_loss = 1-dice  
            
            print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Val iou: {val_iou:.2f}')
            
            model_saved = False

            if epoch_val_loss<best_eval_loss:
                model_saved = True
                best_eval_loss = epoch_val_loss
                torch.save(model, os.path.join(RESULT_PATH,'saved_' + type(optimizer).__name__ + '_model_pt'))
                print("Model Saved Successfully") 
            
            update_excel(epoch+1, epoch_train_loss, epoch_val_loss, val_iou, model_saved, worksheet)
            
            workbook.save("./result/progress.xlsx")
        print('Training done.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(os.path.join(RESULT_PATH,'saved_' + type(optimizer).__name__ + '_model_pt'))

    model = model.to(device)
    model.eval()
    y_pred_test=np.array([])
    ytest=np.array([])
    iou = []
    dice_coefficient = []
    for i, (images, labels)  in enumerate(test_loader):
        # concat = torch.cat((x0, x1), 0)
        \
        images = images.cuda()
        output = model(images)
        # predicted = np.argmax(np.squeeze(output.detach().cpu().numpy()),axis=0)
        # predicted = np.float32(np.squeeze(output.detach().cpu().numpy()))
        predicted = np.float32(np.squeeze(output.detach().cpu().numpy(),axis=0)>0.5)
        # filepath_to_save = os.path.join(RESULT_PATH,"label_test-pt.npy")
        # np.save(filepath_to_save, predicted)
        gt = np.squeeze(labels.detach().cpu().numpy(),axis=0)
        if y_pred_test.size == 0:
            y_pred_test = predicted
            ytest = gt
        else:
            y_pred_test = np.concatenate((y_pred_test, predicted),axis=0)
            ytest = np.concatenate((ytest, gt),axis=0)

        iou_val, dice_coefficient_val = calculate_2d_metrics(gt,predicted)

        iou.append(iou_val)
        dice_coefficient.append(dice_coefficient_val)

    # print("Average IOU is:" , sum(iou) / len(iou) )
    # print("Average DC is:" , sum(dice_coefficient) / len(dice_coefficient) )
    iou_val, dice_coefficient_val = calculate_average_metrics(ytest,y_pred_test)
    print("Final IOU:", iou_val)
    print("Final DC:", dice_coefficient_val)

if __name__ == '__main__':
    train()
