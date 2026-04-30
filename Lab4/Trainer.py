import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

# 峰值訊號與雜訊比（Peak Signal-to-Noise Ratio）用於評估壓縮或影像處理中失真的程度
def Generate_PSNR(imgs1, imgs2, data_range=1.): # data_range 是 pixel 值的範圍
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def kl_criterion(mu, logvar, batch_size):   # KL divergence 是用於衡量兩個機率分布間差異的指標
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size  # 平均每個樣本的 KL divergence
    return KLD

class kl_annealing():
    def __init__(self, args, current_epoch = 0, iters_per_cycle = 0, current_pos = 0):
        # TODO
        self.args = args
        self.current_epoch = current_epoch
        self.iters_per_cycle = args.num_epoch / args.kl_anneal_cycle
        self.current_pos = current_pos
        self.beta = 0.0
        
    def update(self):
        # TODO
        self.current_epoch += 1
        self.current_pos = self.current_epoch % self.iters_per_cycle    
        
    def get_beta(self):
        # TODO
        if self.args.kl_anneal_type == 'Cyclical':
            self.beta = min(self.current_pos * self.frange_cycle_linear(), 1.0)
        elif self.args.kl_anneal_type == 'Monotonic':
            self.args.kl_anneal_ratio += 0.1
            self.beta = min(self.args.kl_anneal_ratio, 1.0)
        else:
            self.beta = 1.0
        
        return self.beta

    def frange_cycle_linear(self, start = 0.0, stop = 1.0, ratio = 1):    # 計算週期性線性範圍函式的值，該函式在週期內根據位置逐漸變化，並可通過參數進行調整
        # TODO
        beta_rate = (stop - start) / (self.iters_per_cycle-1)*ratio
        return beta_rate
        
class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        self.best_psnr = 0
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3) # 生成模型，將特徵轉換成圖像
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr # 初始機率
        self.tfr_d_step = args.tfr_d_step   # 機率的遞減步幅
        self.tfr_sde = args.tfr_sde # 在哪個 epoch 開始進行 Teacher Forcing 比例的調整
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        loss_list = {}
        loss_list['loss'] = []
        loss_list['tfr'] = []
        for i in range(self.args.num_epoch):
            total_loss = 0
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False    # 根據機率決定是否啟用 Teacher forcing
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                total_loss += loss
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            average_loss = (total_loss/len(train_loader)).detach().cpu()
            loss_list['loss'].append(average_loss)
            loss_list['tfr'].append(self.tfr)
            
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        
        show_curve(loss_list, title = 'Loss', x_label = 'epoch', y_label = 'loss', file_name = 'loss')    
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        # 將 tensor 進行維度重排，原本:(B, seq, C, H, W)
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)

        loss = 0
        out = img[0]

        for i in range(1, self.train_vi_len):   # 遍歷 seq 中的每一幀(除了第 0 幀)
            label_feat = self.label_transformation(label[i])
            if adapt_TeacherForcing:
                human_feat_hat = self.frame_transformation(img[i - 1])  # 將前一張 ground truth 照片轉換為預測的人體特徵
            else:
                human_feat_hat = self.frame_transformation(out) # 將輸出圖像轉換為預測的人體特徵
            
            z, mu, logvar = self.Gaussian_Predictor(self.frame_transformation(img[i]), label_feat) # Posterior predicton
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    # 用 last generated frame(human_feat_hat)、Pose(label_feat)、Z sample from prior distribution 進行解碼器融合操作，獲取解碼參數
            out = self.Generator(parm)  # 使用解碼參數生成新的圖像輸出

            mse_loss = self.mse_criterion(out, img[i])
            kld_loss = kl_criterion(mu, logvar, self.batch_size)
        
            beta = self.kl_annealing.get_beta()
            loss += mse_loss + kld_loss*beta

        self.optim.zero_grad()  # 清空梯度
        loss.backward()
        self.optimizer_step()
        
        return loss

    def val_one_step(self, img, label):
        # TODO
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)

        loss = 0
        total_psnr = 0
        psnr_list = {}
        psnr_list['PSNR'] = []

        out = img[0]
        
        for i in range(1, self.val_vi_len):
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)
            
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            out = self.Generator(parm)

            mse_loss = self.mse_criterion(out, img[i])
            # kld_loss = kl_criterion(mu, logvar, self.batch_size)
            # 因為實際情況沒有 Posterior 產生的分佈，因此都假設實際分佈很貼近 prior distribution
            # 此時的 loss 就只關注 mse，也就是圖片品質
            # 至於 KL term ，validation 不需理會他
            #beta = self.kl_annealing.get_beta()
            loss += mse_loss  

            psnr = Generate_PSNR(out, img[i]).cpu() # 計算生成圖像和實際圖像之間的 PSNR
            psnr_list['PSNR'].append(psnr)
            total_psnr += psnr
        
        avg_psnr = total_psnr / self.val_vi_len
        print(avg_psnr)
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
        
        show_curve(psnr_list, title = 'PSNR', x_label = 'frame', y_label = 'psnr', file_name = 'val_psnr')
        return loss
                
    def make_gif(self, images_list, img_name):
        new_list = []   # 創建一個新的圖像列表用於儲存處理過的圖像
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))   # # 將每個圖像張量轉換為 PIL 形式並添加到新列表中
        
        # 將新列表中的圖像保存為 GIF 文件
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,   # 丟棄最後不足一個批次的數據
                                  shuffle=False)  # 不將訓練資料集打亂
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.args.tfr_sde: # 到達了可以開始調整的時間點
            self.tfr = max(0, self.tfr - self.args.tfr_sde) # 隨著訓練的進行，逐步降低 Teacher Forcing 比例，越來越少依賴真實目標輸出，而更多地自行生成

    
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

def show_curve(data, title, x_label, y_label, file_name):
    plt.figure()
    plt.title(title)
    for i in data:
        plt.plot(data[i], label = i)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)

def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
     
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")

    args = parser.parse_args()
    
    main(args)