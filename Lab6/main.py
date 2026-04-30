import warnings
warnings.filterwarnings('ignore')
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import argparse
from diffusers import DDPMScheduler, UNet2DModel
from model import MyConditionedUNet
from dataloader import ICLEVR_dataset
from tqdm import tqdm
from evaluator import evaluation_model

class Trainer():
    def __init__(self, dataset, model, noise_scheduler, device, args):
        self.dataset = dataset
        self.device = device
        self.epochs = args.epoch
        self.n_samples = args.n_samples
        self.image_channel = args.image_channel
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.best_accuracy = -10
        self.model = model.to(self.device)
        self.noise_scheduler = noise_scheduler
        self.loss_fn = nn.MSELoss()
        self.data_loader = DataLoader(self.dataset, self.batch_size, shuffle = True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr)
        
    def get_test_label(self, data_path = "test.json"):
        label_dict = json.load(open("/home/pp037/DL/Lab6/objects.json"))
        labels = json.load(open(os.path.join('/home/pp037/DL/Lab6', data_path)))
        
        label_list = []
        for i in range(len(labels)):
            onehot_label = torch.zeros(24, dtype = torch.float32)
            for j in range(len(labels[i])):
                onehot_label[label_dict[labels[i][j]]] = 1 
            label_list.append(onehot_label)
        
        return label_list

    def save_images(self, images, name):
        grid = torchvision.utils.make_grid(images)
        save_image(grid, fp = "/home/pp037/DL/Lab6/image" + name +".png")

    def sample(self, epoch = 0):
        test_label = torch.stack(self.get_test_label()).to(self.device)
        new_test_label = torch.stack(self.get_test_label(data_path = 'new_test.json')).to(self.device)
        
        x_test = torch.randn([len(test_label), self.image_channel, self.image_size, self.image_size], device = self.device)
        x_new_test = torch.randn([len(new_test_label), self.image_channel, self.image_size, self.image_size], device = self.device)
        with tqdm(self.noise_scheduler.timesteps, unit = 'Step', desc = 'Test') as tqdm_loader:
            for index, t in enumerate(tqdm_loader):
                with torch.no_grad():
                    residual_test = self.model(x_test, t, test_label).sample
                    residual_new_test = self.model(x_new_test, t, new_test_label).sample
                x_test = self.noise_scheduler.step(residual_test, t, x_test).prev_sample  
                x_new_test = self.noise_scheduler.step(residual_new_test, t, x_new_test).prev_sample         
        
        test_label = torch.tensor(test_label, dtype = torch.float32)
        new_test_label = torch.tensor(new_test_label, dtype = torch.float32)
        
        image = (x_test / 2 + 0.5).clamp(0, 1)
        new_image = (x_new_test / 2 + 0.5).clamp(0, 1)

        test_accuracy = evaluation_model().eval(x_test, test_label)
        new_test_accuracy = evaluation_model().eval(x_new_test, new_test_label)
        print(f'Test: {test_accuracy}')
        print(f'New Test: {new_test_accuracy}')
        accuracy = 0.5 * new_test_accuracy + 0.5 * test_accuracy

        if accuracy >= self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model,'Best_model.pkl')
        
        save_image(make_grid(image, nrow = 8), f"./image/test_{epoch}.png")
        save_image(make_grid(new_image, nrow = 8), f"./image/new_test_{epoch}.png")

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch + 1}')
            total_loss = 0
            with tqdm(self.data_loader, unit = 'Batch', desc = 'Train') as tqdm_loader:
                for index, (image, label) in enumerate(tqdm_loader):
                    image = image.to(self.device)
                    label = torch.tensor(label.to(device = self.device), dtype = torch.float32)

                    noise = torch.randn_like(image).to(self.device)
                    timesteps = torch.randint(0, 999, (image.shape[0],)).long().to(self.device)
                    noise_image = self.noise_scheduler.add_noise(image, noise, timesteps)
                    predict_noise = self.model(noise_image, timesteps, label).sample

                    loss = self.loss_fn(predict_noise, noise)
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 5.0)
                    self.optimizer.step()

                    total_loss += loss.detach().cpu()
                    average_loss = total_loss / (index + 1)
                    tqdm_loader.set_postfix(Average_loss = average_loss.item())

            if epoch % 5 == 0:
                self.sample(epoch)

    def test(self, model_path = None):
        if model_path != None:
            torch.load(self.model, model_path)
        self.sample()

def main():
    model = MyConditionedUNet(
        sample_size = 64,  
        in_channels = 3,                
        out_channels = 3,
        layers_per_block = 2,
        block_out_channels = (64, 128, 256, 256, 512, 512),
        down_block_types = (
            "DownBlock2D",         
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",      
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",      
            "UpBlock2D",           
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
        class_embed_type = "timestep")
    
    train_dataset = ICLEVR_dataset(root = args.data_path)
    noise_scheduler = DDPMScheduler(num_train_timesteps = 1000, beta_schedule = 'squaredcos_cap_v2')
    trainer = Trainer(train_dataset, model, noise_scheduler, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), args)
    
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = '/home/pp037/DL/Lab6/Best_model.pkl')
    parser.add_argument('--data_path', default = '/home/pp037/DL/Lab6')
    parser.add_argument('--epoch', default = 100, type = int)
    parser.add_argument('--n_samples', default = 64, type = int)
    parser.add_argument('--image_channel', default = 3, type = int)
    parser.add_argument('--image_size', default = 64, type = int)
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--lr', default = 1e-4, type = float)
    args = parser.parse_args()

    main()