import os
import random
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
import itertools

from dataset import *
from utils import *
from model import *
from loss import *
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

class Trainer:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epoch", type=int, default=0, help="epoch to start \
                                        training from, 0 starts from scratch, >0 starts from saved checkpoints")#
        parser.add_argument("--n_epochs", type=int, default=200, help="total number of epochs of training")
        parser.add_argument("--lr_decay", type=float, default=2e-6, help="learning rate decay")
        parser.add_argument("--step_size", type=int, default=1, help="learning rate decay step size")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--lr_G", type=float, default=2e-4, help="G adam: learning rate")
        parser.add_argument("--lr_D", type=float, default=2e-4, help="D adam: learning rate")
        parser.add_argument("--percep", type=float, default=0.0, help="perceptual loss weight")
        parser.add_argument("--style", type=float, default=0.0, help="style loss weight")
        parser.add_argument("--ckpt_interval", type=int, default=2, help="interval between model checkpoints")
        parser.add_argument("--log_interval", type=int, default=200, help="interval between print logs")
        parser.add_argument("--output_dir", type=str, default="results", help="path to save model and images")
        parser.add_argument("--log_name", type=str, default="horse2zebra", help="log file name, the data category")#
        parser.add_argument("--root", type=str, default="/home/bjyang/ece50024/cyclegan/", help="root path")
        parser.add_argument("--mode", type=str, default="train", help="train/test")#


        self.opt = parser.parse_args()
        self.device = torch.device("cuda")
        self.eps = 1e-6
        self.opt.n_cpu = self.opt.batch_size//8
        self.opt.sub_folder = self.opt.log_name.split('_')[0]

        # dataloader
        if self.opt.mode != "test":
            self.train_dataset = Unpaired_Data(mode='train', name=self.opt.sub_folder)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, 
                                            shuffle=True, num_workers=self.opt.n_cpu, drop_last=False, pin_memory=True)
    
        test_datasetA = test_img(name=self.opt.sub_folder, category='A')
        test_datasetB = test_img(name=self.opt.sub_folder, category='B')
        self.test_loaderA = DataLoader(test_datasetA, batch_size=1, shuffle=False, pin_memory=True)
        self.test_loaderB = DataLoader(test_datasetB, batch_size=1, shuffle=False, pin_memory=True)

        self.init_model_optimizer()
        self.prep_model()

    def weights_init(self, m):
        """
        Uses the DCGAN initializations for the weights
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('InstanceNorm') != -1:
            # import pdb; pdb.set_trace()
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def init_model_optimizer(self):
        # loss function
        self.criterion_cycle = nn.L1Loss().to(self.device)
        self.criterion_gan = GANLoss().to(self.device)
        self.criterion_perceptual = PerceptualLoss(layer_weights={'pool5': 1.0, 'relu5_1': 1.0}).to(self.device)
        self.criterion_style = style_loss
        # initialize models
        self.G_AB = Generator()
        self.G_BA = Generator()
        self.D1 = Discriminator()
        self.D2 = Discriminator()
        # self.G.apply(self.weights_init)
        
        self.G_AB.apply(self.weights_init)
        self.G_BA.apply(self.weights_init)
        self.D1.apply(self.weights_init)
        self.D2.apply(self.weights_init)

        # optimizer and scheduler
        if self.opt.mode == "train":
            self.G_optimizer = Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=self.opt.lr_G)
            # self.G_optimizer = Adam(self.G.parameters(), lr=self.opt.lr_G)
            # self.G_AB_optimizer = Adam(self.G_AB.parameters(), lr=self.opt.lr_G)
            # self.G_BA_optimizer = Adam(self.G_BA.parameters(), lr=self.opt.lr_G)
            self.D1_optimizer = Adam(self.D1.parameters(), lr=self.opt.lr_D)
            self.D2_optimizer = Adam(self.D2.parameters(), lr=self.opt.lr_D)
    
    def prep_model(self):
        # self.G = self.G.to(self.device)
        self.G_AB = self.G_AB.to(self.device)
        self.G_BA = self.G_BA.to(self.device)
        self.D1 = self.D1.to(self.device)
        self.D2 = self.D2.to(self.device)
        # self.G = nn.DataParallel(self.G)
        self.G_AB = nn.DataParallel(self.G_AB)
        self.G_BA = nn.DataParallel(self.G_BA)
        self.D1 = nn.DataParallel(self.D1)
        self.D2 = nn.DataParallel(self.D2)
        if self.opt.epoch!=0:
            self.load_model(self.opt.epoch)
    
    def load_model(self, e):
        # self.G.load_state_dict(torch.load(os.path.join(self.opt.root, self.opt.output_dir, 
        #             'ckpts', self.opt.log_name, 'G_' + str(int(e)) + ".pth")))
        self.G_AB.load_state_dict(torch.load(os.path.join(self.opt.root, self.opt.output_dir,
                    'ckpts', self.opt.log_name, 'G_AB_' + str(int(e)) + ".pth")))
        self.G_BA.load_state_dict(torch.load(os.path.join(self.opt.root, self.opt.output_dir,
                    'ckpts', self.opt.log_name, 'G_BA_' + str(int(e)) + ".pth")))
        self.D1.load_state_dict(torch.load(os.path.join(self.opt.root, self.opt.output_dir,
                    'ckpts', self.opt.log_name, 'D1_' + str(int(e)) + ".pth")))
        self.D2.load_state_dict(torch.load(os.path.join(self.opt.root, self.opt.output_dir,
                    'ckpts', self.opt.log_name, 'D2_' + str(int(e)) + ".pth")))

    def inference(self, img, ref):
        # fake_img, fake_ref = self.G(img, ref)
        # recon_img, recon_ref = self.G(fake_img, fake_ref)
        fake_img = self.G_AB(img)
        fake_ref = self.G_BA(ref)
        recon_img = self.G_BA(fake_img)
        recon_ref = self.G_AB(fake_ref)
        score_fake_img = self.D1(fake_img.detach()) # detach to avoid BP to G
        score_fake_ref = self.D2(fake_ref.detach())
        score_real_ref = self.D1(ref)
        score_real_img = self.D2(img)
        return {'fake_img': fake_img, 'fake_ref': fake_ref, 
                'recon_img': recon_img, 'recon_ref': recon_ref,
                 'score_fake_img': score_fake_img, 'score_fake_ref': score_fake_ref,
                'score_real_img': score_real_img, 'score_real_ref': score_real_ref}

    def train_a_epoch(self, e, data_loader, loss_dict):
        loss_print=0

        for i, train_data in enumerate(tqdm(data_loader)):
            img = train_data["img"].to(self.device)
            ref = train_data["ref"].to(self.device)

            pred = self.inference(img, ref)
            d1_loss = self.criterion_gan(pred["score_fake_img"], False) + self.criterion_gan(pred["score_real_ref"], True)
            d2_loss = self.criterion_gan(pred["score_fake_ref"], False) + self.criterion_gan(pred["score_real_img"], True)
            loss_dict['D1_loss'].append(d1_loss.item())
            loss_dict['D2_loss'].append(d2_loss.item())
            # train D1
            self.D1_optimizer.zero_grad()
            d1_loss.backward()
            self.D1_optimizer.step()
            # train D2
            self.D2_optimizer.zero_grad()
            d2_loss.backward()
            self.D2_optimizer.step()

            self.D1.eval()
            self.D2.eval()

            # train G
            score_fake_img = self.D1(pred["fake_img"])
            score_fake_ref = self.D2(pred["fake_ref"])
            # score_real_ref = self.D1(ref)
            # score_real_img = self.D2(img)

            gan_loss = self.criterion_gan(score_fake_img, True) + self.criterion_gan(score_fake_ref, True)
            if self.opt.percep == 0:
                perceptual_loss = torch.tensor(0).to(self.device)
            else:
                perceptual_loss = self.criterion_perceptual(pred["fake_img"], img)[0] + \
                                self.criterion_perceptual(pred["fake_ref"], ref)[0]
            if self.opt.style == 0:
                style_loss = torch.tensor(0).to(self.device)
            else:
                style_loss = self.criterion_style(pred["fake_img"], ref) + \
                                self.criterion_style(pred["fake_ref"], img)
            recon_loss = self.criterion_cycle(pred["recon_img"], img) + self.criterion_cycle(pred["recon_ref"], ref)
            loss_dict['GAN_loss'].append(gan_loss.item())
            loss_dict['perceptual_loss'].append(perceptual_loss.item())
            loss_dict['style_loss'].append(style_loss.item())
            loss_dict['recon_loss'].append(recon_loss.item())
            g_loss = 10 * recon_loss + gan_loss + \
                        self.opt.percep * perceptual_loss + self.opt.style * style_loss
            loss_dict['total_loss'].append(g_loss.item())

            self.G_optimizer.zero_grad()
            # self.G_AB_optimizer.zero_grad()
            # self.G_BA_optimizer.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()
            # self.G_AB_optimizer.step()
            # self.G_BA_optimizer.step()

            self.D1.train()
            self.D2.train()

            loss_print+=g_loss.item()

            if (i+1) % self.opt.log_interval == 0:
                print(self.G_optimizer.param_groups[-1]['lr'], e, i, ': ', loss_print/self.opt.log_interval)
                # print(self.G_AB_optimizer.param_groups[-1]['lr'], e, i, ': ', loss_print/self.opt.log_interval)
                print('D1_loss: ', np.mean(loss_dict['D1_loss'][-self.opt.log_interval:]))
                print('D2_loss: ', np.mean(loss_dict['D2_loss'][-self.opt.log_interval:]))
                print('GAN_loss: ', np.mean(loss_dict['GAN_loss'][-self.opt.log_interval:]))
                print('recon_loss: ', np.mean(loss_dict['recon_loss'][-self.opt.log_interval:]))
                print('perceptual_loss: ', np.mean(loss_dict['perceptual_loss'][-self.opt.log_interval:]))
                print('style_loss: ', np.mean(loss_dict['style_loss'][-self.opt.log_interval:]))
                print('total_loss: ', np.mean(loss_dict['total_loss'][-self.opt.log_interval:]))
                if not os.path.exists('./results/logs/'):
                    os.makedirs('./results/logs/')
                with open('./results/logs/'+self.opt.log_name+'.txt', 'a') as f:
                    f.write("train: lr %.8f|%d|%d|%4.7f\n"%(self.G_optimizer.param_groups[-1]['lr'], e, i, loss_print/self.opt.log_interval))
                    # f.write("train: lr %.8f|%d|%d|%4.7f\n"%(self.G_AB_optimizer.param_groups[-1]['lr'], e, i, loss_print/self.opt.log_interval))
                    f.write("D1_loss: %4.7f\n"%(np.mean(loss_dict['D1_loss'][-self.opt.log_interval:])))
                    f.write("D2_loss: %4.7f\n"%(np.mean(loss_dict['D2_loss'][-self.opt.log_interval:])))
                    f.write("GAN_loss: %4.7f\n"%(np.mean(loss_dict['GAN_loss'][-self.opt.log_interval:])))
                    f.write("recon_loss: %4.7f\n"%(np.mean(loss_dict['recon_loss'][-self.opt.log_interval:])))
                    f.write("perceptual_loss: %4.7f\n"%(np.mean(loss_dict['perceptual_loss'][-self.opt.log_interval:])))
                    f.write("style_loss: %4.7f\n"%(np.mean(loss_dict['style_loss'][-self.opt.log_interval:])))
                    f.write("total_loss: %4.7f\n"%(np.mean(loss_dict['total_loss'][-self.opt.log_interval:])))
                loss_print=0.0
                plt.figure(figsize=(20, 10))
                plt.subplot(2, 4, 1)
                plt.title('D1_loss')
                plt.plot(loss_dict['D1_loss'])
                plt.subplot(2, 4, 2)
                plt.title('D2_loss')
                plt.plot(loss_dict['D2_loss'])
                plt.subplot(2, 4, 3)
                plt.title('GAN_loss')
                plt.plot(loss_dict['GAN_loss'])
                plt.subplot(2, 4, 4)
                plt.title('recon_loss')
                plt.plot(loss_dict['recon_loss'])
                plt.subplot(2, 4, 5)
                plt.title('perceptual_loss')
                plt.plot(loss_dict['perceptual_loss'])
                plt.subplot(2, 4, 6)
                plt.title('style_loss')
                plt.plot(loss_dict['style_loss'])
                plt.subplot(2, 4, 7)
                plt.title('total_loss')
                plt.plot(loss_dict['total_loss'])
                plt.savefig('./results/logs/'+self.opt.log_name+'_loss.png')
                plt.close()

                # show images in figure
                def denorm(x):
                    out = (x + 1) / 2
                    return out.clamp_(0, 1)
                plt.figure(figsize=(30, 20))
                plt.subplot(2, 3, 1)
                plt.title('img')
                plt.imshow(denorm(img[0]).cpu().numpy().transpose(1, 2, 0))
                plt.subplot(2, 3, 4)
                plt.title('ref')
                plt.imshow(denorm(ref[0]).cpu().numpy().transpose(1, 2, 0))
                plt.subplot(2, 3, 2)
                plt.title('fake_img')
                plt.imshow(denorm(pred["fake_img"][0].detach()).cpu().numpy().transpose(1, 2, 0))
                plt.subplot(2, 3, 5)
                plt.title('fake_ref')
                plt.imshow(denorm(pred["fake_ref"][0].detach()).cpu().numpy().transpose(1, 2, 0))
                plt.subplot(2, 3, 3)
                plt.title('recon_img')
                plt.imshow(denorm(pred["recon_img"][0].detach()).cpu().numpy().transpose(1, 2, 0))
                plt.subplot(2, 3, 6)
                plt.title('recon_ref')
                plt.imshow(denorm(pred["recon_ref"][0].detach()).cpu().numpy().transpose(1, 2, 0))
                plt.savefig('./results/logs/'+self.opt.log_name+'_img.png')
                plt.close()


        if e % self.opt.ckpt_interval == 0:
            for model_name in ['G_AB', 'G_BA', 'D1', 'D2']:
                ckpt_model_filename = model_name + '_' + str(e) + ".pth"
                ckpt_model_path = os.path.join(self.opt.root, self.opt.output_dir, 
                                                'ckpts', self.opt.log_name, ckpt_model_filename)
                directory = os.path.join(self.opt.root, self.opt.output_dir, 'ckpts', self.opt.log_name)
                if not os.path.exists(directory):
                    Path(directory).mkdir(parents=True, exist_ok=True)

                state = getattr(self, model_name).state_dict()
                torch.save(state, ckpt_model_path)

                print("model saved to %s" % ckpt_model_path)

        # learning rate decay
        if e>100:
            self.opt.lr_G = self.opt.lr_G - self.opt.lr_decay
            self.opt.lr_D = self.opt.lr_D - self.opt.lr_decay
            for i in range(len(self.G_optimizer.param_groups)): 
                self.G_optimizer.param_groups[i]['lr'] = self.opt.lr_G
            # for i in range(len(self.G_AB_optimizer.param_groups)):
            #     self.G_AB_optimizer.param_groups[i]['lr'] = self.opt.lr_G
            for i in range(len(self.G_optimizer.param_groups)):
                self.G_optimizer.param_groups[i]['lr'] = self.opt.lr_G
            for i in range(len(self.D1_optimizer.param_groups)):
                self.D1_optimizer.param_groups[i]['lr'] = self.opt.lr_D
            for i in range(len(self.D2_optimizer.param_groups)):
                self.D2_optimizer.param_groups[i]['lr'] = self.opt.lr_D
    

    def train(self):
        if self.opt.mode == "train":
            loss_dict = {'D1_loss':[], 'D2_loss':[], 'GAN_loss':[], 'recon_loss':[], \
                    'perceptual_loss':[], 'style_loss':[], 'total_loss':[]}

            for e in range(self.opt.epoch, int(self.opt.n_epochs)+1):
                self.train_a_epoch(e, self.train_loader, loss_dict)
                
            # final test
            self.test(self.opt.n_epochs, file_name=self.opt.log_name)

        elif self.opt.mode == "test":
            self.load_model(self.opt.epoch)

            print("test start")
            self.test(self.opt.epoch, file_name=self.opt.log_name)
            print("test end")
        

    def test(self, e, file_name):
        with torch.no_grad():
            self.load_model(e)
            self.G_AB.eval()
            self.G_BA.eval()
            self.D1.eval()
            self.D2.eval()
            def denorm(x):
                out = (x + 1) / 2
                return out.clamp_(0, 1)
            for _, test_data in enumerate(tqdm(self.test_loaderA)):
                img = test_data["img"].to(self.device)
                img_name = test_data["img_name"][0]
                pred = self.G_AB(img)
                if not os.path.exists('./results/imgs/'+self.opt.log_name+'/A/'):
                    os.makedirs('./results/imgs/'+self.opt.log_name+'/A/')
                cv2.imwrite('./results/imgs/'+self.opt.log_name+'/A/'+img_name[:-4]+'.png', tensor2img(denorm(pred[0]).cpu()))
            
            for _, test_data in enumerate(tqdm(self.test_loaderB)):
                img = test_data["img"].to(self.device)
                img_name = test_data["img_name"][0]
                pred = self.G_BA(img)
                if not os.path.exists('./results/imgs/'+self.opt.log_name+'/B/'):
                    os.makedirs('./results/imgs/'+self.opt.log_name+'/B/')
                cv2.imwrite('./results/imgs/'+self.opt.log_name+'/B/'+img_name[:-4]+'.png', tensor2img(denorm(pred[0]).cpu()))

            print("test done")        

if __name__ == '__main__':
    Trainer = Trainer()
    Trainer.train()


# train&val&test:
# CUDA_VISIBLE_DEVICES=0 python train.py --epoch 100 --log_name "horse2zebra_relu5-1"
# CUDA_VISIBLE_DEVICES=0 python train.py --mode "test" --log_name "horse2zebra_original_2generator2" --epoch 200