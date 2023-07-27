from tqdm import tqdm
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models import Model_Cond_Diffusion, Model_cnn_mlp
from train import ClawCustomDataset, CarRacingCustomDataset
from data_preprocessing import DataHandler
import matplotlib.pyplot as plt
import math

class Trainer():
    def __init__(self, n_epoch, lrate, device, n_hidden, batch_size, n_T,
                 net_type, drop_prob, extra_diffusion_steps, embed_dim,
                 guide_w, betas, dataset_path):
        self.n_epoch = n_epoch
        self.lrate = lrate
        self.device = device
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.n_T = n_T
        self.net_type = net_type
        self.drop_prob = drop_prob
        self.extra_diffusion_steps = extra_diffusion_steps
        self.embed_dim = embed_dim
        self.guide_w = guide_w
        self.betas = betas
        self.dataset_path = dataset_path

    def main(self):
        wandb_run = self.config_wandb(project_name="car-racing-diffuser-bc")
        torch_data_train, dataload_train = self.prepare_dataset()
        x_dim, y_dim = self.get_x_and_y_dim(torch_data_train)
        conv_model = self.create_conv_model(x_dim, y_dim)
        model = self.create_agent_model(conv_model, x_dim, y_dim)
        optim = self.create_optimizer(model)
        self.train(model, dataload_train, optim)

    def config_wandb(self, project_name):
        return wandb.init(  # Set the wandb project where this run will be logged
                            project=project_name,

                            # Set the session name
                            # name="greyscale_96x96",

                            # track hyperparameters and run metadata
                            config={
                                "n_epoch": self.n_epoch,
                                "lrate": self.lrate,
                                "device": self.device,
                                "n_hidden": self.n_hidden,
                                "batch_size": self.batch_size,
                                "n_T": self.n_T,
                                "net_type": self.net_type,
                                "drop_prob": self.drop_prob,
                                "extra_diffusion_steps": self.extra_diffusion_steps,
                                "embed_dim": self.embed_dim,
                                "guide_w": self.guide_w
                            }
                        )

    def prepare_dataset(self):
        tf = transforms.Compose([])
        torch_data_train = CarRacingCustomDataset(
            self.dataset_path, transform=tf, train_or_test='train', train_prop=0.90
        )
        dataload_train = DataLoader(
            torch_data_train, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        return torch_data_train, dataload_train
    
    def get_x_and_y_dim(self, torch_data_train):
        torch_data_train.image_all = np.expand_dims(torch_data_train.image_all, axis=-1)
        x_dim = torch_data_train.image_all.shape[1:]
        y_dim = torch_data_train.action_all.shape[1]
        return x_dim, y_dim
    
    def create_conv_model(self, x_dim, y_dim):
        return Model_cnn_mlp(x_dim, self.n_hidden, y_dim,
                             embed_dim=self.embed_dim,
                             net_type=self.net_type).to(self.device)
    
    def create_agent_model(self, conv_model, x_dim, y_dim):
        return Model_Cond_Diffusion(
            conv_model,
            betas=self.betas,
            n_T = self.n_T,
            device=self.device,
            x_dim=x_dim,
            y_dim=y_dim,
            drop_prob=self.drop_prob,
            guide_w=self.guide_w
        ).to(self.device)
    
    def create_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.lrate)

    def decay_lr(self, epoch):
        return lrate * ((np.cos((epoch / self.n_epoch) * np.pi) + 1) / 2)

    def concatenate_observations(self):
        pass
    
    def train(self, model, dataload_train, optim):
        for ep in tqdm(range(self.n_epoch), desc="Epoch"):
            results_ep = [ep]
            model.train()

            lr_decay = self.decay_lr(ep)
            # train loop
            pbar = tqdm(dataload_train)
            loss_ep, n_batch = 0, 0
            for x_batch, y_batch in pbar:
                DataHandler().plot_batch(x_batch, y_batch, self.batch_size, render=False)
                x_batch = x_batch.type(torch.FloatTensor).to(device)
                y_batch = y_batch.type(torch.FloatTensor).to(device)
                loss = model.loss_on_batch(x_batch, y_batch)
                optim.zero_grad()
                loss.backward()
                loss_ep += loss.detach().item()
                n_batch += 1
                pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
                optim.step()

            y_hat_batch = model.sample(x_batch)
            action_MSE = extract_action_mse(y_batch, y_hat_batch)

            # log metrics to wandb
            wandb.log({"loss": loss_ep/n_batch,
                        "lr": lr_decay,
                        "left_action_MSE": action_MSE[0],
                        "acceleration_action_MSE": action_MSE[1],
                        "right_action_MSE": action_MSE[2]})
                
            results_ep.append(loss_ep / n_batch)


def extract_action_mse(y, y_hat):
    assert len(y) == len(y_hat)
    y_diff_pow_2 = torch.pow(y - y_hat, 2)
    y_diff_sum = torch.sum(y_diff_pow_2, dim=0)/len(y)
    mse = torch.pow(y_diff_sum, 0.5)
    return mse


if __name__ == '__main__':

    n_epoch = 40
    lrate = 1e-4
    device = "cpu"
    n_hidden = 128
    batch_size = 32
    n_T = 50
    net_type = "transformer"
    drop_prob = 0.0
    extra_diffusion_steps = 16
    embed_dim = 128
    guide_w = 0.0
    betas = (1e-4, 0.02)
    dataset_path = "tutorial"

    trainer_instance = Trainer( n_epoch=n_epoch,
                                lrate=lrate,
                                device=device,
                                n_hidden=n_hidden,
                                batch_size=batch_size,
                                n_T=n_T,
                                net_type=net_type,
                                drop_prob=drop_prob,
                                extra_diffusion_steps=extra_diffusion_steps,
                                embed_dim=embed_dim,
                                guide_w=guide_w,
                                betas=betas,
                                dataset_path=dataset_path)
    trainer_instance.main()