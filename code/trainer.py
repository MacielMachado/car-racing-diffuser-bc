import os
import git
import torch
import wandb
import logging
import numpy as np
from tqdm import tqdm
from utils import Params
import matplotlib.pyplot as plt
from run_car_racing import Tester
from torchvision import transforms
from cart_racing_v2 import CarRacing
from torch.utils.data import DataLoader
from train import CarRacingCustomDataset
from data_preprocessing import DataHandler
from run_car_racing import Tester
from cart_racing_v2 import CarRacing
from models import Model_Cond_Diffusion, Model_cnn_mlp, Model_cnn_bc

class Trainer():
    def __init__(self, n_epoch, lrate, device, n_hidden, batch_size, n_T,
                 net_type, drop_prob, extra_diffusion_steps, embed_dim,
                 guide_w, betas, dataset_path, run_wandb, record_run,
                 name='', param_search=False, embedding="Model_cnn_mlp",
                 dataset_origin="human"):
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
        self.name = name
        self.param_search = param_search
        self.run_wandb = run_wandb
        self.record_rum = record_run
        self.embedding = embedding
        self.dataset_origin = dataset_origin
        self.best_reward = float('-inf')
        self.patience = 20
        self.early_stopping_counter = 0

    def main(self):
        if self.run_wandb:
            self.config_wandb(project_name="OpenAI-Car-Racing-Article-Diffuser-Server", name=self.name)
        torch_data_train, dataload_train = self.prepare_dataset()
        x_dim, y_dim = self.get_x_and_y_dim(torch_data_train)
        conv_model = self.create_conv_model(x_dim, y_dim)
        model = self.create_agent_model(conv_model, x_dim, y_dim)
        optim = self.create_optimizer(model)
        model = self.train(model, dataload_train, optim)
        with torch.no_grad():
            self.evaluate(model.eval(), CarRacing(), name='eval_'+self.name)
        
    def evaluate(self, model, env, name, middle):
        tester = Tester(model, env, render=True, device=self.device)
        if middle:
            return tester.run_trainer(self.dataset_origin)
        else:
            tester.run()

    def config_wandb(self, project_name, name):
        wandb.login(key='9bcc371f01af2fc8ddab2c3ad226caad57dc4ac5')
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
                "guide_w": self.guide_w,
                "dataset": self.dataset_path,
                "model": self.embedding,
                "commit_hash": self.get_git_commit_hash()
            }
        if name != '':
            return wandb.init(project=project_name,
                              name=name+'_'+self.dataset_path.split(os.sep)[-1]+'_'+self.get_git_commit_hash()[0:4],
                              config=config)
        return wandb.init(project=project_name, config=config)

    def get_git_commit_hash(self):
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def prepare_dataset(self):
        tf = transforms.Compose([])
        torch_data_train = CarRacingCustomDataset(
            self.dataset_path, transform=tf, train_or_test='train', train_prop=1.0,
            dataset_origin=self.dataset_origin
        )
        dataload_train = DataLoader(
            torch_data_train, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        return torch_data_train, dataload_train
    
    def get_x_and_y_dim(self, torch_data_train):
        if len(torch_data_train.image_all.shape) == 3:
            torch_data_train.image_all = np.expand_dims(torch_data_train.image_all, axis=-1)
        if len(torch_data_train.action_all.shape) == 3:
            torch_data_train.action_all = np.squeeze(torch_data_train.action_all, axis=1)
        x_dim = torch_data_train.image_all.shape[1:]
        y_dim = torch_data_train.action_all.shape[1]
        return x_dim, y_dim
    
    def create_conv_model(self, x_dim, y_dim):

        if self.dataset_origin == 'ppo':
            cnn_out_dim = 512
        else:
            cnn_out_dim = 1152

        if self.embedding == "Model_cnn_bc":
            return Model_cnn_bc(self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type).to(self.device)
        elif self.embedding == "Model_cnn_mlp":
            return Model_cnn_mlp(x_dim, self.n_hidden, y_dim,
                                embed_dim=self.embed_dim,
                                net_type=self.net_type,
                                cnn_out_dim=cnn_out_dim).to(self.device)
        else:
            raise NotImplementedError
    
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

    def decay_lr(self, epoch, lrate):
        return lrate * ((np.cos((epoch / self.n_epoch) * np.pi) + 1) / 2)

    def concatenate_observations(self):
        pass
    
    def train(self, model, dataload_train, optim):
        for ep in tqdm(range(self.n_epoch), desc="Epoch"):
            results_ep = [ep]
            model.train()

            lr_decay = self.decay_lr(ep, self.lrate)
            # train loop
            pbar = tqdm(dataload_train)
            loss_ep, n_batch = 0, 0
            for x_batch, y_batch in pbar:
                DataHandler().plot_batch(x_batch, y_batch, self.batch_size, render=False)
                x_batch = x_batch.type(torch.FloatTensor).to(self.device)
                y_batch = y_batch.type(torch.FloatTensor).to(self.device)
                loss = model.loss_on_batch(x_batch, y_batch)
                optim.zero_grad()
                loss.backward()
                loss_ep += loss.detach().item()
                n_batch += 1
                pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
                optim.step()

                with torch.no_grad():
                    y_hat_batch = model.sample(x_batch)
                    action_MSE = extract_action_mse(y_batch, y_hat_batch)

                if self.run_wandb:
                    # log metrics to wandb
                    wandb.log({"loss": loss_ep/n_batch,
                                "lr": lr_decay,
                                "left_action_MSE": action_MSE[0],
                                "acceleration_action_MSE": action_MSE[1],
                                "right_action_MSE": action_MSE[2]})
                        
                    results_ep.append(loss_ep / n_batch)
            
            if ep % 10 == 0:
                stop = self.early_stopping(model, ep)
                if stop:
                    break

            if ep in [1, 20, 40, 80, 150, 250, 500, 600, 749]:
                name=f'model_novo_ep_{ep}'
                self.save_model(model, name, ep)

        if self.run_wandb:
            wandb.finish()
        
        return model
    
    def early_stopping(self, model, ep):
        with torch.no_grad():
            reward = self.evaluate(model.eval(), CarRacing(), name=self.name+'_eval', middle=True)
        wandb.log({"reward": reward})
        if reward > self.best_reward:
            self.best_reward = reward
            self.counter = 0
            name=self.name+'_model_best_reward'
            self.save_model(model, name, ep)
        else:
            self.counter += 1
        stop = False

        if self.counter >= self.patience:
            print(f'Early stopping after {ep+1} epochs without improvement.')
            stop = True
        return stop 


    def save_model(self, model, name, ep=''):
        # if self.param_search == True:
        #     return torch.save(model.state_dict(), os.path.join(os.getcwd(),name+'.pkl'))
        os.makedirs(os.getcwd()+'/model_pytorch/'+self.dataset.split(os.sep)[1], exist_ok=True)
        torch.save(self.model.state_dict(), os.getcwd()+'/model_pytorch/'+self.dataset_path.split(os.sep)[1]+'/'+self.dataset_path.split(os.sep)[2]+'_'+self.get_git_commit_hash()+'_ep_'+f'{ep}'+'.pkl')
        # return torch.save(model.state_dict(), 'experiments/' + self.name + '.pkl')

def extract_action_mse(y, y_hat):
    assert len(y) == len(y_hat)
    y_diff_pow_2 = torch.pow(y - y_hat, 2)
    y_diff_sum = torch.sum(y_diff_pow_2, dim=0)/len(y)
    mse = torch.pow(y_diff_sum, 0.5)
    return mse


if __name__ == '__main__':

    dataset_path = "dataset_fixed"
    params = Params("experiments/default/params.json")
    trainer_instance = Trainer( n_epoch=params.n_epoch,
                                lrate=params.lrate,
                                device=params.device,
                                n_hidden=params.n_hidden,
                                batch_size=1,
                                n_T=params.n_T,
                                net_type=params.net_type,
                                drop_prob=params.drop_prob,
                                extra_diffusion_steps=params.extra_diffusion_steps,
                                embed_dim=params.embed_dim,
                                guide_w=params.guide_w,
                                betas=(1e-4, 0.02),
                                dataset_path=dataset_path,
                                name='trainer_400',
                                run_wandb=False,
                                record_run=False,
                                embedding=params.embedding,
                                dataset_origin="human")
    trainer_instance.main()