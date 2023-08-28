import os
import cv2
import torch
import wandb
import numpy as np
from utils import Params
import matplotlib.pyplot as plt
from cart_racing_v2 import CarRacing
from data_preprocessing import DataHandler
from record_observations import RecordObservations
from models import Model_Cond_Diffusion, Model_cnn_mlp


class Tester(RecordObservations):
    def __init__(self, model, env, render=True, device="cpu", name="version"):
        super(Tester).__init__()
        self.model = model
        self.env = env
        self.name = name
        self.render = render
        self.device = device

    def run(self, run_wandb, name=''):
        self.name = name
        if run_wandb:
            self.config_wandb(project_name="car-racing-diffuser-bc-v2", name=name)
        episode = 0
        reward_list = []          
        while episode < 10:
            obs, _ = self.env.reset()
            reward_list = []   
            reward = 0
            counter=0
            done = False
            truncated = False
            while counter < 1000:
                self.model.eval()
                obs_tensor = self.preprocess_obs(obs)
                torch.from_numpy(obs_tensor).float().to(self.device).shape
                obs_tensor = (
                    torch.Tensor(obs_tensor).type(torch.FloatTensor).to(self.device)
                    )
                action = self.model.sample(obs_tensor).to(self.device)
                obs, new_reward, done, _, truncated = self.env.step(action.detach().cpu().numpy()[0])
                self.array_to_img(obs, action, frame=counter)
                reward += new_reward
                counter += 1
                print(f"{version} - episode: {episode} - count: {counter} - reward: {reward}")
                if done or truncated: 
                    break
                if run_wandb:
                    wandb.log({"reward": reward})
            if run_wandb: wandb.finish()
            episode += 1
            reward_list.append(reward)
        self.scatter_plot_reward(reward_list)

    def scatter_plot_reward(self, reward_list):
        plt.subplot()
        plt.scatter(range(len(reward_list)), reward_list)
        plt.title(f"Reward Scatter {self.name}")
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.grid()
        path = "experiments/"+self.name+"/"
        os.makedirs(path, exist_ok=True)
        plt.savefig(path+self.name+"_scatter.png")
        plt.close()

    def config_wandb(self, project_name, name):
        config={}
        if name != '':
            return wandb.init(project=project_name, name=name, config=config)
        return wandb.init(project=project_name, config=config)

    def preprocess_obs(self, obs_list):
        obs_list = DataHandler().to_greyscale(obs_list)
        obs_list = DataHandler().normalizing(obs_list)
        obs_list = np.expand_dims(obs_list, axis=0)
        obs_list = DataHandler().stack_with_previous(obs_list)
        return obs_list


def get_dim(x, y):
    pass

if __name__ == '__main__':
    versions = [f for f in os.listdir("experiments") if ('version' in f and '.pkl' not in f)]
    versions = ["version_3", "version_4"]
    for version in sorted(versions):
        params = Params("experiments/" + version + "/params.json")

        n_epoch = params.n_epoch
        lrate = params.lrate
        device = "cpu"
        n_hidden = params.n_hidden
        batch_size = params.batch_size
        n_T = params.n_T
        net_type = params.net_type
        drop_prob = params.drop_prob
        extra_diffusion_steps = params.extra_diffusion_steps
        embed_dim = params.embed_dim
        x_shape = (96, 96, 4)
        y_dim = 3

        env = CarRacing(render_mode="human") 
        nn_model = Model_cnn_mlp(
            x_shape, n_hidden, y_dim, embed_dim=embed_dim, net_type=net_type
        ).to(device)

        model = Model_Cond_Diffusion(
            nn_model,
            betas=(1e-4, 0.02),
            n_T=n_T,
            device=device,
            x_dim=x_shape,
            y_dim=y_dim,
            drop_prob=drop_prob,
            guide_w=0.0,)

        # model.load_state_dict(torch.load("model_casa2.pkl"))
        model.load_state_dict(torch.load("experiments/"+version+"/"+version+".pkl",
                            map_location=torch.device('cpu')))

        stop = 1
        tester = Tester(model, env, render=True)
        try:
            tester.run(run_wandb=False, name=version)
        except Exception as exception:
            print("---------------------------------------------------")
            print(f"The {version} couldn't be trained due to ")
            print(f'{exception}')
            print("---------------------------------------------------")
            continue