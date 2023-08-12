import cv2
import torch
import wandb
import numpy as np
from cart_racing_v2 import CarRacing
from data_preprocessing import DataHandler
from record_observations import RecordObservations
from models import Model_Cond_Diffusion, Model_cnn_mlp


class Tester(RecordObservations):
    def __init__(self, model, env, render=True, device="cpu"):
        super(Tester).__init__()
        self.model = model
        self.env = env
        self.render = render
        self.device = device

    def run(self, run_wandb, name=''):
        if run_wandb:
            self.config_wandb(project_name="car-racing-diffuser-bc-v2", name=name)
        obs, _ = self.env.reset()
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
                ).unsqueeze(0)
            action = self.model.sample(obs_tensor).to(self.device)
            obs, new_reward, done, _, truncated = self.env.step(action.detach().cpu().numpy()[0])
            self.array_to_img(obs, action, frame=counter)
            reward += new_reward
            counter += 1
            print(reward)
            if done or truncated: 
                break
            if run_wandb:
                wandb.log({"reward": reward})
        if run_wandb: wandb.finish()


    def config_wandb(self, project_name, name):
        config={}
        if name != '':
            return wandb.init(project=project_name, name=name, config=config)
        return wandb.init(project=project_name, config=config)

    def preprocess_obs(self, obs_list):
        obs_list = DataHandler().to_greyscale(obs_list)
        obs_list = DataHandler().normalizing(obs_list)
        obs_list = np.expand_dims(obs_list, -1)
        # obs_list = obs_list/255.0
        # dim = (32, 32)
        # obs_list_resized = np.array([cv2.resize(obs_list[i], dim, interpolation=cv2.INTER_AREA) for i in range(len(obs_list))])
        return obs_list


def get_dim(x, y):
    pass

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
    x_shape = (96, 96, 1)
    y_dim = 3

    env = CarRacing(render_mode="human") 
    nn_model = Model_cnn_mlp(
        x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type
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

    model.load_state_dict(torch.load("model_casa2.pkl"))

    stop = 1
    tester = Tester(model, env, render=True)
    tester.run(run_wandb=False)