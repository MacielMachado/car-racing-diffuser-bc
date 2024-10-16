import os
import sys
import utils
import argparse
import numpy as np
from trainer import Trainer

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='/home/casa/projects/bruno/datasets/car-racing/ppo_dataset/tutorial_ppo_expert_68',
                    help='Directory containing the dataset')
parser.add_argument('--data_origin', default='ppo',
                    help='Inform if the data is from a "human" expert or a "ppo" expert')

def launch_training_job(parent_dir, data_dir, job_name, params, dataset_origin):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    trainer_instance = Trainer(n_epoch=params.n_epoch,
                               lrate=params.lrate,
                               device=params.device,
                               n_hidden=params.n_hidden,
                               batch_size=params.batch_size,
                               n_T=params.n_T,
                               net_type=params.net_type,
                               drop_prob=params.drop_prob,
                               extra_diffusion_steps=params.extra_diffusion_steps,
                               embed_dim=params.embed_dim,
                               guide_w=params.guide_w,
                               betas=(1e-4, 0.02),
                               dataset_path=data_dir,
                               name=job_name,
                               param_search=True,
                               run_wandb=False,
                               record_run=True,
                               embedding=params.embedding,
                               dataset_origin=dataset_origin)
    trainer_instance.main()
   

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'default/params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)

    # data_dir_list = [r'Datasets/human/tutorial_human_expert_1/',
    #                  r'Datasets/human/tutorial_human_expert_2/',
    #                  r'Datasets/human/tutorial_human_expert_0_top_20',
    #                  r'Datasets/ppo/tutorial_ppo_expert_68/',]

    data_dir_list = [r'Datasets/human/tutorial_human_expert_1/']
    n_epoch_list = [750]
    lrate_list = [1e-4, 1e-5]
    device_list = ["mps"]
    n_hidden_list = [128, 512]
    batch_size_list = [32, 512]
    n_T_list = [20, 50]
    net_type_list = ["transformer", "fc"]
    drop_prob_list = [0.0]
    extra_diffusion_steps_list = [16]
    embed_dim_list = [128]
    guide_w_list = [0.0]
    betas_list = [[1e-4, 0.02]]

    params_grid = np.meshgrid(n_epoch_list,
                             lrate_list,
                             device_list,
                             n_hidden_list,
                             batch_size_list,
                             n_T_list,
                             net_type_list,
                             drop_prob_list,
                             extra_diffusion_steps_list,
                             embed_dim_list,
                             guide_w_list,
                             data_dir_list)
    params_list = np.array(params_grid).T.reshape(-1, len(params_grid))
    
    params = utils.Params(json_path)

    for index, item in enumerate(params_list):
        params.n_epoch=int(item[0])
        params.lrate=float(item[1])
        params.device=item[2]
        params.n_hidden=int(item[3])
        params.batch_size=int(item[4])
        params.n_T=int(item[5])
        params.net_type=item[6]
        params.drop_prob=float(item[7])
        params.extra_diffusion_steps=int(item[8])
        params.embed_dim=int(item[9])
        params.guide_w=float(item[10])
        job_name = str(item[11]).split(os.sep)[2]+f"_version_{index}"
        model_dir = os.path.join(args.parent_dir, job_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        utils.set_logger(os.path.join(args.parent_dir, job_name, 'train.log'))
        
        try:
            launch_training_job(args.parent_dir,
                                str(item[11]),
                                job_name,
                                params,
                                str(item[11]).split(os.sep)[1])
            
        except Exception as exception:
            print("---------------------------------------------------")
            print(f"The {job_name} couldn't be trained due to ")
            print(f'{exception}')
            print("---------------------------------------------------")
            continue
    
    
