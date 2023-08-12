import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob
import os

class RecordObservations():
    def __init__(self, frame_size=(96, 96)):
        self.actions = []
        self.observations = []
        self.path = 'recording/'
        self.frame_size = frame_size

    def save_frame(self, observation, action):
        self.actions.append(action.copy())
        self.observations.append(observation.copy())

    def reset_recording(self):
        self.actions = []
        self.observations = []

    def save_numpy_array(self, name=''):
        np.save(self.path+'states_'+'.npy', 
                self.states)
        np.save(self.path+'actions_'+'.npy', 
                self.actions)
        
    def array_to_img(self, obs, action, frame=''):
        im = Image.fromarray(obs)
        I1 = ImageDraw.Draw(im)
        action = list(action.detach().cpu().numpy()[0])
        text = f'left:{action[0]:.3f}\nacc:{action[1]:.3f}\nacc:{action[2]:.3f}'
        I1.text((10, 10),
                text,
                fill=(255, 0, 0),
                align='left')
        os.makedirs('frame', exist_ok=True)
        im_name = 'frame/observation_'+str(frame)+'.jpeg'
        im.save(im_name)
        
    def load_npy(self, dataset_path):
        return np.load(dataset_path, allow_pickle=True)

    def save_record(self, frames, name=''):
        name = name if name != '' else 'output_video'
        name = name + '.mp4'
        out = cv2.VideoWriter(name,
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              60,
                              self.frame_size)
        for img in frames:
            out.write(img)

        out.write(img)

    def save_record_image_folder(self, dataset_path):
        out = cv2.VideoWriter('output_video_images.mp4',
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              60,
                              self.frame_size)

        for filename in sorted(os.listdir(dataset_path)):
            img = cv2.imread(dataset_path+filename)
            out.write(img)

        out.release()

if __name__ == '__main__':
    recorder = RecordObservations()
    dataset_path = r'frame/'
    recorder.save_record_image_folder(dataset_path)