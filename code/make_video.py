import cv2
import numpy as np
from data_preprocessing import DataHandler

class VideoMaker(DataHandler):
    def __init__(self):
        """
        Initializes a VideoMaker object.

        Parameters:
        - path (str): The path to the input data.

        Returns:
        - None
        """
        super(VideoMaker).__init__()

    def load_array(self, path):
        """
        Loads data from the specified path.

        Returns:
        - numpy.ndarray: The loaded data as a NumPy array.
        """
        return self.load_data(path)

    @staticmethod
    def save_record(frames, name='', origin='human'):
        """
        Saves a sequence of frames as an MP4 video.

        Parameters:
        - frames (list): A list of frames (NumPy arrays) to be saved as a video.
        - name (str, optional): The name of the output video file (default is 'output_video.mp4').

        Returns:
        - None
        """
        name = name if name != '' else 'output_video_dataset_2'
        name = name + '.mp4'
        fps = 30
        if origin == 'human':
            shape = (96, 96)
        elif origin == 'ppo':
            shape = (84, 84)
        out = cv2.VideoWriter(name,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              shape)
        counter = 0
        for img in frames:
            if origin == 'ppo':
                img = np.transpose(img, axes=(1, 2, 0))
                img = img[:,:,0]
            image_array = img.astype(np.uint8)
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            out.write(image)
        out.release()


if __name__ == '__main__':
    video_maker = VideoMaker()
    array  = video_maker.load_array('Datasets/tutorial_human_expert_0_top_20/states.npy')
    video_maker.save_record(array, name='Datasets/tutorial_human_expert_0_top_20/video', origin='human')
    stop = 1

