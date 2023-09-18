import cv2
import numpy as np
from data_preprocessing import DataHandler

class VideoMaker(DataHandler):
    def __init__(self, path):
        """
        Initializes a VideoMaker object.

        Parameters:
        - path (str): The path to the input data.

        Returns:
        - None
        """
        super(VideoMaker).__init__()
        self.path = path

    def load_array(self):
        """
        Loads data from the specified path.

        Returns:
        - numpy.ndarray: The loaded data as a NumPy array.
        """
        return self.load_data(self.path)

    @staticmethod
    def save_record(frames, name=''):
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
        out = cv2.VideoWriter(name,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (96, 96))
        counter = 0
        for img in frames:
            image_array = img.astype(np.uint8)
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            out.write(image)
        out.release()


if __name__ == '__main__':
    video_maker = VideoMaker('tutorial_2/states_expert.npy')
    array  = video_maker.load_array()
    video_maker.save_record(array)
    stop = 1

