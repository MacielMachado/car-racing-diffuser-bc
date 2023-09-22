from data_preprocessing import DataHandler
import matplotlib.pyplot as plt
import numpy as np

class ActionsAnalyzer(DataHandler):
    def __init__(self, path):
        super(ActionsAnalyzer).__init__()
        self.path = path

    def load_action(self):
        """
        Loads data from the specified path.

        Returns:
        - numpy.ndarray: The loaded data as a NumPy array.
        """
        return self.load_data(self.path)
    
    def make_histograms(self, actions):   
        """
        Plots histograms for each dimension of a 3D NumPy array.

        Parameters:
        - actions (numpy.ndarray): The 3D NumPy array with dimensions (N, M, 3).

        Returns:
        - None
        """
        if actions.shape[-1] != 3:
            raise ValueError("The input array must have three dimensions.")

        # Separe as três dimensões do array
        dim1 = actions[:, 0]
        dim2 = actions[:, 1]*4.5
        dim3 = actions[:, 2]

        # Plote os histogramas
        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.hist(dim1, bins=50, color='r', alpha=0.7)
        plt.text(min(dim1), 0, f"mean: {np.mean(dim1):.4f}\nstd deviation: {np.std(dim1):.4f}")
        plt.title("Steering Wheel Direction")

        plt.subplot(132)
        plt.hist(dim2, bins=50, color='g', alpha=0.7)
        plt.text(min(dim2), 0, f"mean: {np.mean(dim2):.4f}\nstd deviation: {np.std(dim2):.4f}")
        plt.title("Gas")

        plt.subplot(133)
        plt.hist(dim3, bins=50, color='b', alpha=0.7)
        plt.text(min(dim3), 0, f"mean: {np.mean(dim3):.4f}\nstd deviation: {np.std(dim3):.4f}")
        plt.title("Brake")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    analyzer = ActionsAnalyzer(path='data/actions_4_5_20.npy')
    actions = analyzer.load_action()
    analyzer.make_histograms(actions)