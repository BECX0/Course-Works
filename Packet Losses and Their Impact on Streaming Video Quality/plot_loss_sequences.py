import numpy as np
import matplotlib.pyplot as plt
def plot_loss_Sequences(data):
    data_np = np.array(data, dtype=np.uint8)
    def plot_data(data, title, ax):
        cmap = plt.get_cmap('Blues')
        cmap.set_under('white')
        ax.imshow(data.reshape(1, -1), aspect='auto', cmap=cmap, vmin=0.1)
        ax.set_title(title)
        ax.get_yaxis().set_visible(False)
    fig, ax = plt.subplots(figsize=(10, 2))
    plot_data(data_np, 'Random Binary Data', ax)
    plt.show()
