import os
import numpy as np
import matplotlib.pyplot as plt

def process_file(file_path):
    # Add your processing logic here
    # Example: print the file path
    if '.npy' in file_path:
        #print("Processing file:", file_path)
        return np.load(file_path)

def iterate_directory(root_dir):
    loss_values = []
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            loss_arr = process_file(file_path)
            if loss_arr is not None:
                loss_values.append(process_file(file_path))
    return loss_values


def plot_convergence(directory_path,data):
    fig, ax = plt.subplots()
    for i, run in enumerate(data):
        ax.plot(run, label=f'Run {i+1}')
        
    ax.set_xlabel('Generace')
    ax.set_ylabel('Nejlepší hodnota fitness')
    ax.set_title('Graf konvergence modelu s jednou vrstvou pro francouzskou vlajku 15x15 a různé nastavení parametrů mutace')
    #ax.legend()
    plt.savefig('convergence_experiment_1.png')


# Replace 'your_directory_path' with the path to the directory you want to iterate
#directory_path = '../checkpoints/convergence_avail_two_channels_scratch/'
directory_path = '../checkpoints/experiment_1_mutation_settings/'

# Check if the directory exists
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    loss_values = iterate_directory(directory_path)
    plot_convergence(directory_path,loss_values)
else:
    print("Invalid directory path.")
