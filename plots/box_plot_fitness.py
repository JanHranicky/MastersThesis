import os
import numpy as np
import matplotlib.pyplot as plt

def process_file(file_path):
    # Add your processing logic here
    # Example: print the file path
    if '.npy' in file_path:
        #print("Processing file:", file_path)
        return np.load(file_path)

def add_file_entry(dictionary, loss_arr,file_path):
    name_split = file_path.split('/')
    settings = name_split[-3].split('+')
    
    mutation_weight = settings[-2].split('_')[-1]
    print(mutation_weight)
    if mutation_weight not in dictionary:
        dictionary[mutation_weight] = [(loss_arr.shape,loss_arr)]
    else:
        dictionary[mutation_weight].append((loss_arr.shape,loss_arr))
    
    return dictionary
    
def iterate_directory(runs, root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            loss_arr = process_file(file_path)
            if loss_arr is not None:
                runs = add_file_entry(runs, loss_arr,file_path)
    return runs


def plot_box_plot(mutation,best_fitness_values, title='Box Plot of Differential Evolution Runs'):
    """
    Generate a box plot of the best fitness values for each iteration of differential evolution runs.

    Parameters:
    - best_fitness_values: NumPy array containing best fitness values for each iteration.
    - title: Title for the plot (default is 'Box Plot of Differential Evolution Runs').
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a box plot
    ax.boxplot(best_fitness_values, showfliers=False)  # Set showfliers to True if you want to show outliers

    # Set labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Fitness Value')
    ax.set_title(title)

    plt.savefig(f'box_plot_{mutation}.png')

# Replace 'your_directory_path' with the path to the directory you want to iterate
directory_paths = ['../checkpoints/experiment_1_mutation_settings/']




# Check if the directory exists
runs = {}
for directory_path in directory_paths:
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        runs = iterate_directory(runs, directory_path)

runs = {k: runs[k] for k in sorted(runs)}
averaged_runs = {}
fitness_list = []
mutation_parameter_list = []
for r in runs:
    print(f'plotting box plot for mutation {r}')
    concat = [np.min(r) for (_,r) in runs[r]]
    fitness_list.append(concat)
    mutation_parameter_list.append(r)
    #plot_box_plot(r,concat)



# Create a figure and axis
fig, ax = plt.subplots()

# Plot a boxplot for each mutation setting
ax.boxplot(fitness_list, labels=[f'{m}' for m in mutation_parameter_list], showfliers=False)

# Set labels and title
ax.set_xlabel('Nastavení parametru mutace')
ax.set_ylabel('Počet odlišných pixelů')
ax.set_title('Boxplot pro různá nastavená mutace')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45, ha='right')   
plt.savefig(f'experiment_1.png')