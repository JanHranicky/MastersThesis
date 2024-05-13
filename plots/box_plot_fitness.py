import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import plot_utils 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plots fitness box plots of DNCA model trained by differential evolution algorithm SHADE')

    # Add arguments
    parser.add_argument('-f', '--folder', type=str, help='Folder which will be used to plot the data', default='../checkpoints/')
    parser.add_argument('-p', '--plot_dir', type=str, help='Folder in which the plots will be saved', default='./box_plots/')

    return parser.parse_args()

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

def process_file(loss_dict, file_path):
    if '.npy' not in file_path:
        return loss_dict

    print(f"Processing directory: {file_path}")
    try:
        has_run = "run" in file_path.split('/')[-2]
        folder_name = file_path.split('/')[-3] if has_run else file_path.split('/')[-2]
        algorithm = folder_name.split('+')[1]
        
        if algorithm == 'gd':
            print(f"Folder: {file_path}. Does not contain SHADE algorithm outputs, skipping.\n")
            return loss_dict 
        
        return plot_utils.process_de_file(loss_dict,file_path)
    except:
        print(f'Folder: {file_path}. Is named in an incorrect format, skipping.\n')
        return loss_dict


def plot_box_plot(name,directory,best_fitness_values, title, x_label, y_label):
    """
    Generate a box plot of the best fitness values for each iteration of differential evolution runs.

    Parameters:
    - best_fitness_values: NumPy array containing best fitness values for each iteration.
    - title: Title for the plot (default is 'Box Plot of Differential Evolution Runs').
    """
    fig, ax = plt.subplots()

    ax.boxplot(best_fitness_values, showfliers=False)  # Set showfliers to True if you want to show outliers

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    
    plt.xticks([], [])
    
    ax.set_title(title, loc='center', wrap=True)

    plt.savefig(f'{directory}/{name}.png')

def box_plot_de_fitness(loss_dict,dir_path):
    for img in loss_dict:
        for o in loss_dict[img]:
            for state in loss_dict[img][o]:
                for channel in loss_dict[img][o][state]:
                    for interval in loss_dict[img][o][state][channel]:
                        for iteration in loss_dict[img][o][state][channel][interval]:
                            for size in loss_dict[img][o][state][channel][interval][iteration]:
                                min_arr = []
                                for run in loss_dict[img][o][state][channel][interval][iteration][size]:
                                    fitness_dict = loss_dict[img][o][state][channel][interval][iteration][size][run]
                                    min_fitness = np.min(fitness_dict)
                                    min_arr.append(min_fitness)
                                plot_box_plot(
                                    f'box_plot+img_{img}+operator_{o}+state_{state}+channel_{channel}+interval_{interval}+iteration_{iteration}+pop_size_{size}+run_nums_{len(min_arr)}',
                                    dir_path,
                                    min_arr,
                                    f'vzor: {img}, operator: {o}, počet stavů: {state}, počet kanálů: {channel}, interval: {interval}, max. iterace: {iteration}, vel. populace: {size}, počet běhů: {len(min_arr)}',
                                    f'Počet běhů {len(min_arr)}',
                                    'Nejlepší hodnota L2'
                                    )

def box_plot_fitness(directory_path,data):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    box_plot_de_fitness(data['shade'],directory_path)

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
    # Check if the directory exists
    if os.path.exists(args.folder) and os.path.isdir(args.folder):
        print(f'Iterating directory: {args.folder}')
        fitness_dict = plot_utils.iterate_directory(args.folder,process_file,{})
        
        box_plot_fitness(args.plot_dir,fitness_dict)
        print(fitness_dict)
    else:
        print("Invalid directory path: " + args.folder)
