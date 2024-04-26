import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from textwrap import wrap

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plots convergence of DNCA model')

    # Add arguments
    parser.add_argument('-f', '--folder', type=str, help='Folder which will be used to plot the data', default='../checkpoints/')
    parser.add_argument('-p', '--plot_dir', type=str, help='Folder which will be used to plot the data', default='./convergence_plots/')

    return parser.parse_args()

def process_file(loss_dict, file_path):
    if '.npy' not in file_path:
        return loss_dict

    print(f"Processing directory: {file_path}")
    try:
        has_run = "run" in file_path.split('/')[-1]
        folder_name = file_path.split('/')[-3] if has_run else file_path.split('/')[-2]
        algorithm = folder_name.split('+')[1]
        
        return save_file_entry(loss_dict,algorithm,file_path)
    except:
        print(f'Folder: {file_path}. Is named in an incorrect format, skipping.')
        return loss_dict
    
def iterate_directory(root_dir):
    loss_dict = {}
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            loss_dict = process_file(loss_dict,file_path)
    return loss_dict

def save_file_entry(loss_dict,algorithm,file_path):
    if algorithm == "shade":
        if algorithm not in loss_dict:
            loss_dict[algorithm] = {}
        return process_de_file(loss_dict,file_path)
    elif algorithm == "gd":
        if algorithm not in loss_dict:
            loss_dict[algorithm] = {}
        return process_gd_file(loss_dict,file_path)
    else:
        return loss_dict

def add_to_nested_dict(dictionary, keys, value=None):
    """
    Add a nested dictionary structure to the given dictionary if it doesn't already exist,
    with the most nested item being a list instead of a dictionary.
    :param dictionary: The dictionary to which the nested structure will be added.
    :param keys: A list of keys representing the nested structure.
    :param value: Optional initial value for the most nested item.
    """
    current_dict = dictionary
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            current_dict[key] = value if value is not None else []
        else:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

def process_de_file(loss_dict,file_path):
    try:
        has_run = "run" in file_path.split('/')[-1]
        folder_name = file_path.split('/')[-3] if has_run else file_path.split('/')[-2]
        folder_name_split = folder_name.split('+')
        
        img = folder_name_split[-1]
        operator = folder_name_split[-2]
        states = folder_name_split[2].split('_')[1]
        channels = folder_name_split[3].split('_')[1]
        train_interval = folder_name_split[4].split('_')[2]
        iterations = folder_name_split[5].split('_')[1]
        pop_size = folder_name_split[6].split('_')[2]
                
        keys = ['shade', img, operator, states, channels, train_interval, iterations, pop_size]
        add_to_nested_dict(loss_dict, keys,[])

        loss_dict['shade'][img][operator][states][channels][train_interval][iterations][pop_size].append(np.load(file_path))
        return loss_dict
    except:
        print(f'Folder: {file_path}. Is named in an incorrect format, skipping.')
        return loss_dict

def process_gd_file(loss_dict,file_path):
    try:
        has_run = "run" in file_path.split('/')[-1]
        folder_name = file_path.split('/')[-3] if has_run else file_path.split('/')[-2]
        folder_name_split = folder_name.split('+')
        
        states = folder_name_split[2].split('_')[1]
        channels = folder_name_split[3].split('_')[1]
        train_interval = folder_name_split[4].split('_')[2]
        img = folder_name_split[-1]
                
        keys = ['gd', img, states, channels, train_interval]
        add_to_nested_dict(loss_dict, keys,[])

        loss_dict['gd'][img][states][channels][train_interval].append(np.load(file_path))
        return loss_dict
    except:
        print(f'Folder: {file_path}. Is named in an incorrect format, skipping.')
        return loss_dict


def plot_gd_convergence(loss_dict,dir_path):
    for img in loss_dict:
        for state in loss_dict[img]:
            for channel in loss_dict[img][state]:
                for interval in loss_dict[img][state][channel]:
                            make_plot(
                                loss_dict[img][state][channel][interval],
                                'Epochy',
                                'Nejlepší hodnota MSE',
                                f'vzor: {img}, počet stavů: {state}, počet kanálů: {channel}, interval: {interval}',
                                f'convergence+img_{img}+state_{state}+channel_{channel}+interval_{interval}',
                                dir_path
                                )

def plot_de_convergence(loss_dict,dir_path):
    for img in loss_dict:
        for o in loss_dict[img]:
            for state in loss_dict[img][o]:
                for channel in loss_dict[img][o][state]:
                    for interval in loss_dict[img][o][state][channel]:
                        for iteration in loss_dict[img][o][state][channel][interval]:
                            for size in loss_dict[img][o][state][channel][interval][iteration]:
                                make_plot(
                                    loss_dict[img][o][state][channel][interval][iteration][size],
                                    'Generace',
                                    'Nejlepší hodnota L2',
                                    f'vzor: {img}, operator: {o}, počet stavů: {state}, počet kanálů: {channel}, interval: {interval}, max. iterace: {iteration}, vel. populace: {size}',
                                    f'convergence+img_{img}+operator_{o}+state_{state}+channel_{channel}+interval_{interval}+iteration_{iteration}+pop_size_{size}',
                                    dir_path
                                    )
                  
def make_plot(data,x_label,y_label,title, file_name,dir_path):
    fig, ax = plt.subplots()
    for i, run in enumerate(data):
        ax.plot(run, label=f'Run {i+1}')
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc='center', wrap=True)
    #ax.legend()
    plt.savefig(f'{dir_path}/{file_name}.png')
    
def plot_convergence(directory_path,data):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    plot_de_convergence(data['shade'],directory_path)
    plot_gd_convergence(data['gd'],directory_path)

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
    # Check if the directory exists
    if os.path.exists(args.folder) and os.path.isdir(args.folder):
        print(f'Iterating directory: {args.folder}')
        loss_dict = iterate_directory(args.folder)
        print(loss_dict)
        
        plot_convergence(args.plot_dir,loss_dict)
    else:
        print("Invalid directory path: " + args.folder)
    exit()




