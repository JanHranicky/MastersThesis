import os
import numpy as np

def iterate_directory(root_dir,process_func,dict):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            dict = process_func(dict,file_path)
    return dict

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
            if key not in current_dict:
                current_dict[key] = value if value is not None else []
        else:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

def process_de_file(loss_dict,file_path):
    try:
        has_run = "run" in file_path.split('/')[-2]
        folder_name = file_path.split('/')[-3] if has_run else file_path.split('/')[-2]
        folder_name_split = folder_name.split('+')
        
        img = folder_name_split[-1]
        operator = folder_name_split[-2]
        states = folder_name_split[2].split('_')[1]
        channels = folder_name_split[3].split('_')[1]
        train_interval = folder_name_split[4].split('_')[2]
        iterations = folder_name_split[5].split('_')[1]
        pop_size = folder_name_split[6].split('_')[2]
        run = '1' if not has_run else  file_path.split('/')[-2].split('_')[1]
        
        keys = ['shade', img, operator, states, channels, train_interval, iterations, pop_size,run]
        add_to_nested_dict(loss_dict, keys,[])

        loss_dict['shade'][img][operator][states][channels][train_interval][iterations][pop_size][run].append(np.load(file_path))
        
        print("\n")
        return loss_dict
    except:
        print(f'Folder: {file_path}. Is named in an incorrect format, skipping.\n')
        return loss_dict

def process_gd_file(loss_dict,file_path):
    try:
        has_run = "run" in file_path.split('/')[-2]
        folder_name = file_path.split('/')[-3] if has_run else file_path.split('/')[-2]
        folder_name_split = folder_name.split('+')
        
        states = folder_name_split[2].split('_')[1]
        channels = folder_name_split[3].split('_')[1]
        train_interval = folder_name_split[4].split('_')[2]
        img = folder_name_split[-1]
        run = '1' if not has_run else file_path.split('/')[-2].split('_')[1]       
        
        try:
            has_run_key = ('1' if not run else run) in loss_dict['gd'][img][states][channels][train_interval]
        except:
            has_run_key = False #dictionary has not yet been constructed
            
        run = run if not has_run_key else max(list(map(int, loss_dict['gd'][img][states][channels][train_interval].keys())))
        
         
        keys = ['gd', img, states, channels, train_interval,run]
        add_to_nested_dict(loss_dict, keys,[])
        print(loss_dict['gd'][img][states][channels][train_interval])

        loss_dict['gd'][img][states][channels][train_interval][run].append(np.load(file_path))
        return loss_dict
    except:
        print(f'Folder: {file_path}. Is named in an incorrect format, skipping.')
        return loss_dict




