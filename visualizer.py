import tkinter as tk
from tkinter import filedialog as fd
from ui import display, output
from core import utils
import keras
import tensorflow as tf 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

_BUTTONS_DISABLED = True

def load_model(display, buttons):
    filename = load_file([('Keras models', '*.keras')])
    if not filename:
        print("No file provided.\n")
        return
    try:
        print(f"Trying to load model {filename}\n")
        model = keras.models.load_model(filename)
        display.set_model(model)
        
        print(f"Successfully loaded model. Number of channels: {model.channel_n}, Trained img: {model.target_img_path}\n")
                
        load_image(display, model.target_img_path)
        
        enable_buttons(buttons)
    except Exception as e:
        print(f"Failed to load the model {filename}. Encountered exception: {e}\n")
        disable_buttons(buttons)

def load_file(types):
    filename = fd.askopenfilename(filetypes=types)
    if not filename:
        print('no file selected')
        return None
    return filename 

def load_image(display, file_name=None):
    if not file_name:
        file_name = load_file([('Png images', '*.png')])
        if not file_name:
            print("No file provided.\n")
            return
    try:
        print(f"Trying to load image: {file_name}")
        image_data = Image.open(file_name).convert("RGB")
            
        height, width = image_data.size
        display.set_image(width, height)
        
        if display.model and not display.model.full_range or display.model.channel_n < 3:
            display.gt_tf = utils.img_to_discrete_tensor(image_data,display.model.states)
            display.color_dict = utils.extract_color_dict(image_data,display.gt_tf)
            #display.color_dict = utils.color_dict(file_name.split('/')[-1].split('.')[0])
            
            print(f"initialized to {display.color_dict}")

        print(f"Successfully loaded the image.\n")
    except Exception as e:  
        print(f"Failed to load image {file_name}. Encountered exception: {e}\n")

def enable_buttons(buttons):
    for button in buttons:
        button.config(state=tk.NORMAL)
    global _BUTTONS_DISABLED
    _BUTTONS_DISABLED = False

def disable_buttons(buttons):
    for button in buttons:
        button.config(state=tk.DISABLED)
    global _BUTTONS_DISABLED
    _BUTTONS_DISABLED = True

i = 0
def model_step(display):
    global i
    if display.model_input is None:
        x = utils.init_batch(1, display.x, display.y, display.model.channel_n)
    else:
        x = display.model_input
        
    x = display.model(x)
    display.model_input = x
    if not display.model.full_range or display.model.channel_n < 3:
        f = Image.fromarray(np.uint8(x[0][:,:,0].numpy()),mode="L")
        display_img = utils.grayscale_to_rgb(f,display.color_dict)
        if i == 20:
            display_img.save('visualize_color.png')
    else:
        display_img = utils.tf2pil(x[0].numpy())
    i += 1
    display.forward(display_img)

def tksleep(t):
    'emulating time.sleep(seconds)'
    ms = int(t*1000)
    root = tk._get_default_root('sleep')
    var = tk.IntVar(root)
    root.after(ms, var.set, 1)
    root.wait_variable(var)


animate = False
def play(button, d):
    global animate
    animate = not animate
    
    button.configure(text="Stop" if animate else "Play")
    while animate:
        model_step(d)
        tksleep(0.05)

def stop():
    global animate
    animate = False

def reset(d):
    d.reset()
    
def step_back(d):
    d.back()
  
def keypress(event, key):
    global d,_BUTTONS_DISABLED
    if _BUTTONS_DISABLED:
        return
    if key == "left":
        step_back(d)
    elif key == "right":
        model_step(d)
    elif key == "space":
        global play_button
        play(play_button, d)


def main():
    canvas_x, canvas_y = 1000, 1000
    root = tk.Tk()
    root.title("CA state visualization")
    root.geometry("1600x1080")
    root.minsize(1600, 1080)
    root.maxsize(1600, 1080)
    root.resizable(0, 0)
    
    global play_button
    previous_button, next_button, play_button, reset_button = None, None, None, None
    
    def create_button(text, command, state=tk.NORMAL):
        return tk.Button(btn_grp, text=text, command=command, state=state)

    btn_grp = tk.Frame(root, pady=10)
    btn_grp.grid()

    load_model_button = create_button("Load Model", lambda: load_model(d, [previous_button, next_button, play_button, reset_button]))
    load_model_button.grid(row=0, column=0)

    #load_img_button = create_button("Load Image", lambda: load_image(d))
    #load_img_button.grid(row=0, column=1)
    
    space_1 = tk.Label(btn_grp, text="   ")
    space_1.grid(row=0, column=2)

    previous_button = create_button("Previous", lambda: step_back(d), state=tk.DISABLED)
    previous_button.grid(row=0, column=3)

    next_button = create_button("Next", lambda: model_step(d), state=tk.DISABLED)
    next_button.grid(row=0, column=4)
    
    space_2 = tk.Label(btn_grp, text="   ")
    space_2.grid(row=0, column=5)
    
    play_button = create_button("Play", lambda: play(play_button, d), state=tk.DISABLED)
    play_button.grid(row=0, column=6)

    reset_button = create_button("Reset", lambda: reset(d), state=tk.DISABLED)
    reset_button.grid(row=0, column=7)
    
    textbox = tk.Text(root, wrap='word')
    textbox.grid(row=3, column=1, sticky="nsew")
    redirected_output = output.RedirectedOutput(textbox)
    
    step_cnt = display.StepCounter(root)
    step_cnt.grid(row=2, column=1, sticky="nsew")
        
    global d
    d = display.Display(canvas_x, canvas_y, step_cnt, root)
    root.bind('<Left>', func=lambda event: keypress(event, key="left"))
    root.bind('<Right>', func=lambda event: keypress(event, key="right"))
    root.bind('<space>', func=lambda event: keypress(event, key="space"))
    
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=3)
    root.columnconfigure(1, weight=1)
    
    root.mainloop()

if __name__ == '__main__':
    main()
