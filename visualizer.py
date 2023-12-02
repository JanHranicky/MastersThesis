import tkinter as tk
from tkinter import filedialog as fd
from ui import display
from core import utils
from tkinter import messagebox
import tensorflow as tf
import keras
import numpy as np

def load_model(display):
    keras_files = [('Keras models','*.keras')]
    filename = fd.askopenfilename(filetypes=keras_files)
    if not filename:
        print('no file selected')
        return
    model = keras.models.load_model(filename)
    display.set_model(model)
    print(display.model)

def load_image(display):
    image_files = [('Png images','*.png')]
    filename = fd.askopenfilename(filetypes=image_files)
    if not filename:
        print('no file selected')
        return
    image_data = utils.open_image(filename)
    width,height = image_data.shape[0],image_data.shape[1]
    image_data = utils.generate_empty_img(width,height)
    display.set_image(width,height)

step = 0
def model_step(display):
    if display.model_input is None:
        x = utils.init_batch(1,display.x,display.y,display.model.channel_n)
    else:
        x = display.model_input
    
    x = display.model(x)
    display.model_input = x
    print(f'step: {step}')
    print(utils.tf2pil(x[0].numpy()))
    
    display.display_image(utils.tf2pil(x[0].numpy()))
    #x = np.clip(x[0].numpy().astype(int), 0, 255)
    #display.display_image(utils.to_rgb(x))
    #print('one model step')


def tksleep(t):
    'emulating time.sleep(seconds)'
    ms = int(t*1000)
    root = tk._get_default_root('sleep')
    var = tk.IntVar(root)
    root.after(ms, var.set, 1)
    root.wait_variable(var)

animate = False
step = 0

def play(d):
    global animate
    animate = True
    while animate:
        model_step(d)
        tksleep(0.05)
        
        global step
        step += 1

def stop():
    print('in stop()')
    global animate
    animate = False

def reset(d):
    global step
    step = 0
    d.model_input = None
    d.set_image(d.x,d.y)
    
def main():
    canvas_x,canvas_y = 1000,1000

    root = tk.Tk()
    root.title("CA state visualization")

    button_frame = tk.Frame(root)
    button_frame.pack() 

    load_model_button = tk.Button(button_frame, text="Load Model", command=lambda: load_model(d))
    load_model_button.pack(side=tk.LEFT)
    load_img_button = tk.Button(button_frame, text="Load Image", command=lambda: load_image(d))
    load_img_button.pack(side=tk.LEFT)
    
    previous_button = tk.Button(button_frame, text="Previous", )
    previous_button.pack(side=tk.LEFT)
    next_button = tk.Button(button_frame, text="Next",command=lambda : model_step(d))
    next_button.pack(side=tk.LEFT)
    
    play_button = tk.Button(button_frame, text="Play", command= lambda: play(d))
    play_button.pack(side=tk.LEFT)
    stop_button = tk.Button(button_frame, text="Stop",command=lambda : stop())
    stop_button.pack(side=tk.LEFT)
    reset_button = tk.Button(button_frame, text="Reset",command=lambda : reset(d))
    reset_button.pack(side=tk.LEFT)
    
    d = display.Display(canvas_x,canvas_y)
    
    root.mainloop()
    
if __name__ == '__main__':
    main()
    