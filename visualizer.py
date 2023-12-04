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

def model_step(display):
    if display.model_input is None:
        x = utils.init_batch(1,display.x,display.y,display.model.channel_n)
    else:
        x = display.model_input
    
    x = display.model(x)
    display.model_input = x
    print(utils.tf2pil(x[0].numpy()))
    
    display.forward(utils.tf2pil(x[0].numpy()))

def tksleep(t):
    'emulating time.sleep(seconds)'
    ms = int(t*1000)
    root = tk._get_default_root('sleep')
    var = tk.IntVar(root)
    root.after(ms, var.set, 1)
    root.wait_variable(var)


animate = False
def play(button,d):
    global animate
    animate = not animate
    
    button.configure(text= "Stop" if animate else "Play")
    while animate:
        model_step(d)
        tksleep(0.05)

def stop():
    print('in stop()')
    global animate
    animate = False

def reset(d):
    d.reset()
    
def step_back(d):
    d.back()
  
def keypress(event, key):
    global d
    if key == "left":
        step_back(d)
    elif key == "right":
        model_step(d)
    elif key == "space":
        global play_button
        play(play_button,d)

def create_button_group(parent):
    group = tk.Frame(parent)
    group.pack()
    
    return group

d = None
play_button = None

def main():
    canvas_x,canvas_y = 1000,1000

    root = tk.Tk()
    root.title("CA state visualization")

    load_button_group = create_button_group(root)
    load_model_button = tk.Button(load_button_group, text="Load Model", command=lambda: load_model(d))
    load_model_button.pack(side=tk.LEFT)
    load_img_button = tk.Button(load_button_group, text="Load Image", command=lambda: load_image(d))
    load_img_button.pack(side=tk.LEFT)
    
    control_button_group = create_button_group(root)
    previous_button = tk.Button(control_button_group, text="Previous", command= lambda: step_back(d))
    previous_button.pack(side=tk.LEFT)
    next_button = tk.Button(control_button_group, text="Next",command=lambda : model_step(d))
    next_button.pack(side=tk.LEFT)
    
    animation_button_group = create_button_group(root)
    global play_button
    play_button = tk.Button(animation_button_group, text="Play", command= lambda: play(play_button,d))
    play_button.pack(side=tk.LEFT)
    reset_button = tk.Button(animation_button_group, text="Reset",command=lambda : reset(d))
    reset_button.pack(side=tk.LEFT)
    
    global d
    d = display.Display(canvas_x,canvas_y)
    root.bind('<Left>',func=lambda event: keypress(event, key="left"))
    root.bind('<Right>',func=lambda event: keypress(event, key="right"))
    root.bind('<space>',func=lambda event: keypress(event, key="space"))
    
    root.mainloop()
    
if __name__ == '__main__':
    main()
    