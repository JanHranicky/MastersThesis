import tkinter as tk
import numpy as np

class Display():
    DEFAULT_SIZE = 30
    
    def __init__(self,c_x,c_y) -> None:
        self.m = None
        
        self.c_x = c_x
        self.c_y = c_y
        
        self.x = self.DEFAULT_SIZE
        self.y = self.DEFAULT_SIZE
        
        self.canvas = None
        self.canvas_grid = []
        self.model_input = None
        
        self.set_image(self.x,self.y)

    def calculate_sq_size(self,x,y):
        sq_size_setter = x if x > y else y
        self.sq_size = self.c_x // sq_size_setter 

    def set_image(self,x,y):
        self.x = x 
        self.y = y 
        
        self.calculate_sq_size(x,y)        
        self.init_canvas_grid(x,y)
        
    def set_model(self,m):
        self.model = m

    def rgb_to_hex(sekf,r,g,b):
        # Ensure the RGB values are within the valid range (0-255)
        r, g, b = [min(max(val, 0), 255) for val in [r,g,b]]

        # Convert each component to its hexadecimal representation
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return hex_color

    def pixel_clicked(self,event):
        x, y = event.x, event.y
        pixel_x = x // self.sq_size
        pixel_y = y // self.sq_size
        print(f'[x,y]=[{x},{y}]')
        print(f"Clicked on pixel ({pixel_x}, {pixel_y})")
        print(f'color of clicked = {self.canvas.itemcget(self.canvas_grid[pixel_x][pixel_y], "fill")}')
    def init_canvas_grid(self,width,height):
        if self.canvas is not None:
            self.canvas.pack_forget() #remove old canvas before creating new one
            
        self.canvas_grid = [] #reset canvas grid
        self.canvas = tk.Canvas(width=self.c_x, height=self.c_y)  # Adjust the width and height as needed
        self.canvas.pack()
        
        for y in range(width):
            self.canvas_grid.append([])
            for x in range(height):
                self.canvas_grid[y].append(self.canvas.create_rectangle(x * self.sq_size, y * self.sq_size,
                                        (x + 1) * self.sq_size, (y + 1) * self.sq_size,
                                        fill=self.rgb_to_hex(255,255,255)))
        self.canvas.bind("<Button-1>", self.pixel_clicked)
    
    def display_image(self,image_data):
        if self.canvas is None:
            self.init_canvas_grid(self.x,self.y)
            print('canvas intialized')
        
        width, height = image_data.size
        for y in range(height):
            for x in range(width):
                pixel = image_data.getpixel((x, y))  # Get RGB values of the pixel
                hex_color = "#{:02x}{:02x}{:02x}".format(*pixel)
                #print(f"Pixel at ({x}, {y}): {hex_color}")
                id =  self.canvas_grid[y][x]
                self.canvas.itemconfig(id, fill=hex_color)
        
        #for y, row in enumerate(image_data):
        #    for x, pixel in enumerate(row):
        #        #print(f'at index [y][x]=[{y}][{x}]')
        #        id =  self.canvas_grid[y][x]
        #        self.canvas.itemconfig(id, fill=self.rgb_to_hex(pixel[0],pixel[1],pixel[2]))
        #        #print(f'set color of canvas rectangle to {self.rgb_to_hex(pixel[0],pixel[1],pixel[2])}')