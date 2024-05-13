import tkinter as tk
import sys
from io import StringIO

class RedirectedOutput:
    def __init__(self, textbox):
        self.textbox = textbox
        self.textbox.configure(state='disabled')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.textbox.configure(state='normal')  # Enable editing temporarily
        self.textbox.insert(tk.END, message)
        self.textbox.see(tk.END)  # Scrolls to the end of the textbox
        self.textbox.configure(state='disabled')  # Disable editing again

    def flush(self):
        pass

    def __del__(self):
        sys.stdout = self.stdout