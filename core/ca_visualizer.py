from core import model,utils
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

STEP_NUM = 1000
CHECKPOINT_PATH = '../checkpoints/vut_fit_default_model/small/13660'
GT_IMG_PATH = '../img/vut_logo_small.png'


def make_gif(name,frames):
    frame_one = frames[0]
    frame_one.save(name+".gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

def run_model_for_i(ca,w,h,steps,checkpoint_path = None):
    x = utils.init_batch(1,w,h)
    
    if checkpoint_path:
        ca.load_weights(checkpoint_path)

    frames = [utils.tf2pil(utils.batch_to_array(x)[0])]
    for i in range(steps):
        x = ca(x)
        frames.append(utils.tf2pil(utils.batch_to_array(x)[0]))
        
    return frames
    
    
width,height = Image.open(GT_IMG_PATH).size
x = utils.init_batch(1,width,height)

ca = model.CA()
ca.load_weights(CHECKPOINT_PATH)

frames = [utils.tf2pil(utils.batch_to_array(x)[0])]
for i in range(STEP_NUM):
    x = ca(x)
    frames.append(utils.tf2pil(utils.batch_to_array(x)[0]))

make_gif('test',frames)