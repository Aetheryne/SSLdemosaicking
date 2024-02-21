import os
import argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', type=str, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
args = parser.parse_args()

imgs_dir = args.imgs_dir
width = args.width
height = args.height

os.makedirs(imgs_dir + '\\cropped', exist_ok=True)

box = (0, 0, width, height)

for img in os.listdir(imgs_dir):
    in_img = Image.open(imgs_dir + '\\' + img)
    out_img = in_img.crop(box)
    out_img.save(imgs_dir + '\\cropped\\' + img)