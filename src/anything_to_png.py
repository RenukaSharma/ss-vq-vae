import os
_DATA_DIR = os.environ.get('DATA_DIR', '.')
_RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join('.', 'results'))
from tqdm import tqdm
from PIL import Image
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

img_folder = _DATA_DIR
for file_name in tqdm(os.listdir(img_folder)):
    img = Image.open(os.path.join(img_folder, file_name))
    img.save(_DATA_DIR+file_name[:-4]+ ".png","PNG")
