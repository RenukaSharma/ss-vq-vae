import torch 
import torchvision.transforms as transforms
from PIL import Image
import os
_DATA_DIR = os.environ.get('DATA_DIR', '.')
_RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join('.', 'results'))
from PIL import ImageFile
import numpy as np 

ImageFile.LOAD_TRUNCATED_IMAGES=True

random_crop_64 = transforms.RandomCrop((64,64))
root_dir = _DATA_DIR

# defect_type="thread" # to change
dest_dir = _DATA_DIR
for file_name in os.listdir(root_dir):
    # print(file_name)
    file_wo_ext = file_name[:-4]
    print(file_wo_ext)
    img = Image.open(os.path.join(root_dir, file_name))
    for i in range(5):
	    
	    img_cropped = random_crop_64(img)
	    img_cropped.save(os.path.join(dest_dir, str(file_wo_ext)+"_"+str(i)+".png"))