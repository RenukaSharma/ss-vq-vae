import torch 
import torchvision.transforms as transforms
from PIL import Image
import os
_DATA_DIR = os.environ.get('DATA_DIR', '.')
_RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join('.', 'results'))
from PIL import ImageFile
import numpy as np 

ImageFile.LOAD_TRUNCATED_IMAGES=True

random_crop_32 = transforms.RandomCrop((64,64))
root_dir = _DATA_DIR
mask_dir = _DATA_DIR
defect_type="thread" # to change
dest_dir = _DATA_DIR
# for defect_type in {"color", "cut", "hole", "metal_contamination", "thread"}:
# 	if (not os.path.exists(os.path.join(dest_dir,defect_type))):
# 		os.mkdir(os.path.join(dest_dir,defect_type))
# 	if (not os.path.exists(os.path.join(dest_dir,defect_type,"mask"))):
# 		os.mkdir(os.path.join(dest_dir,defect_type,"mask"))
for file_name in os.listdir(os.path.join(root_dir,defect_type)):
    # print(file_name)
    file_wo_ext = file_name[:-4]
    print(file_wo_ext)
    img = Image.open(os.path.join(root_dir,defect_type, file_name))
    mask = Image.open(os.path.join(mask_dir,defect_type, file_wo_ext+"_mask.png"))
    for i in range(64):
    	for j in range(64):

    		x1= 16*i 
    		y1= 16*j 

    		# x1= 16*i 
    		# y1= 16*j

    		x2= x1+32
    		y2= y1+32

    		if( x2 <= img.size[0] and y2 <= img.size[1]):

    			img_cropped = img.crop((x1,y1,x2,y2))
    			mask_cropped = mask.crop((x1,y1,x2,y2))
    			# img_cropped = random_crop_32(img)
    			# print(mask_cropped)

    			if(np.sum(np.array(mask_cropped)/255) >= 512):			    	

    				print("np.sum(np.array(mask_cropped)/255)=", np.sum(np.array(mask_cropped)/255))
			    	img_cropped.save(os.path.join(dest_dir, defect_type,str(file_wo_ext)+str(i)+"_"+str(j)+".png" ))
			    	mask_cropped.save(os.path.join(dest_dir, defect_type,"mask",str(file_wo_ext)+str(i)+"_"+str(j)+".png" ))
"""
dest_dir = _DATA_DIR
import matplotlib.pyplot as plt 


# plot image (2 subplots)

fig, ax= plt.subplots()
ax.tick_params(axis='both', which='both', bottom=False, top=False)
# plt.xticks([])
for i in range(1,11):
	img1 = Image.open(_DATA_DIR+str(i)+".png")
	img2 = Image.open(_DATA_DIR+str(i)+".png")
	plt.subplot(4,10,i)
	plt.imshow(img1)
	# plt.tick_params(axis='both', which='both', bottom=False, top=False)
	plt.xticks([])

	plt.subplot(4,10,i+10)
	plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
	plt.xticks([])
	# plt.tick_params(axis='both', which='both', bottom=False, top=False)
for i in range(11,21):
	img1 = Image.open(_DATA_DIR+str(i)+".png")
	img2 = Image.open(_DATA_DIR+str(i)+".png")
	plt.subplot(4,10,i+10)
	plt.imshow(img1)
	plt.xticks([])
	# plt.tick_params(axis='both', which='both', bottom=False, top=False)

	plt.subplot(4,10,i+20)
	plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
	plt.xticks([])
	# plt.tick_params(axis='both', which='both', bottom=False, top=False)
plt.show()
# plt.axis("off")
plt.savefig(os.path.join(_DATA_DIR, "fig.png"))
"""