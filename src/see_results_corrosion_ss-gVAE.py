import numpy as np
import pickle
import torch
from networks.main import build_autoencoder
from datetime import datetime
timestamp_ = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
# model = torch.load(os.path.join(_DATA_DIR, "weights.pt"))

# model.eval()

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
filename = os.path.join(_DATA_DIR, "Image7.jpg")
# filename = os.path.join(_DATA_DIR, "Image4.jpg")
# filename = os.path.join(_DATA_DIR, "image1.jpg")


# input_image = cv2.imread(filename, 1) #.transpose(2,1,0)
input_image = Image.open(filename)

# preprocess = transforms.Compose([
#     transforms.Resize((32,2)),
#     transforms.ToTensor()    
# ])
preprocess = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

input_tensor = preprocess(input_image)
# input_tensor = input_tensor.type(torch.FloatTensor)/255.0
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    # model.to('cuda')

############## for ss-gVE #############
model_path = os.path.join(_RESULTS_DIR, "model.tar")

map_location = 'cuda:0'
model_dict = torch.load(model_path, map_location=map_location)
# with open(model_path, 'rb') as f:
#      buffer = io.BytesIO(f.read())
# torch.load(buffer)
ae_net = None
net_name = 'cifar10_LeNet'
if ae_net is None:
    ae_net = build_autoencoder(net_name)

print(type(model_dict))
print(model_dict.keys())
print(type(model_dict['ae_net_dict']))

ae_net.load_state_dict(model_dict['ae_net_dict'])
ae_net = ae_net.to(device=map_location)
outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(input_batch, ablation_type='A', eps=1e-5)
# scores = torch.log(torch.norm(sample, dim=1) ** 2)
# scores = (torch.norm(sample, dim=1) ** 2)

# outputs_recons_mu= torch.mean(outputs_recons_mu, 0)
outputs_recons_mu= outputs_recons_mu.squeeze().cpu().detach().numpy()
input_batch = input_batch.squeeze().cpu().detach().numpy()

scores = np.log((outputs_recons_mu-input_batch) ** 2)
mat=scores
print("mat.shape",mat.shape)      

# if(scores >= 234.858):
#     mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = 1
# else:
#     mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = 0

# mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = scores


file_index = "7"
print("Matrix:\n",mat)
# np.save(_RESULTS_DIR+file_index+"_mat", mat)
print("max val:", mat.max())
print("min val:", mat.min())
print("mean val:", mat.mean())
mat = ((mat - mat.min())/(mat.max()-mat.min()))
# output_predictions = ((mat - mat.min())/(mat.max() - mat.min())).astype('uint8')
output_predictions = mat
# output_predictions = mat.argmax(0)

#######################################

# with torch.no_grad():
#     output = model(input_batch)['out'][0]
# output_predictions = output.argmax(0)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(2)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((input_image.shape[1], input_image.shape[0]))

"""
output = output_predictions.cpu().numpy()
print("max value for output_predictions:", output.max())
print("min value for output_predictions:", output.min())
print("Unique values:", np.unique(output))
output = output/2.0
print("max value for output_predictions:", output.max())
print("min value for output_predictions:", output.min())
print("Unique values:", np.unique(output))
print("count of 1's",np.count_nonzero(output == 1))
print("count of 0's",np.count_nonzero(output == 0))
print("count of 2's",np.count_nonzero(output == 2))

r = Image.fromarray(output).resize((512, 512))
print("Mode of r:", r.mode)
r = r.convert("1")
# r.putpalette(colors)

r.save(os.path.join(_DATA_DIR, "res_nopalette_PILImage.png"))
import matplotlib.pyplot as plt
# plt.imshow(r, cmap=)
plt.imshow(r)
plt.savefig(os.path.join(_DATA_DIR, "res_nopalette.png"))
# plt.show()
"""
# plt.imsave(_DATA_DIR+timestamp_+".png", output_predictions)
# plt.imsave(os.path.join(_DATA_DIR, "result_Image7.png"), output_predictions, cmap="jet")
# r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((512, 512))
output_predictions = output_predictions.transpose(1,2,0)
output_predictions = output_predictions[:,:,0]
# r = Image.fromarray((output_predictions*255.0).astype(np.uint8)).resize((512,512))
r = Image.fromarray((output_predictions*255.0).astype(np.uint8)).resize((512,512))
# r = Image.fromarray(mat).resize((512,512))
out = np.asarray(r)
# out = output_predictions
print("max val:", out.max())
print("min val:", out.min())
print("Unique values:", np.unique(out))
print("The output is:\n",out)
# r.putpalette(colors)

fig, ax = plt.subplots(figsize=(8,8))
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
plt.grid(False)
plt.axis('off')
plt.tight_layout()


# out=out.transpose(1,2,0)
plt.imshow(r)
plt.savefig(_DATA_DIR+timestamp_+".png")

# plt.imsave(os.path.join(_DATA_DIR, "result_Image7.png"), out, cmap="jet")
# plt.show(input_image)

# img2 = transforms.ToPILImage(mode='RGB')(input_tensor)
# plt.imshow(img2)
# plt.savefig(os.path.join(_DATA_DIR, "input_2.png"))
# plt.show()