import tkinter
from sklearn.manifold import TSNE
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
from matplotlib import cm
import numpy as np
import click
import torch
import logging
import random
import numpy as np
import PIL
from PIL import Image

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid, plot_matrix_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import torchvision.transforms as transforms

from datetime import datetime

timestamp_ = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
import gradio as gr

def segment(image):
    # Implement image segmentation model here...
    # Return segmented image
    print("segmenting the image")
    orig_h = image.shape[0]
    orig_w = image.shape[1]
    print("shape of image=", image.shape)
    import numpy as np
    import pickle
    import torch
    # model = torch.load(os.path.join(_DATA_DIR, "weights.pt"))
    model_path = torch.load(os.path.join(_RESULTS_DIR, "model.tar"))
    # model_path = os.path.join(_RESULTS_DIR, "model.tar") # full sup ss-gVAE from CARPET
    # model_path.eval()
    xp_path = _DATA_DIR

    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms
    import cv2
  
    # input_image = cv2.imread(filename, 1) #.transpose(2,1,0)
    # input_image = Image.open(filename)
######### Adding fro t-SNE plotter code: START ##################
    eta = 1.0
    recon_param = 1.0
    # device = "cuda:7"
    # device = "cpu"
    device= "cuda"
    net_name = "cifar10_LeNet"
    deepSAD = DeepSAD(eta, recon_param, xp_path=xp_path)
    deepSAD.set_network(net_name)

    deepSAD.load_corrosion_model(model=model_path, load_ae=True, map_location=device)

    deepSAD.ae_net = deepSAD.ae_net.to(device)
    deepSAD.ae_net.eval()
    dataset_name = "corrosion"
    data_path = _DATA_DIR
    normal_class = 0
    n_known_outlier_classes = 1
    known_outlier_class = 1
    ratio_known_normal = 0
    ratio_known_outlier = 0.49
    ratio_pollution = 0
    seed = 0

    # dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                            # ratio_known_normal, ratio_known_outlier, ratio_pollution,
                            # random_state=np.random.RandomState(seed), length=5000)
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32,32)),
        transforms.ToTensor()    
    ])

    image = cv2.resize(image, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

    input_tensor = preprocess(image)
    # input_tensor = input_tensor.type(torch.FloatTensor)/255.0
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        deepSAD.ae_net.to('cuda')

    outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = deepSAD.ae_net(input_batch)
    anomaly_score = torch.abs(outputs_recons_mu - input_batch)
    # anomaly_score = anomaly_score.squeeze()

    # with torch.no_grad():
    #     output = model(input_batch)['out'][0]
    # output_predictions = output.argmax(0)
    
    # eps = 1e-5
    # reconImg, sampleRecon, samplingParasAll, out_class = deepSAD.ae_net(final_train_data, ablation_type='A', eps=eps)
    # # outputs_recons_mu, outputs_recons_beta = samplingParasAll['final']['mu'], samplingParasAll['final']['sigma']
    # outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta = samplingParasAll['final']['mu'], samplingParasAll['final']['sigma'], samplingParasAll['final']['beta']

    # dist_first = 0.0
    # dist_second = 0.0
    # for key in samplingParasAll['hidden'].keys():       
    #     para = samplingParasAll['hidden'][key]
    #     sample = para['sample']
    #     outputs_encoded_mu = para['mu']
    #     outputs_encoded_alpha = para['sigma']
    #     outputs_encoded_beta = para['beta']
        
    #     # dist_first += log_gg(sample, outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, self.delta, self.tau, self.eps)
    
    #     # dist_second += 0.5 * torch.mean(sample**2,dim=tuple(range(1, sample.dim())))

    print(anomaly_score.shape)
    anomaly_score = np.squeeze(anomaly_score.cpu().detach().numpy())
    
################# adding from t-SNE plotter: END ###################

    # r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((512, 512))
    # out = np.asarray(r)
    # print("max val:", out.max())
    # print("min val:", out.min())
    # print("Unique values:", np.unique(out))
    # print("The output is:\n",out)
    # # r.putpalette(colors)

    # import matplotlib.pyplot as plt
    # plt.imshow(r)
    # print("Saving the image res_Jun14_22.png")
    # plt.savefig(os.path.join(_DATA_DIR, "res_Jun14_22.png"))
    
    # return_img = output.cpu().numpy()
    # min = anomaly_score.min()

    # anomaly_score = np.mean(anomaly_score, axis=0) # taking average across channels

    avg = anomaly_score.mean()
    h = anomaly_score.shape[1]
    w = anomaly_score.shape[2]
    
    img_thres= np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel
            pixel = anomaly_score[:,y, x].mean()
            img_thres[y, x] = 0 if pixel < avg else 1    
    
    return img_thres

    # img_thres= np.zeros((3,h,w))
    # # loop over the image, pixel by pixel
    # for channel in range(3):
    #     avg = anomaly_score[channel,:].mean()
    #     min = anomaly_score.min()
    #     max = anomaly_score.max()
    #     for y in range(0, h):
    #         for x in range(0, w):
    #             # threshold the pixel
    #             pixel = anomaly_score[channel,y, x]
    #             img_thres[channel,y, x] = min if pixel < avg else max

    return_img = img_thres.transpose([1,2,0])
    print("Returning imag eof size", return_img.shape)
    return return_img

iface = gr.Interface(fn=segment, inputs="image", outputs="image").launch(share=True)