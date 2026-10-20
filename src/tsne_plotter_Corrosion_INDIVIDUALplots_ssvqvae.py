import tkinter
from cv2 import threshold
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
from networks.main import build_autoencoder
import torchvision.transforms as transforms

from datetime import datetime

timestamp_ = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

matplotlib.rcParams['legend.fontsize'] = 15
matplotlib.rcParams['legend.handletextpad']=0.01

dataset_name = 'corrosion'
# category = "carpet"
data_path = _DATA_DIR
# data_path = "../../nanofibre/"
# net_name = "nanofibre_vae"
# net_name = "mvtec_net"
net_name = "cifar10_LeNet"
xp_path = _DATA_DIR
normal_class = 0
n_known_outlier_classes = 1
known_outlier_class = 1
ratio_known_normal = 0
ratio_known_outlier = 0.2
ratio_pollution = 0
seed = 0

subplot_x=1
subplot_y=1
fig, ax = plt.subplots(subplot_x, subplot_y)

ratio_sup = "0sup_ssvqvae"
def get_tsne_projecttion(model_path, threshold_val= 2523.2, loc_x=0, loc_y=0, max_count_tp= 1000, max_count_tn=500, max_count_fp = 60, max_count_fn=60, title="A_ssvqvae",saved = False):
    net_name = "vq_vae_net"
    if (saved):
        outputs_encoded_mu = np.load(_DATA_DIR+title+"_"+ratio_sup+"sup.npy")
        train_labels = np.load(_DATA_DIR+title+"_"+ratio_sup+"sup.npy")
        predicted_labels = np.load(_DATA_DIR+title+"_"+ratio_sup+"sup.npy")
        tsne_proj = np.load(_DATA_DIR+title+"_"+ratio_sup+"sup.npy")
        
    else:
        eta = 1.0
        recon_param = 1.0
        device = "cuda"
        # device = "cpu"

        deepSAD = DeepSAD(eta, recon_param, xp_path=xp_path)
        net_name = "vq_vae_net"
        num_hiddens = 128
        num_residual_hiddens= 32
        num_residual_layers= 2
        embedding_dim= 64
        num_embeddings= 512
        commitment_cost= 1
        build_autoencoder(net_name=net_name, num_hiddens= num_hiddens, num_residual_hiddens= num_residual_hiddens, num_residual_layers= num_residual_layers, embedding_dim= embedding_dim, num_embeddings= num_embeddings, commitment_cost= commitment_cost, weight_decay= 0.5e-6)

        deepSAD.load_model(model_path=model_path, load_ae=True, map_location=device)

        deepSAD.ae_net = deepSAD.ae_net.to(device)
        deepSAD.ae_net.eval()

        dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                                ratio_known_normal, ratio_known_outlier, ratio_pollution,
                                random_state=np.random.RandomState(seed), length=5000)

        torch.cuda.empty_cache()
        # train_labels = dataset.train_set.dataset.targets[0:10000]
        train_labels = dataset.test_set.targets
        print(type(train_labels))

        torch.cuda.empty_cache()
        # final_train_data = torch.stack([I for I in dataset.train_set.dataset.tensor_batch]) # for corrosion
        final_train_data = torch.stack([I for I in dataset.test_set.tensor_batch]) # for corrosion
        # t1 = transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.ToTensor()])
        # final_train_data = torch.stack([t1(I) for I in dataset.test_set.data])
        # final_train_data = final_train_data
        # train_labels = train_labels
        print(final_train_data.shape)
        final_train_data = final_train_data.to(device)
        """
        final_train_data = torch.stack([I for I in dataset.train_set.tensor_batch])
        final_train_data = final_train_data
        train_labels = train_labels
        print(final_train_data.shape)
        final_train_data = final_train_data.to(device)
        """

        # outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample, class_out = deepSAD.ae_net(final_train_data)

        # outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample, class_out = deepSAD.ae_net(final_train_data)
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

        torch.cuda.empty_cache()
        # final_train_data = final_train_data.detach().cpu().numpy()        
        # torch.cuda.empty_cache()
        # final_train_data = final_train_data[0:10000,]
        # final_train_data = torch.from_numpy(final_train_data)
        # torch.cuda.empty_cache()
        # final_train_data = final_train_data.to(device)
        vq_loss, data_recon, z, quantized, perplexity = deepSAD.ae_net(final_train_data)

        print(z.shape)
        print(quantized.shape)
        emb = torch.norm(z-quantized, dim=(2,3))
        scores = torch.norm(emb, dim=1)
        emb = emb.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        
        np.save(_DATA_DIR+title+"_"+ratio_sup+"sup.npy",emb)

        train_labels = np.array(train_labels)
        np.save(_DATA_DIR+title+"_"+ratio_sup+"sup.npy", np.array(train_labels))        

        predicted_labels = np.zeros_like(train_labels)
        for i in range(scores.shape[0]):
            if(np.log(scores[i]) > np.log(scores).mean()): # thershold= 2523.2
            # if(np.log(scores[i]) > threshold_val): # thershold= 2523.2
                predicted_labels[i]=1
            else:
                predicted_labels[i]=0
        np.save(_DATA_DIR+title+"_"+ratio_sup+"sup.npy", np.array(predicted_labels))

        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(emb)
        np.save(_DATA_DIR+title+"_"+ratio_sup+"sup.npy", tsne_proj)

  
    num_categories = 2
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = [] # for legend

    for i in range(predicted_labels.shape[0]):
        list_1.append(tsne_proj[i,0]) # x's
        list_2.append(tsne_proj[i,1]) # y's
        
        if (predicted_labels[i]==train_labels[i]): 
            legend_="True" 
            marker_dict_1 ={0:"<", 1:">"}  
            list_3.append(marker_dict_1[train_labels[i]])
        else:
            legend_="False"
            marker_dict_2 ={0:"P", 1:"X"}
            list_3.append(marker_dict_2[train_labels[i]])
        
        if predicted_labels[i]==0:
            label_ = "Inliers"
        else:
            label_ = "Outliers"
        # print("Appending",legend_,label_)
        list_4.append(legend_+" "+label_)

    flag_tp=0
    count_tp = 0
    flag_tn=0
    count_tn = 0
    flag_fp=0
    count_fp=0
    flag_fn=0
    count_fn=0
    # colors=["#0000FF", "#00FF00", "#FF0066"]
    color_scheme = {"tp":"olive", "tn":"salmon", "fp":"indigo", "fn":"deepskyblue"}

    marker_size = 60 

    for x, y, m, l in zip(list_1, list_2, list_3, list_4):
        if l=="True Inliers":
            if count_tp < max_count_tp:
                # ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_tp==0 else "", c = color_scheme["tp"], s= marker_size)
                ax.scatter([x], [y] , marker = m, label = l if flag_tp==0 else "", c = color_scheme["tp"], s= marker_size)
                count_tp+=1
            flag_tp=1
        if l=="True Outliers":
            if count_tn < max_count_tn:
                # ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_tn==0 else "", c = color_scheme["tn"], s= marker_size)
                ax.scatter([x], [y] , marker = m, label = l if flag_tn==0 else "", c = color_scheme["tn"], s= marker_size)
                count_tn+=1
            flag_tn=1    

    for x, y, m, l in zip(list_1, list_2, list_3, list_4):    
        if l=="False Inliers":        
            if(count_fp< max_count_fp):
                # ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_fp==0 else "", c = color_scheme["fp"], s= marker_size)
                ax.scatter([x], [y] , marker = m, label = l if flag_fp==0 else "", c = color_scheme["fp"], s= marker_size)
                count_fp+=1
            flag_fp=1
            
        if l=="False Outliers":        
            if(count_fn< max_count_fn):
                # ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_fn==0 else "", c = color_scheme["fn"], s= marker_size)        
                ax.scatter([x], [y] , marker = m, label = l if flag_fn==0 else "", c = color_scheme["fn"], s= marker_size)
                count_fn+=1
            flag_fn=1
    
    # Hide axes ticks
    # ax[loc_x,loc_y].set_xticks([])
    # ax[loc_x,loc_y].set_yticks([])
    # ax[loc_x,loc_y].set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(title)

def get_tsne_projecttion_ru_vae(model_path,  loc_x=0, loc_y=0, max_count_tp= 1000, max_count_tn=500, max_count_fp = 60, max_count_fn=60, title="Ablation A1",saved = False):
    if (saved):
        outputs_encoded_mu = np.load(_DATA_DIR+title+".npy")
        train_labels = np.load(_DATA_DIR+title+".npy")
        predicted_labels = np.load(_DATA_DIR+title+".npy")
        tsne_proj = np.load(_DATA_DIR+title+".npy")
        
    else:
        eta = 1.0
        recon_param = 1.0
        # device = "cuda:7"
        device = "cpu"

        deepSAD = DeepSAD(eta, recon_param, xp_path=xp_path)
        deepSAD.set_network(net_name)

        deepSAD.load_model(model_path=model_path, load_ae=True, map_location=device)

        deepSAD.ae_net = deepSAD.ae_net.to(device)
        deepSAD.ae_net.eval()

        dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                                ratio_known_normal, ratio_known_outlier, ratio_pollution,
                                random_state=np.random.RandomState(seed), length=5000)

        train_labels = dataset.test_set.targets
        print(type(train_labels))
        
        
        # final_train_data = torch.stack([I for I in dataset.test_set.tensor_batch])
        t1 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()])
        final_train_data = torch.stack([t1(I) for I in dataset.test_set.data])
        # final_train_data = final_train_data
        train_labels = train_labels
        print(final_train_data.shape)
        final_train_data = final_train_data.to(device)
        """
        final_train_data = torch.stack([I for I in dataset.train_set.tensor_batch])
        final_train_data = final_train_data
        train_labels = train_labels
        print(final_train_data.shape)
        final_train_data = final_train_data.to(device)
        """

        # outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample, class_out = deepSAD.ae_net(final_train_data)

        # outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample, class_out = deepSAD.ae_net(final_train_data) # RU-VAE

        outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = deepSAD.ae_net(final_train_data) # ss-gVAE
        
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

        print(outputs_encoded_mu.shape)
        outputs_encoded_mu = np.squeeze(outputs_encoded_mu.cpu().detach().numpy())
        np.save(_DATA_DIR+title+".npy",outputs_encoded_mu)

        train_labels = np.array(train_labels)
        np.save(_DATA_DIR+title+".npy", np.array(train_labels))
        scores = torch.norm(sample, dim=1) ** 2
        scores = scores.detach().numpy()

        predicted_labels = np.zeros_like(train_labels)
        for i in range(scores.shape[0]):
            if(np.log(scores[i]) > np.log(scores).mean()):
                predicted_labels[i]=1
            else:
                predicted_labels[i]=0
        np.save(_DATA_DIR+title+".npy", np.array(predicted_labels))

        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(outputs_encoded_mu)
        np.save(_DATA_DIR+title+".npy", tsne_proj)

  
    num_categories = 2
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = [] # for legend

    for i in range(predicted_labels.shape[0]):
        list_1.append(tsne_proj[i,0]) # x's
        list_2.append(tsne_proj[i,1]) # y's
        
        if (predicted_labels[i]==train_labels[i]): 
            legend_="True" 
            marker_dict_1 ={0:"<", 1:">"}  
            list_3.append(marker_dict_1[train_labels[i]])
        else:
            legend_="False"
            marker_dict_2 ={0:"P", 1:"X"}
            list_3.append(marker_dict_2[train_labels[i]])
        
        if predicted_labels[i]==0:
            label_ = "Inliers"
        else:
            label_ = "Outliers"
        # print("Appending",legend_,label_)
        list_4.append(legend_+" "+label_)

    flag_tp=0
    count_tp = 0
    flag_tn=0
    count_tn = 0
    flag_fp=0
    count_fp=0
    flag_fn=0
    count_fn=0
    # colors=["#0000FF", "#00FF00", "#FF0066"]
    color_scheme = {"tp":"olive", "tn":"salmon", "fp":"indigo", "fn":"deepskyblue"}

    marker_size = 30 

    for x, y, m, l in zip(list_1, list_2, list_3, list_4):
        
        if l=="True Outliers":
            if count_tn < max_count_tn:
                ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_tn==0 else "", c = color_scheme["tn"], s= marker_size)
                count_tn+=1
            flag_tn=1 
    for x, y, m, l in zip(list_1, list_2, list_3, list_4):
        if l=="True Inliers":
            if count_tp < max_count_tp:
                ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_tp==0 else "", c = color_scheme["tp"], s= marker_size)
                count_tp+=1
            flag_tp=1   

    for x, y, m, l in zip(list_1, list_2, list_3, list_4):    
        if l=="False Inliers":        
            if(count_fp< max_count_fp):
                ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_fp==0 else "", c = color_scheme["fp"], s= marker_size)
                count_fp+=1
            flag_fp=1
    for x, y, m, l in zip(list_1, list_2, list_3, list_4):        
        if l=="False Outliers":        
            if(count_fn< max_count_fn):
                ax[loc_x,loc_y].scatter([x], [y] , marker = m, label = l if flag_fn==0 else "", c = color_scheme["fn"], s= marker_size)        
                count_fn+=1
            flag_fn=1
    
    # Hide axes ticks
    ax[loc_x,loc_y].set_xticks([])
    ax[loc_x,loc_y].set_yticks([])

    ax[loc_x,loc_y].set_title(title)



####################################################
##ss-ms-gVAE-VAE - 1
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # ss-ms-gVAE
# model_path = os.path.join(_RESULTS_DIR, "model.tar")
# get_tsne_projecttion(model_path, loc_x=0, loc_y=0, max_count_tp= 1000, max_count_tn=1000, max_count_fp = 20, max_count_fn=20, title="ss-ms-gVAE", saved = True)
# print("ss-ms-gVAE done---------------")
# ####################################################
# ##RU-VAE - 2
# model_path = os.path.join(_RESULTS_DIR, "model.tar")
# get_tsne_projecttion_ru_vae(model_path, loc_x=0, loc_y=1, max_count_tp= 1000, max_count_tn=500, max_count_fp = 40, max_count_fn=40, title="RU-VAE", saved = True)
# print("RU-VAE done-------------------")
# ######################################################
# ##ss-gVAE - 3
# model_path = os.path.join(_RESULTS_DIR, "model.tar")
# model_path = os.path.join(_RESULTS_DIR, "model.tar") 
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # full sup ss-gVAE
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0.2 sup ss-gVAE
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0 sup ss-gVAE
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # full sup deepSAD
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0.2 sup deepSAD
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0 sup ss-vq-vae
# get_tsne_projecttion(model_path, threshold_val= 2523.2, loc_x=1, loc_y=0, max_count_tp= 1000, max_count_tn=500, max_count_fp = 80, max_count_fn=70, title="ss-VQ-VAE", saved = False)

model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0 sup ss-vq-vae
get_tsne_projecttion(model_path, threshold_val= 2523.2, loc_x=1, loc_y=0, max_count_tp= 1000, max_count_tn=500, max_count_fp = 80, max_count_fn=70, title="ss-VQ-VAE", saved = False)
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0.2 sup ss-vq-vae
# get_tsne_projecttion(model_path, threshold_val= 114.003, loc_x=1, loc_y=0, max_count_tp= 20000, max_count_tn=20000, max_count_fp = 15, max_count_fn=25, title="ss-VQ-VAE", saved = False)
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0.49 sup ss-vq-vae
# get_tsne_projecttion(model_path, threshold_val= 88.3961, loc_x=1, loc_y=0, max_count_tp= 1000, max_count_tn=500, max_count_fp = 10, max_count_fn=10, title="ss-VQ-VAE", saved = True)
print("Done------------------")
# # #####################################################
# # ## ablation B - 4
# # model_path = os.path.join(_RESULTS_DIR, "model.tar")
# # get_tsne_projecttion(model_path, loc_x=1, loc_y=1, max_count_tp= 1000, max_count_tn=500, max_count_fp = 60, max_count_fn=60, title="Ablation A1")

# # #######################################################
# # ## ablation C - 5
# # model_path = os.path.join(_RESULTS_DIR, "model.tar")
# # get_tsne_projecttion(model_path, loc_x=2, loc_y=0, max_count_tp= 1000, max_count_tn=500, max_count_fp = 60, max_count_fn=60, title="Ablation A2")


# # #######################################################
# # ## ablation D - 6
# # model_path = os.path.join(_RESULTS_DIR, "model.tar")
# # get_tsne_projecttion(model_path, loc_x=2, loc_y=1, max_count_tp= 1000, max_count_tn=500, max_count_fp = 60, max_count_fn=60, title="Ablation A3")


# # #######################################################
# # ## ablation E - 7
# model_path = os.path.join(_RESULTS_DIR, "model.tar")
# # get_tsne_projecttion(model_path, loc_x=3, loc_y=0, max_count_tp= 1000, max_count_tn=500, max_count_fp = 60, max_count_fn=60, title="Ablation A4")
# get_tsne_projecttion_ru_vae(model_path, loc_x=1, loc_y=1, max_count_tp= 1000, max_count_tn=500, max_count_fp = 60, max_count_fn=60, title="DeepSAD", saved = True)
# print("DeepSAD done-------------------")


# ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1], label = label_, alpha=0.5, marker = marker_dict_1[lab])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())  
# predicted X, actual Y; Inliers- Predicted Inliers, Outliers- Predicted Outliers, True- 

##### To keep legends on ##################
# lgnd = ax.legend([by_label['True Inliers'],by_label['False Outliers'],by_label['True Outliers'],by_label['False Inliers']],['Predicted Inlier, Actual Inlier','Predicted Outlier, Actual Inlier','Predicted Outlier, Actual Outlier','Predicted Inlier, Actual Outlier'], markerscale=2, loc='lower right') # legend, remove if you don't want the legend

# lgnd = ax[0,0].legend([by_label['True Inliers'],by_label['False Outliers'],by_label['True Outliers'],by_label['False Inliers']],['Predicted Inlier, Actual Inlier','Predicted Outlier, Actual Inlier','Predicted Outlier, Actual Outlier','Predicted Inlier, Actual Outlier'], markerscale=2) # legend, remove if you don't want the legend
# lgnd.legendHandles[0]._sizes = [60]
# lgnd.legendHandles[1]._sizes = [60]
# lgnd.legendHandles[2]._sizes = [60]
# lgnd.legendHandles[3]._sizes = [60]
###########################################

# Hide axes ticks
# ax[0,0].set_xticks([])
# ax[0,0].set_yticks([])
# ax.set_xticks([])
# ax.set_yticks([])
# ax.legend(fontsize='large', markerscale=2)
# plt.show()
# plt.savefig(_DATA_DIR+timestamp_+".png")

# plt.grid(False)
plt.axis('off')
plt.tight_layout()
# Hide grid lines
# ax[0,0].grid(False)
plt.savefig(_DATA_DIR+ratio_sup+"sup"+timestamp_+".png", dpi=300, bbox_inches='tight')

# legends_only = Image.open(os.path.join(_DATA_DIR, 'just_legends_.png'))
# legends_only = legends_only.resize((800,300))
# height = legends_only.size[1]
# width = legends_only.size[0]
# fig.figimage(legends_only, fig.bbox.xmax +width//2.15, fig.bbox.ymax -height//1.5)
# fig.savefig(_DATA_DIR+timestamp_+".png", dpi=300, bbox_inches='tight')

