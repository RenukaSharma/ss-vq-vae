import json
import os


# ae_path = "../log_cifar10/run29/"
# vae_path = "../log_cifar10/run28/"


# lengths = [128, 512, 1000, 2000, 3000, 5000]
# seeds = [124, 273, 42, 300, 10]
# ae_test_aucs = []
# vae_test_aucs = []

# files = ["seed_VAE_" + str(seed) for seed in seeds]

# for length in lengths:
#     temp_ae_aucs = []
#     temp_vae_aucs = []
#     for seed in seeds:
#         ae_file = ae_path + "/length_" + str(length) + "/seed_D_" + str(seed) + "/results.json"
#         vae_file = vae_path + "/length_" + str(length) + "/seed_VAE_" + str(seed) + "/results.json"
#         ae_config = json.load(open(ae_file, 'r'))
#         vae_config = json.load(open(vae_file, 'r'))
#         temp_ae_aucs.append(ae_config["test_auc"])
#         temp_vae_aucs.append(vae_config["test_auc"])

#     ae_test_aucs.append(temp_ae_aucs)
#     vae_test_aucs.append(temp_vae_aucs)

# print(ae_test_aucs)
# print(vae_test_aucs)


ae_path = "../log_cifar10/run33/"
vae_path = "../log_cifar10/run28/"


ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
seeds = [124, 273, 42, 300, 10]
ae_test_aucs = []
vae_test_aucs = []

files = ["seed_VAE_" + str(seed) for seed in seeds]

for length in ratios:
    temp_ae_aucs = []
    temp_vae_aucs = []
    for seed in seeds:
        ae_file = ae_path + "/ratio_" + str(length) + "/seed_B_" + str(seed) + "/results.json"
        # vae_file = vae_path + "/length_" + str(length) + "/seed_VAE_" + str(seed) + "/results.json"
        ae_config = json.load(open(ae_file, 'r'))
        # vae_config = json.load(open(vae_file, 'r'))
        temp_ae_aucs.append(ae_config["test_auc"])
        # temp_vae_aucs.append(vae_config["test_auc"])

    ae_test_aucs.append(temp_ae_aucs)
    # vae_test_aucs.append(temp_vae_aucs)

print(ae_test_aucs)
# print(vae_test_aucs)





# for f in files:
#     cfp = path + f + "/results.json"
#     config = json.load(open(cfp, 'r'))
#     test_aucs.append(config["test_auc"])

# print(test_aucs)

