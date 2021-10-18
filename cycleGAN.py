#!/usr/bin/env python

from skimage.io import imread
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import torch
import numpy as np


from matplotlib import pyplot as plt
from tqdm import tqdm


#images = "/home/bandyadkas/cellcyle/data/"
images = "/home/hhealey/nereus/Side_Projects/ML_stickleback_populations/Oregon_ML_images_aspng/"


# In[5]:

transform = {
        'train': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            #TODO if it is needed, add the random crop
           # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=.5, hue=.3)
            #transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #transforms.Normalize(mean=(0), std=(1))
        ])
    }


# In[6]:

#print("you are almost here")
all_images = datasets.ImageFolder(images)
print(len(all_images))
print(all_images)

#print("you are here")


# In[7]:


# train_size = int(0.7 * len(all_images))
# val_size = int(0.15 * len(all_images))
# test_size = len(all_images) - (train_size + val_size)
# print(train_size, val_size, test_size)
# assert train_size + val_size + test_size == len(all_images)


# In[8]:


#train_set, val_set, test_set = torch.utils.data.random_split(all_images, [train_size, val_size, test_size])
#train_set, val_set, test_set = torch.utils.data.random_split(all_images, [22538, 4829, 4831])


# In[9]:


def _get_weights(subset,full_dataset):
    ys = np.array([y for _, y in subset])
    counts = np.bincount(ys)
    label_weights = 1.0 / counts
    weights = label_weights[ys]

    print("Number of images per class:")
    for c, n, w in zip(full_dataset.classes, counts, label_weights):
        print(f"\t{c}:\tn={n}\tweight={w}")
        
    return weights


# In[10]:


def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))    # take a random sample
    img, mask = dataset[idx]                    # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 2)               # make two plots on one figure
    axarr[0].imshow(img[0], cmap="gist_ncar")                     # show the image
    #axarr[1].imshow(mask[0])                    # show the masks
    _ = [ax.axis('off') for ax in axarr]        # remove the axes
    print('Image size is %s' % {img[0].shape})
    print(img.shape)
    plt.show()

# show_random_dataset_image(train_set)
# show_random_dataset_image(train_set)
# show_random_dataset_image(train_set)
# show_random_dataset_image(train_set)


# show_random_dataset_image(val_set)
# show_random_dataset_image(test_set)


# In[ ]:



#train_weights = _get_weights(train_set,all_images)
#train_sampler = WeightedRandomSampler(train_weights, len(train_weights))


# In[ ]:


#train_loader = DataLoader(train_set, batch_size=8, drop_last=True, sampler=train_sampler)
#val_loader = DataLoader(val_set, batch_size=8 , drop_last=True, shuffle=True)
#test_loader = DataLoader(test_set, batch_size=8, drop_last=True, shuffle=True)



pop_classes = ['Chub_site', 'Columbia_River_Mouth', 'Cushman_Slough', 'Green_Island', 'Riverbend']

def class_dir(name):
    return f'{pop_classes.index(name)}_{name}'

class_A = 'Cushman_Slough'
class_B = 'Green_Island'

import cycle_gan

print(class_A)
print(class_B)

#cycle_gan.prepare_dataset('/home/hhealey/nereus/Side_Projects/ML_stickleback_populations/Oregon_ML_images_aspng/', [class_A, class_B])

#cycle_gan.train('training_data/', class_A, class_B, 224)
 
cycle_gan.test(
    data_dir='training_data/',
    class_A=class_A,
    class_B=class_B,
    img_size=224,
    checkpoints_dir='/home/hhealey/nereus/Side_Projects/ML_stickleback_populations/checkpoints/resnet_9blocks_Cushman_Slough_Green_Island/resnet_9blocks_Cushman_Slough_Green_Island/',
    classifier_checkpoint='/home/hhealey/nereus/Side_Projects/ML_stickleback_populations/modelsave_resnet18.pth'
)

import glob
import json

result_dir = f'training_data/cycle_gan/{class_dir(class_A)}_{class_dir(class_B)}/results/test_latest/images/'
classification_results = []
for f in glob.glob(result_dir + '/*.json'):
    result = json.load(open(f))
    result['basename'] = f.replace('_aux.json', '')
    classification_results.append(result)
classification_results.sort(
    key=lambda c: c['aux_real'][pop_classes.index(class_A)] * c['aux_fake'][pop_classes.index(class_B)],
    reverse=True)