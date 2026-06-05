import os
import pathlib
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import torch.optim as optim
import torch.nn.utils as utils
import math
import torch.nn.utils.prune as prune  # Added import for weight pruning
import torchvision.datasets as datasets
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128

# Path to the compressed archive file
data_path = ''
data_path1 = ''
trainloader, testloader, trainset, testset=load_ImageNet('./')
