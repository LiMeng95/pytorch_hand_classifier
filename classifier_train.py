import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import Hand
from utils import Trainer
from network import resnet34, resnet101

# Hyper-params
data_root = './data/'
model_path = './models/'
batch_size = 60  # resnet34: batch_size=120
num_workers = 2

init_lr = 0.01
lr_decay = 0.8
momentum = 0.9
weight_decay = 0.000
nesterov = True

# Set Training parameters
params = Trainer.TrainParams()
params.max_epoch = 1000
params.criterion = nn.CrossEntropyLoss()
params.use_gpu = True
params.save_dir = model_path
params.ckpt = None
params.save_freq_epoch = 100

# load data
print("Loading dataset...")
train_data = Hand(data_root,train=True)
val_data = Hand(data_root,train=False)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

# models
# model = resnet34(pretrained=False, modelpath=model_path, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
# model.fc = nn.Linear(512, 6)
model = resnet101(pretrained=False, modelpath=model_path, num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
model.fc = nn.Linear(512*4, 6)

# optimizer
trainable_vars = [param for param in model.parameters() if param.requires_grad]
print("Training with sgd")
params.optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)

# Train
params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=lr_decay, patience=10, cooldown=10, verbose=True)
trainer = Trainer(model, params, train_dataloader, val_dataloader)
trainer.train()
