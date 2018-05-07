from torch import nn
from utils import Tester
from network import resnet34, resnet101

# Set Test parameters
params = Tester.TestParams()
params.use_gpu = True
params.ckpt = './models/ckpt_epoch_800_res101.pth'  #'./models/ckpt_epoch_400_res34.pth'
params.testdata_dir = './testimg/'

# models
# model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
# model.fc = nn.Linear(512, 6)
model = resnet101(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
model.fc = nn.Linear(512*4, 6)

# Test
tester = Tester(model, params)
tester.test()
