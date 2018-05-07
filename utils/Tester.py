from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F


class TestParams(object):
    # params based on your local env
    use_gpu = False             # default do not use gpu

    # loading existing checkpoint
    ckpt = './models/ckpt_epoch_800_res101.pth'     # path to the ckpt file

    testdata_dir = './testimg/'

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params):
        assert isinstance(test_params, TestParams)
        self.params = test_params

        # load model
        self.model = model
        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES
        if self.params.use_gpu:
            logger.info('Set CUDA_VISIBLE_DEVICES to 0...')
            self.model = self.model.cuda()

        self.model.eval()

    def test(self):

        img_list = os.listdir(self.params.testdata_dir)

        for img_name in img_list:
            print('Processing image: ' + img_name)

            img = Image.open(os.path.join(self.params.testdata_dir, img_name))
            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if self.params.use_gpu:
                img_input = img_input.cuda()

            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)

            print('Prediction number: ' + str(prediction[0]))

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(torch.load(ckpt))
