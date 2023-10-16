from model import Discriminator
from model import Encoder_Decoder
from torch.autograd import Variable
import torch.optim as optim
import torch
import numpy as np
import os
import time
import torch.nn as nn
import cv2


class Solver(object):
    """Solver for training and testing PIMoG."""

    def __init__(self, data_loader, data_loader_test, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test
        # Model configurations.
        self.image_size = config.image_size
        self.num_channels = config.num_channels
        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2
        self.lambda3 = config.lambda3
        self.num_epoch = config.num_epoch
        self.distortion = config.distortion

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.model_name = config.model_name
        self.result_dir = config.result_dir
        self.embedding_epoch = config.embedding_epoch

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # Build the model.
        self.build_model()

    def build_model(self):
        if self.dataset in ['test_embedding']:
            self.net_ED = Encoder_Decoder(self.distortion)
            self.net_ED = self.net_ED.to(self.device)
            self.net_E = self.net_ED.Encoder
            self.net_ED = torch.nn.DataParallel(self.net_ED)
            self.net_ED.load_state_dict(torch.load(
                self.model_save_dir + '/' + 'SSRH.pth'))
        elif self.dataset in ['test_accuracy']:
            self.net = Encoder_Decoder(self.distortion)
            self.print_network(self.net, self.dataset)
            self.net.to(self.device)
            self.net_D = self.net.Decoder
            self.net = torch.nn.DataParallel(self.net)
            self.net.load_state_dict(torch.load(
                self.model_save_dir + '/' + 'SSRH.pth'))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

    def test_embedding(self):
        # Set data loader.
        data_loader = self.data_loader
        data_loader_test = self.data_loader_test
        criterion_MSE = nn.MSELoss()
        self.net_ED.eval()
        for i, (data, m, num, name) in enumerate(data_loader):
            inputs, m = Variable(data), Variable(m.float())
            inputs, m = inputs.to(self.device), m.to(self.device)
            inputs.requires_grad = True
            num = num.to('cpu').numpy()
            Encoded_image, Decoded_message = self.net_ED(inputs, m)
            loss_de = criterion_MSE(Decoded_message, m)
            loss_de.backward()
            inputgrad = inputs.grad.data
            mask = torch.zeros(inputgrad.shape).to(self.device)
            for ii in range(inputgrad.shape[0]):
                a = inputgrad[ii, :, :, :]
                a = (1 - (a - a.min()) / (a.max() - a.min())) + 1
                mask[ii, :, :, :] = a

            for j in range(Encoded_image.shape[0]):
                I1 = (inputs[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I1 = np.transpose(I1, (1, 2, 0))
                I2 = (Encoded_image[j, :, :, :].detach().to('cpu').numpy() + 1) / 2 * 255
                I2 = np.transpose(I2, (1, 2, 0))
                index = num[j]
                cv2.imwrite(self.result_dir + '/embed/' + str(index) + '.png', I2)
        print('Embed finished!')

    def test_accuracy(self):
        data_loader_test = self.data_loader_test

        correct = 0
        total = 0
        for i, (data, m, num, name) in enumerate(data_loader_test):
            inputs, m = Variable(data), Variable(m.float())
            inputs, m = inputs.to(self.device), m.to(self.device)
            self.net_D.eval()
            Decoded_message = self.net_D(inputs)
            decoded_rounded = Decoded_message.detach().cpu().numpy().round().clip(0, 1)
            # print(decoded_rounded)
            p1 = np.sum(np.abs(decoded_rounded - m.detach().cpu().numpy()))
            p2 = inputs.shape[0] * m.shape[1]
            # print(name[0])
            print('Accuracy of ' + name[0] + ' image: %.3f' % ((1 - p1 / p2) * 100) + '%')
            correct += np.sum(np.abs(decoded_rounded - m.detach().cpu().numpy()))
            total += inputs.shape[0] * m.shape[1]

        print('Accuracy of ' + self.distortion + ' image: %.3f' % ((1 - correct / total) * 100) + '%')
