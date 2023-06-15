import unittest
from hypernet.hypernet1d import HyperConv1d, HyperSpectralConv1d, SpectralConv1d
from hypernet.hypernet2d import HyperConv2d, HyperSpectralConv2d, SpectralConv2d
from hypernet.hypernet3d import HyperConv3d, HyperSpectralConv3d, SpectralConv3d
from hypernet.hypernet1d import HyperFC as HFC1
from hypernet.hypernet2d import HyperFC as HFC2
from hypernet.hypernet3d import HyperFC as HFC3

import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.use_deterministic_algorithms(True)


class TestConv(unittest.TestCase):
    def setUp(self) -> None:
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

    def test_hconv1d(self):
        batch_size = 18
        inp = torch.rand(batch_size, 3, 24, dtype=torch.float64).cuda()
        hconv_layer = HyperConv1d(3, 4, 5).cuda().double()
        conv_layer = [nn.Conv1d(3, 4, 5).cuda().double() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack([c.weight for c in conv_layer], 0)
        hconv_biases = torch.stack([c.bias for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, torch.cat([hconv_weights.flatten(1), hconv_biases.flatten(1)], -1))
        self.assertTrue(torch.equal(out_conv, out_hconv))
        print('passed conv1d')

    def test_hconv2d(self):
        batch_size = 19
        inp = torch.rand(batch_size, 4, 20, 24, dtype=torch.float64).cuda()
        hconv_layer = HyperConv2d(4, 5, 6).cuda().double()
        conv_layer = [nn.Conv2d(4, 5, 6).cuda().double() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack([c.weight for c in conv_layer], 0)
        hconv_biases = torch.stack([c.bias for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, torch.cat([hconv_weights.flatten(1), hconv_biases.flatten(1)], -1))
        self.assertTrue(torch.equal(out_conv, out_hconv))
        print('passed conv2d')

    def test_hconv3d(self):
        batch_size = 20
        inp = torch.rand(batch_size, 4, 16, 18, 32, dtype=torch.float64).cuda()
        hconv_layer = HyperConv3d(4, 5, 6).cuda().double()
        conv_layer = [nn.Conv3d(4, 5, 6).cuda().double() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack([c.weight for c in conv_layer], 0)
        hconv_biases = torch.stack([c.bias for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, torch.cat([hconv_weights.flatten(1), hconv_biases.flatten(1)], -1))
        self.assertTrue(torch.equal(out_conv, out_hconv))
        print('passed conv3d')

    def test_hspectralconv1d(self):
        batch_size = 2
        inp = torch.rand(batch_size, 64, 1024).cuda().float()
        hconv_layer = HyperSpectralConv1d(64, 64, 16).cuda()
        conv_layer = [SpectralConv1d(64, 64, 16).cuda() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack([torch.view_as_real(c.weights1) for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, hconv_weights)
        self.assertTrue(torch.equal(out_conv, out_hconv))
        print('passed spectralconv1d')

    def test_hspectralconv2d(self):
        batch_size = 15
        inp = torch.rand(batch_size, 64, 64, 64).cuda().float()
        hconv_layer = HyperSpectralConv2d(64, 64, 16, 16).cuda()
        conv_layer = [SpectralConv2d(64, 64, 16, 16).cuda() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack(
            [torch.stack([torch.view_as_real(c.weights1), torch.view_as_real(c.weights2)], -1) for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, hconv_weights)
        self.assertTrue(torch.equal(out_conv, out_hconv))
        print('passed spectralconv2d')

    def test_hspectralconv3d(self):
        batch_size = 15
        inp = torch.rand(batch_size, 8, 64, 64, 64).cuda().float()
        hconv_layer = HyperSpectralConv3d(8, 8, 16, 16, 16).cuda()
        conv_layer = [SpectralConv3d(8, 8, 16, 16, 16).cuda() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack(
            [torch.stack(
                [torch.view_as_real(c.weights1), torch.view_as_real(c.weights2), torch.view_as_real(c.weights3),
                 torch.view_as_real(c.weights4)], -1) for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, hconv_weights)
        self.assertTrue(torch.equal(out_conv, out_hconv))

        print('passed spectralconv3d')

    def test_hfc1d(self):
        batch_size = 18
        inp = torch.rand(batch_size, 64, 3, dtype=torch.float64).cuda()
        hconv_layer = HFC1(3, 78).cuda().double()
        conv_layer = [nn.Linear(3, 78).cuda().double() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack([c.weight for c in conv_layer], 0)
        hconv_biases = torch.stack([c.bias for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, torch.cat([hconv_weights.flatten(1), hconv_biases.flatten(1)], -1))
        self.assertTrue(torch.equal(out_conv, out_hconv))

        print('passed HyperFC 1D')

    def test_hfc2d(self):
        batch_size = 19
        inp = torch.rand(batch_size, 64, 20, 3, dtype=torch.float64).cuda()
        hconv_layer = HFC2(3, 78).cuda().double()
        conv_layer = [nn.Linear(3, 78).cuda().double() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack([c.weight for c in conv_layer], 0)
        hconv_biases = torch.stack([c.bias for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, torch.cat([hconv_weights.flatten(1), hconv_biases.flatten(1)], -1))
        self.assertTrue(torch.equal(out_conv, out_hconv))
        print('passed HyperFC 2D')

    def test_hfc3d(self):
        batch_size = 20
        inp = torch.rand(batch_size, 4, 16, 18, 3, dtype=torch.float64).cuda()
        hconv_layer = HFC3(3, 78).cuda().double()
        conv_layer = [nn.Linear(3, 78).cuda().double() for c in range(batch_size)]
        out_conv = torch.cat([c(i[None]) for i, c in zip(inp, conv_layer)], 0)
        hconv_weights = torch.stack([c.weight for c in conv_layer], 0)
        hconv_biases = torch.stack([c.bias for c in conv_layer], 0)
        out_hconv = hconv_layer(inp, torch.cat([hconv_weights.flatten(1), hconv_biases.flatten(1)], -1))
        self.assertTrue(torch.equal(out_conv, out_hconv))
        print('passed HyperFC 3D')


if __name__ == '__main__':
    unittest.main()
