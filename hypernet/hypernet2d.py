import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import numpy as np


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, norm='forward', dim=[-2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                             dtype=self.weights1.dtype)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum('abxy,bdxy->adxy',
                                                                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum('abxy,bdxy->adxy',
                                                                 x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # out_ft[:, :, :self.modes1, :self.modes2] = \
        #     compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # out_ft[:, :, -self.modes1:, :self.modes2] = \
        #     compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, norm='forward', dim=[-2, -1], s=(x.size(-2), x.size(-1)))
        return x


class HyperSpectralConv2d(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = nn.Parameter(
        #     self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
        # self.weights2 = nn.Parameter(
        #     self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
        self.weight_size = torch.prod(torch.tensor(self.get_weights_size())).item()

    @jit.script_method
    def forward(self, x, weights):
        batchsize = x.shape[0]
        weights1, weights2 = weights[..., :weights.shape[-1] // 2], weights[..., weights.shape[-1] // 2:]
        weights1 = torch.view_as_complex(weights1.view(batchsize, *self.get_weights_size(), 2).contiguous())
        weights2 = torch.view_as_complex(weights2.view(batchsize, *self.get_weights_size(), 2).contiguous())

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, norm='forward', dim=[-2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                             dtype=torch.complex64)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum('abxy,abdxy->adxy',
                                                                x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum('abxy,abdxy->adxy',
                                                                 x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, norm='forward', dim=[-2, -1], s=(x.size(-2), x.size(-1)))
        return x

    def get_weights_size(self):
        return self.in_channels, self.out_channels, self.modes1, self.modes2

    def get_size(self):
        return self.weight_size * 4


class HyperConv2d(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_size = torch.prod(torch.tensor(self.get_weights_size())).item()
        self.bias_size = torch.prod(torch.tensor(self.get_bias_size())).item()

    @jit.script_method
    def forward(self, x, w):
        w, b = w[..., :self.weight_size], w[..., self.weight_size:]
        w, b = w.view(x.shape[0], *self.get_weights_size()), b.view(x.shape[0], *self.get_bias_size())
        out = F.conv2d(x.reshape(1, x.shape[0] * x.shape[1], x.shape[2], x.shape[3]),
                       w.reshape(w.shape[0] * w.shape[1], w.shape[2], w.shape[3], w.shape[4]),
                       b.reshape(-1),
                       groups=x.shape[0])
        out = out.reshape(x.shape[0], w.shape[1], out.shape[2], out.shape[3])
        return out

    def get_weights_size(self):
        return self.out_channels, self.in_channels, self.kernel_size, self.kernel_size

    def get_bias_size(self):
        return (self.out_channels,)

    def get_size(self):
        return self.weight_size + self.bias_size


class HyperFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_size = torch.prod(torch.tensor(self.get_weights_size())).item()
        self.bias_size = torch.prod(torch.tensor(self.get_bias_size())).item()

    def forward(self, x, w):
        w, b = w[..., :self.weight_size], w[..., self.weight_size:]
        w, b = w.view(x.shape[0], *self.get_weights_size()), b.view(x.shape[0], *self.get_bias_size())
        return torch.einsum('a...c,adc->a...d', x, w) + b.unsqueeze(1).unsqueeze(1)

        # return F.linear(x, w, b)

    def get_weights_size(self):
        return self.out_channels, self.in_channels

    def get_bias_size(self):
        return (self.out_channels,)

    def get_size(self):
        return self.weight_size + self.bias_size


class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width, args, **kwargs):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        in_channels = kwargs.get('in_channels', 3)
        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, in_channels - 2)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        # x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w0(x)

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        # x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w1(x)

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        # x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w2(x)

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        # x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w3(x)

        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleBlock2dGeLU(SimpleBlock2d):
    def __init__(self, modes1, modes2, width, args, **kwargs):
        super().__init__(modes1, modes2, width, args, **kwargs)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        # x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w0(x)

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        # x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w1(x)

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        # x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w2(x)

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        # x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x2 = self.w3(x)

        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class HyperSimpleBlock2d(jit.ScriptModule):
    def __init__(self, modes1, modes2, width, in_channels=3):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = HyperFC(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = HyperSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = HyperSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = HyperSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = HyperSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = HyperConv2d(self.width, self.width, 1)
        self.w1 = HyperConv2d(self.width, self.width, 1)
        self.w2 = HyperConv2d(self.width, self.width, 1)
        self.w3 = HyperConv2d(self.width, self.width, 1)

        self.fc1 = HyperFC(self.width, 128)
        self.fc2 = HyperFC(128, in_channels - 2)

        self.weight_sizes = [self.fc0.get_size(),
                             self.conv0.get_size(),
                             self.w0.get_size(),
                             self.conv1.get_size(),
                             self.w1.get_size(),
                             self.conv2.get_size(),
                             self.w2.get_size(),
                             self.conv3.get_size(),
                             self.w3.get_size(),
                             self.fc1.get_size(),
                             self.fc2.get_size(),
                             ]

        self.hyper = nn.Sequential(*[nn.Linear(1, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 32),
                                     nn.ReLU(),
                                     nn.Linear(32,
                                               torch.sum(torch.tensor(self.weight_sizes, dtype=torch.long)).item())])

    @jit.script_method
    def forward(self, x, t):
        weights = self.hyper(t)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        i = 0
        counter = 0
        x = self.fc0(x, weights[..., counter:counter + self.fc0.get_size()])
        counter += self.fc0.get_size()
        i += 1

        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x, weights[..., counter:counter + self.conv0.get_size()])
        counter += self.conv0.get_size()
        i += 1

        # x2 = self.w0(x.flatten(2), weights[..., counter:counter + self.w0.get_size()])
        x2 = self.w0(x, weights[..., counter:counter + self.w0.get_size()])

        # x2 = x2.view(batchsize, self.width, size_x, size_y)
        counter += self.w0.get_size()
        i += 1

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x, weights[..., counter:counter + self.conv1.get_size()])
        counter += self.conv1.get_size()
        i += 1

        # x2 = self.w1(x.flatten(2), weights[..., counter:counter + self.w1.get_size()])
        x2 = self.w1(x, weights[..., counter:counter + self.w1.get_size()])

        # x2 = x2.view(batchsize, self.width, size_x, size_y)
        counter += self.w1.get_size()
        i += 1

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x, weights[..., counter:counter + self.conv2.get_size()])
        counter += self.conv2.get_size()
        i += 1

        # x2 = self.w2(x.flatten(2), weights[..., counter:counter + self.w2.get_size()])
        x2 = self.w2(x, weights[..., counter:counter + self.w2.get_size()])
        # x2 = x2.view(batchsize, self.width, size_x, size_y)

        counter += self.w2.get_size()
        i += 1

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x, weights[..., counter:counter + self.conv3.get_size()])
        counter += self.conv3.get_size()
        i += 1

        # x2 = self.w3(x.flatten(2), weights[..., counter:counter + self.w3.get_size()])
        x2 = self.w3(x, weights[..., counter:counter + self.w3.get_size()])

        # x2 = x2.view(batchsize, self.width, size_x, size_y)

        counter += self.w3.get_size()
        i += 1

        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x, weights[..., counter:counter + self.fc1.get_size()])
        counter += self.fc1.get_size()
        i += 1

        x = F.relu(x)
        x = self.fc2(x, weights[..., counter:])

        return x


class HyperSimpleBlock2dGeLU(HyperSimpleBlock2d):
    def __init__(self, modes1, modes2, width, in_channels=3):
        super().__init__(modes1, modes2, width, in_channels=in_channels)

    @jit.script_method
    def forward(self, x, t):
        weights = self.hyper(t)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        i = 0
        counter = 0
        x = self.fc0(x, weights[..., counter:counter + self.fc0.get_size()])
        counter += self.fc0.get_size()
        i += 1

        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x, weights[..., counter:counter + self.conv0.get_size()])
        counter += self.conv0.get_size()
        i += 1

        # x2 = self.w0(x.flatten(2), weights[..., counter:counter + self.w0.get_size()])
        x2 = self.w0(x, weights[..., counter:counter + self.w0.get_size()])

        # x2 = x2.view(batchsize, self.width, size_x, size_y)
        counter += self.w0.get_size()
        i += 1

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x, weights[..., counter:counter + self.conv1.get_size()])
        counter += self.conv1.get_size()
        i += 1

        # x2 = self.w1(x.flatten(2), weights[..., counter:counter + self.w1.get_size()])
        x2 = self.w1(x, weights[..., counter:counter + self.w1.get_size()])

        # x2 = x2.view(batchsize, self.width, size_x, size_y)
        counter += self.w1.get_size()
        i += 1

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x, weights[..., counter:counter + self.conv2.get_size()])
        counter += self.conv2.get_size()
        i += 1

        # x2 = self.w2(x.flatten(2), weights[..., counter:counter + self.w2.get_size()])
        x2 = self.w2(x, weights[..., counter:counter + self.w2.get_size()])
        # x2 = x2.view(batchsize, self.width, size_x, size_y)

        counter += self.w2.get_size()
        i += 1

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x, weights[..., counter:counter + self.conv3.get_size()])
        counter += self.conv3.get_size()
        i += 1

        # x2 = self.w3(x.flatten(2), weights[..., counter:counter + self.w3.get_size()])
        x2 = self.w3(x, weights[..., counter:counter + self.w3.get_size()])

        # x2 = x2.view(batchsize, self.width, size_x, size_y)

        counter += self.w3.get_size()
        i += 1

        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x, weights[..., counter:counter + self.fc1.get_size()])
        counter += self.fc1.get_size()
        i += 1

        x = F.gelu(x)
        x = self.fc2(x, weights[..., counter:])

        return x


if __name__ == '__main__':
    # Testing HyperSpectralConv1d
    # batch_size = 1
    # hn = HyperSpectralConv2d(32, 32, 16, 16)
    # a = torch.rand(batch_size, 32, 64, 64)
    # weight = torch.rand(batch_size, hn.get_size())
    # print(hn(a, weight).shape)

    # Testing whether full hypernetwork works
    hn = HyperSimpleBlock2d(16, 16, 64)
    # x = torch.rand(16, 85, 85, 3)
    # t = torch.rand(16, 1)
    # loss = hn(x, t).sum()
    # loss.backward()
