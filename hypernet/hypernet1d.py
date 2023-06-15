import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

import numpy as np


class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width, args):
        super(SimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)
        self.circular = 'circular' if args['circular'] else 'zeros'

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1, padding_mode=self.circular)
        self.w1 = nn.Conv1d(self.width, self.width, 1, padding_mode=self.circular)
        self.w2 = nn.Conv1d(self.width, self.width, 1, padding_mode=self.circular)
        self.w3 = nn.Conv1d(self.width, self.width, 1, padding_mode=self.circular)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleBlock1dGELU(SimpleBlock1d):
    def __init__(self, modes, width, args):
        super().__init__(modes, width, args)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.complex64),
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm='forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1) // 2 + 1, device=x.device,
                             dtype=self.weights1.dtype)
        out_ft[:, :, :self.modes1] = torch.einsum('abc,bdc->adc', x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, norm='forward')
        return x


class HyperSpectralConv1d(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, modes1):
        super(HyperSpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weight_size = torch.tensor(self.get_weights_size())

    @jit.script_method
    def forward(self, x, weights1):
        batchsize = x.shape[0]
        weights1 = torch.view_as_complex(weights1.view(batchsize, *self.get_weights_size(), 2).contiguous())

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm='forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1) // 2 + 1, device=x.device, dtype=weights1.dtype)
        out_ft[:, :, :self.modes1] = torch.einsum('abc,abdc->adc', x_ft[:, :, :self.modes1], weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, norm='forward')
        return x

    def get_weights_size(self):
        return self.in_channels, self.out_channels, self.modes1

    def get_size(self):
        return torch.prod(self.weight_size) * 2


class HyperConv1d(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_size = torch.tensor(self.get_weights_size())
        self.bias_size = torch.tensor(self.get_bias_size())

    def forward(self, x, w):
        w, b = w[..., :torch.prod(self.weight_size)], w[..., torch.prod(self.weight_size):]
        w, b = w.view(x.shape[0], *self.get_weights_size()), b.view(x.shape[0], *self.get_bias_size())
        out = F.conv1d(x.reshape(1, -1, x.shape[-1]), w.reshape(-1, w.shape[2], w.shape[3]), b.reshape(-1),
                       groups=x.shape[0])
        return out.reshape(x.shape[0], -1, out.shape[-1])

    def get_weights_size(self):
        return self.out_channels, self.in_channels, self.kernel_size

    def get_bias_size(self):
        return (self.out_channels,)

    def get_size(self):
        return torch.prod(self.weight_size) + torch.prod(self.bias_size)


class HyperFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_size = torch.tensor(self.get_weights_size())
        self.bias_size = torch.tensor(self.get_bias_size())

    def forward(self, x, w):
        w, b = w[..., :torch.prod(self.weight_size)], w[..., torch.prod(self.weight_size):]
        w, b = w.view(x.shape[0], *self.get_weights_size()), b.view(x.shape[0], *self.get_bias_size())
        return torch.einsum('abc,adc->abd', x, w) + b.unsqueeze(1)

        # return F.linear(x, w, b)

    def get_weights_size(self):
        return self.out_channels, self.in_channels

    def get_bias_size(self):
        return (self.out_channels,)

    def get_size(self):
        return torch.prod(self.weight_size) + torch.prod(self.bias_size)


class HyperSimpleBlock1d(jit.ScriptModule):
    def __init__(self, modes, width):
        super(HyperSimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = HyperFC(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = HyperSpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = HyperSpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = HyperSpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = HyperSpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = HyperConv1d(self.width, self.width, 1)
        self.w1 = HyperConv1d(self.width, self.width, 1)
        self.w2 = HyperConv1d(self.width, self.width, 1)
        self.w3 = HyperConv1d(self.width, self.width, 1)

        self.fc1 = HyperFC(self.width, 128)
        self.fc2 = HyperFC(128, 1)

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
        batch_size = t.shape[0]

        i = 0
        counter = 0
        x = self.fc0(x, weights[..., counter:counter + self.fc0.get_size()])
        counter += self.fc0.get_size()
        i += 1

        x = x.permute(0, 2, 1)

        x1 = self.conv0(x, weights[..., counter:counter + self.conv0.get_size()])
        counter += self.conv0.get_size()
        i += 1

        x2 = self.w0(x, weights[..., counter:counter + self.w0.get_size()])
        counter += self.w0.get_size()
        i += 1

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x, weights[..., counter:counter + self.conv1.get_size()])
        counter += self.conv1.get_size()
        i += 1

        x2 = self.w1(x, weights[..., counter:counter + self.w1.get_size()])
        counter += self.w1.get_size()
        i += 1

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x, weights[..., counter:counter + self.conv2.get_size()])
        counter += self.conv2.get_size()
        i += 1

        x2 = self.w2(x, weights[..., counter:counter + self.w2.get_size()])
        counter += self.w2.get_size()
        i += 1

        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x, weights[..., counter:counter + self.conv3.get_size()])
        counter += self.conv3.get_size()
        i += 1

        x2 = self.w3(x, weights[..., counter:counter + self.w3.get_size()])
        counter += self.w3.get_size()
        i += 1

        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x, weights[..., counter:counter + self.fc1.get_size()])
        counter += self.fc1.get_size()
        i += 1

        x = F.relu(x)
        x = self.fc2(x, weights[..., counter:])

        return x


class HyperSimpleBlock1dGELU(HyperSimpleBlock1d):
    def __init__(self, modes, width):
        super(HyperSimpleBlock1dGELU, self).__init__(modes, width)

    @jit.script_method
    def forward(self, x, t):
        weights = self.hyper(t)
        batch_size = t.shape[0]

        i = 0
        counter = 0
        x = self.fc0(x, weights[..., counter:counter + self.fc0.get_size()])
        counter += self.fc0.get_size()
        i += 1

        x = x.permute(0, 2, 1)

        x1 = self.conv0(x, weights[..., counter:counter + self.conv0.get_size()])
        counter += self.conv0.get_size()
        i += 1

        x2 = self.w0(x, weights[..., counter:counter + self.w0.get_size()])
        counter += self.w0.get_size()
        i += 1

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x, weights[..., counter:counter + self.conv1.get_size()])
        counter += self.conv1.get_size()
        i += 1

        x2 = self.w1(x, weights[..., counter:counter + self.w1.get_size()])
        counter += self.w1.get_size()
        i += 1

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x, weights[..., counter:counter + self.conv2.get_size()])
        counter += self.conv2.get_size()
        i += 1

        x2 = self.w2(x, weights[..., counter:counter + self.w2.get_size()])
        counter += self.w2.get_size()
        i += 1

        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x, weights[..., counter:counter + self.conv3.get_size()])
        counter += self.conv3.get_size()
        i += 1

        x2 = self.w3(x, weights[..., counter:counter + self.w3.get_size()])
        counter += self.w3.get_size()
        i += 1

        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x, weights[..., counter:counter + self.fc1.get_size()])
        counter += self.fc1.get_size()
        i += 1

        x = F.gelu(x)
        x = self.fc2(x, weights[..., counter:])

        return x


class Hyper1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


if __name__ == '__main__':
    # Testing HyperConv1d:

    # hn = HyperConv1d(32,128,1)
    # a = torch.rand(1,32,64)
    # weight = torch.rand(*hn.get_weights_size())
    # bias = torch.rand(*hn.get_bias_size())
    # print(hn(a,weight,bias).shape)

    # Testing HyperSpectralConv1d
    # hn = HyperSpectralConv1d(32, 32, 16)
    # a = torch.rand(1, 32, 64)
    # weight = torch.rand(*hn.get_weights_size(), dtype=torch.complex64)
    # print(hn(a, weight).shape)

    # Testing HyperFC
    # hn = HyperFC(32, 16)
    # a = torch.rand(1, 32)
    # weight = torch.rand(*hn.get_weights_size())
    # bias = torch.rand(*hn.get_bias_size())
    # print(hn(a, weight, bias).shape)

    # Testing whether full hypernetwork works
    hn = HyperSimpleBlock1d(16, 64)
    x = torch.rand(16, 1024, 2)
    t = torch.rand(16, 1)
    loss = hn(x, t).sum()
    loss.backward()
