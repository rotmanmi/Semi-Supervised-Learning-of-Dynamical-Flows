import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.jit as jit


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.complex64))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.complex64))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.complex64))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.complex64))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, norm='forward', dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                             dtype=torch.complex64)
        out_ft[..., :self.modes1, :self.modes2, :self.modes3] = torch.einsum('abxyz,bdxyz->adxyz',
                                                                             x_ft[..., :self.modes1,
                                                                             :self.modes2,
                                                                             :self.modes3],
                                                                             self.weights1)
        out_ft[..., -self.modes1:, :self.modes2, :self.modes3] = torch.einsum('abxyz,bdxyz->adxyz',
                                                                              x_ft[..., -self.modes1:,
                                                                              :self.modes2,
                                                                              :self.modes3],
                                                                              self.weights2)
        out_ft[..., :self.modes1, -self.modes2:, :self.modes3] = torch.einsum('abxyz,bdxyz->adxyz',
                                                                              x_ft[..., :self.modes1,
                                                                              -self.modes2:,
                                                                              :self.modes3],
                                                                              self.weights3)
        out_ft[..., -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum('abxyz,bdxyz->adxyz',
                                                                               x_ft[..., -self.modes1:,
                                                                               -self.modes2:,
                                                                               :self.modes3],
                                                                               self.weights4)

        # out_ft[:, :, :self.modes1, :self.modes2] = \
        #     compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # out_ft[:, :, -self.modes1:, :self.modes2] = \
        #     compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, norm='forward', dim=[-3, -2, -1], s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class HyperSpectralConv3d(jit.ScriptModule):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weight_size = torch.tensor(self.get_weights_size())

    @jit.script_method
    def forward(self, x, weights):
        batchsize = x.shape[0]
        weights1, weights2, weights3, weights4 = (weights[..., :weights.shape[-1] // 4],
                                                  weights[..., weights.shape[-1] // 4:weights.shape[-1] // 2],
                                                  weights[..., weights.shape[-1] // 2:-weights.shape[-1] // 4],
                                                  weights[..., -weights.shape[-1] // 4:])
        weights1 = torch.view_as_complex(weights1.view(batchsize, *self.get_weights_size(), 2).contiguous())
        weights2 = torch.view_as_complex(weights2.view(batchsize, *self.get_weights_size(), 2).contiguous())
        weights3 = torch.view_as_complex(weights3.view(batchsize, *self.get_weights_size(), 2).contiguous())
        weights4 = torch.view_as_complex(weights4.view(batchsize, *self.get_weights_size(), 2).contiguous())

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, norm='forward', dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                             dtype=torch.complex64)
        out_ft[..., :self.modes1, :self.modes2, :self.modes3] = torch.einsum('abxyz,abdxyz->adxyz',
                                                                             x_ft[..., :self.modes1,
                                                                             :self.modes2,
                                                                             :self.modes3],
                                                                             weights1)
        out_ft[..., -self.modes1:, :self.modes2, :self.modes3] = torch.einsum('abxyz,abdxyz->adxyz',
                                                                              x_ft[..., -self.modes1:,
                                                                              :self.modes2,
                                                                              :self.modes3],
                                                                              weights2)
        out_ft[..., :self.modes1, -self.modes2:, :self.modes3] = torch.einsum('abxyz,abdxyz->adxyz',
                                                                              x_ft[..., :self.modes1,
                                                                              -self.modes2:,
                                                                              :self.modes3],
                                                                              weights3)
        out_ft[..., -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum('abxyz,abdxyz->adxyz',
                                                                               x_ft[..., -self.modes1:,
                                                                               -self.modes2:,
                                                                               :self.modes3],
                                                                               weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, norm='forward', dim=[-3, -2, -1], s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def get_weights_size(self):
        return self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3

    def get_size(self):
        return torch.prod(self.weight_size) * 8


class HyperConv3d(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_size = torch.tensor(self.get_weights_size())
        self.bias_size = torch.tensor(self.get_bias_size())

    @jit.script_method
    def forward(self, x, w):
        w, b = w[..., :torch.prod(self.weight_size)], w[..., torch.prod(self.weight_size):]
        w, b = w.view(x.shape[0], *self.get_weights_size()), b.view(x.shape[0], *self.get_bias_size())
        out = F.conv3d(x.reshape(1, x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]),
                       w.reshape(w.shape[0] * w.shape[1], w.shape[2], w.shape[3], w.shape[4], w.shape[5]),
                       b.reshape(-1),
                       groups=x.shape[0])
        out = out.reshape(x.shape[0], w.shape[1], out.shape[2], out.shape[3], out.shape[4])
        return out

    def get_weights_size(self):
        return self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size

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
        return torch.einsum('a...c,adc->a...d', x, w) + b.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # return F.linear(x, w, b)

    def get_weights_size(self):
        return self.out_channels, self.in_channels

    def get_bias_size(self):
        return (self.out_channels,)

    def get_size(self):
        return torch.prod(self.weight_size) + torch.prod(self.bias_size)


class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, args, **kwargs):
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
        self.modes3 = modes3
        self.width = width
        in_channels = kwargs.get('in_channels', 6)
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, in_channels - 3)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

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

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleBlock3dGeLU(SimpleBlock3d):
    def __init__(self, modes1, modes2, modes3, width, args, **kwargs):
        super().__init__(modes1, modes2, modes3, width, args, **kwargs)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

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

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class HyperSimpleBlock3d(jit.ScriptModule):
    def __init__(self, modes1, modes2, modes3, width, in_channels=6):
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
        self.modes3 = modes3
        self.width = width
        self.fc0 = HyperFC(in_channels, self.width)

        self.conv0 = HyperSpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = HyperSpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = HyperSpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = HyperSpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = HyperConv3d(self.width, self.width, 1)
        self.w1 = HyperConv3d(self.width, self.width, 1)
        self.w2 = HyperConv3d(self.width, self.width, 1)
        self.w3 = HyperConv3d(self.width, self.width, 1)

        self.fc1 = HyperFC(self.width, 128)
        self.fc2 = HyperFC(128, in_channels - 3)

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
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        i = 0
        counter = 0
        x = self.fc0(x, weights[..., counter:counter + self.fc0.get_size()])
        counter += self.fc0.get_size()
        i += 1

        x = x.permute(0, 4, 1, 2, 3)

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

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x, weights[..., counter:counter + self.fc1.get_size()])
        counter += self.fc1.get_size()
        i += 1

        x = F.relu(x)
        x = self.fc2(x, weights[..., counter:])

        return x


class HyperSimpleBlock3dGeLU(HyperSimpleBlock3d):
    def __init__(self, modes1, modes2, modes3, width, in_channels=6):
        super().__init__(modes1, modes2, modes3, width, in_channels=6)

    @jit.script_method
    def forward(self, x, t):
        weights = self.hyper(t)
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        i = 0
        counter = 0
        x = self.fc0(x, weights[..., counter:counter + self.fc0.get_size()])
        counter += self.fc0.get_size()
        i += 1

        x = x.permute(0, 4, 1, 2, 3)

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

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x, weights[..., counter:counter + self.fc1.get_size()])
        counter += self.fc1.get_size()
        i += 1

        x = F.gelu(x)
        x = self.fc2(x, weights[..., counter:])

        return x


if __name__ == '__main__':
    # Testing SpectralConv3d
    # batch_size = 1
    # a = torch.rand(batch_size, 32, 64, 64, 64)
    #
    # hn = SpectralConv3d(32, 32, 16, 16, 16)
    # print(hn(a).shape)

    # Testing HyperSpectralConv3d
    # batch_size = 1
    # hn = HyperSpectralConv3d(32, 32, 16, 16, 16)
    # a = torch.rand(batch_size, 32, 64, 64, 64)
    # weight = torch.rand(batch_size, hn.get_size())
    # print(hn(a, weight).shape)

    # Testing HyperConv3d
    # batch_size = 1
    # hn = HyperConv3d(32, 32, 16)
    # a = torch.rand(batch_size, 32, 64, 64, 64)
    # weight = torch.rand(batch_size, hn.get_size())
    # print(hn(a, weight).shape)

    # Testing whether full hypernetwork works
    hn = HyperSimpleBlock3d(4, 4, 4, 20).cuda()
    x = torch.rand(16, 64, 64, 64, 6).cuda()
    t = torch.rand(16, 1).cuda()
    loss = hn(x, t).sum()
    loss.backward()
    print(loss)
