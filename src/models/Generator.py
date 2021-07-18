from torch import nn


class ResNextLayer(nn.Module):
    def __init__(self, in_ch=128, channels=[64, 64, 128], kernel_size=3):
        super(ResNextLayer, self).__init__()
        ch1, ch2, ch3 = channels
        self.conv1 = ConvLayer(in_ch, ch1, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvLayer(ch1, ch2, kernel_size=kernel_size, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = ConvLayer(ch2, ch3, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        out = out + identity
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution 
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size,
                                                 output_padding)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 32, 3, 2),
            nn.ReLU(),
            ConvLayer(32, 32, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResNextLayer(32, [16, 16, 32], kernel_size=3),
            ResNextLayer(32, [16, 16, 32], kernel_size=3),
            ResNextLayer(32, [16, 16, 32], kernel_size=3),
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(32, 32, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(32, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out
