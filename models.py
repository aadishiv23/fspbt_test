import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


#########################################################
# Upsampling Layer
#########################################################
class UpsamplingLayer(nn.Module):
    def __init__(self, channels, scale_factor=2, mode='nearest'):
        super(UpsamplingLayer, self).__init__()
        self.layer = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.layer(x)


#########################################################
# Self-Attention Layer (from SAGAN)
#########################################################
class SelfAttention(nn.Module):
    """Self-Attention layer as introduced in SAGAN:
    https://arxiv.org/abs/1805.08318
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batchsize, -1, width*height)   # B x (C/8) x N
        proj_key = self.key_conv(x).view(batchsize, -1, width*height)       # B x (C/8) x N
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)           # B x N x N
        attention = F.softmax(energy, dim=-1)                               # B x N x N
        proj_value = self.value_conv(x).view(batchsize, -1, width*height)   # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))             # B x C x N
        out = out.view(batchsize, C, width, height)
        out = self.gamma * out + x
        return out


#########################################################
# Residual Block with Pre-Activation and Dropout
#########################################################
class ResidualBlock(nn.Module):
    """
    A modified residual block with:
    - Pre-activation (norm -> relu -> conv)
    - Optional dropout
    Adapted for stable training and improved generalization.
    """

    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True), use_bias=False, dropout=0.0):
        super(ResidualBlock, self).__init__()
        layers = []
        # Pre-activation block
        if norm_layer is not None:
            layers += [norm_layer(dim)]
        layers += [activation]

        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]

        if norm_layer is not None:
            layers += [norm_layer(dim)]
        layers += [activation]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


#########################################################
# PixelShuffle Upconvolution
#########################################################
def upconv_layer_pixelshuffle(in_filters, out_filters, upscale_factor, norm_layer, nonlinearity, use_bias):
    parts = [
        nn.Conv2d(in_filters, out_filters * (upscale_factor ** 2), kernel_size=3, padding=1, bias=use_bias),
        nn.PixelShuffle(upscale_factor)
    ]

    if norm_layer is not None:
        parts.append(norm_layer(out_filters))
    if nonlinearity is not None:
        parts.append(nonlinearity)
    return nn.Sequential(*parts)


#########################################################
# Improved Generator
#########################################################
class GeneratorJ(nn.Module):
    def __init__(self, input_size=256,
                 norm_layer='instance_norm',
                 gpu_ids=None,
                 use_bias=False,
                 resnet_blocks=9,
                 tanh=False,
                 filters=(64, 128, 128, 128, 128, 64),
                 input_channels=3,
                 append_smoothers=False,
                 dropout=0.0,
                 use_attention=True,
                 activation=nn.ReLU(True)):
        super(GeneratorJ, self).__init__()
        self.input_size = input_size
        self.gpu_ids = gpu_ids
        self.use_bias = use_bias
        self.resnet_blocks = resnet_blocks
        self.append_smoothers = append_smoothers
        self.use_attention = use_attention

        # Choose normalization layer
        if norm_layer is None:
            norm = None
        elif norm_layer == 'batch_norm':
            norm = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm = nn.InstanceNorm2d
        elif norm_layer == 'group_norm':
            # Use fixed GroupNorm groups - can be tuned
            def group_norm_factory(num_features):
                return nn.GroupNorm(num_groups=32, num_channels=num_features)
            norm = group_norm_factory
        else:
            raise ValueError("Unknown normalization: {}".format(norm_layer))

        # Initial convolution layers
        self.conv0 = self._conv_block(input_channels, filters[0], 7, 1, 3, norm, activation, self.use_bias)
        self.conv1 = self._conv_block(filters[0], filters[1], 3, 2, 1, norm, activation, self.use_bias)
        self.conv2 = self._conv_block(filters[1], filters[2], 3, 2, 1, norm, activation, self.use_bias)

        # Residual blocks
        res_blocks = []
        for i in range(self.resnet_blocks):
            res_blocks.append(ResidualBlock(filters[2], norm_layer=norm, activation=activation, use_bias=self.use_bias, dropout=dropout))
        self.resnets = nn.Sequential(*res_blocks)

        # Optional Self-Attention after residual blocks
        if self.use_attention:
            self.attention = SelfAttention(filters[2])
        else:
            self.attention = nn.Identity()

        # Upsampling layers with skip connections
        self.upconv2 = upconv_layer_pixelshuffle(in_filters=filters[2] + filters[2],  # skip from conv2 output
                                                 out_filters=filters[3],
                                                 upscale_factor=2,
                                                 norm_layer=norm,
                                                 nonlinearity=activation,
                                                 use_bias=self.use_bias)

        self.upconv1 = upconv_layer_pixelshuffle(in_filters=filters[3] + filters[1],
                                                 out_filters=filters[4],
                                                 upscale_factor=2,
                                                 norm_layer=norm,
                                                 nonlinearity=activation,
                                                 use_bias=self.use_bias)

        # Final convolution layers with skip connection from x and output_0
        self.conv_11 = nn.Sequential(
            nn.Conv2d(filters[4] + filters[0] + input_channels, filters[5], kernel_size=7, stride=1, padding=3, bias=self.use_bias),
            activation
        )

        if self.append_smoothers:
            # Additional smoother layers
            smoother = [
                nn.Conv2d(filters[5], filters[5], kernel_size=3, padding=1, bias=self.use_bias),
                activation,
            ]
            if norm:
                smoother.append(norm(filters[5]))
            smoother += [
                nn.Conv2d(filters[5], filters[5], kernel_size=3, padding=1, bias=self.use_bias),
                activation
            ]
            self.conv_11_a = nn.Sequential(*smoother)
        else:
            self.conv_11_a = nn.Identity()

        if tanh:
            self.conv_12 = nn.Sequential(nn.Conv2d(filters[5], 3, kernel_size=1, stride=1, padding=0, bias=True),
                                         nn.Tanh())
        else:
            self.conv_12 = nn.Conv2d(filters[5], 3, kernel_size=1, stride=1, padding=0, bias=True)

    def _conv_block(self, in_c, out_c, k, s, p, norm, act, bias):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)]
        if norm is not None:
            layers.append(norm(out_c))
        if act is not None:
            layers.append(act)
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        output_0 = self.conv0(x)
        output_1 = self.conv1(output_0)
        output_2 = self.conv2(output_1)

        # Residual blocks + optional attention
        output = self.resnets(output_2)
        output = self.attention(output)

        # Decoder with skip connections
        # Skip connection at upconv2 from output_2
        output = self.upconv2(torch.cat((output, output_2), dim=1))
        # Skip connection at upconv1 from output_1
        output = self.upconv1(torch.cat((output, output_1), dim=1))
        # Skip connection at final from x and output_0
        output = self.conv_11(torch.cat((output, output_0, x), dim=1))
        output = self.conv_11_a(output)
        output = self.conv_12(output)
        return output


#########################################################
# Improved Discriminator with Spectral Normalization
#########################################################
def spectral_conv(in_c, out_c, k, s, p, bias=True):
    conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
    return nn.utils.spectral_norm(conv)

class DiscriminatorN_IN(nn.Module):
    def __init__(self, num_filters=64, input_channels=3, n_layers=3,
                 use_noise=False, noise_sigma=0.2, norm_layer='instance_norm', use_bias=True):
        super(DiscriminatorN_IN, self).__init__()

        self.num_filters = num_filters
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self.input_channels = input_channels
        self.use_bias = use_bias

        # Select normalization
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        elif norm_layer is None:
            self.norm_layer = None
        else:
            raise ValueError("Unknown norm layer type: {}".format(norm_layer))

        self.net = self.make_net(n_layers, self.input_channels, 1, 4, 2, self.use_bias)

    def make_block(self, flt_in, flt_out, k, stride, padding, bias, norm, relu=True):
        layers = [spectral_conv(flt_in, flt_out, k, stride, padding, bias)]
        if norm is not None:
            layers.append(norm(flt_out))
        if relu:
            layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def make_net(self, n, flt_in, flt_out=1, k=4, stride=2, bias=True):
        model = nn.Sequential()
        # First layer: no norm
        model.add_module('conv0', self.make_block(flt_in, self.num_filters, k, stride, 1, bias, None, True))

        # Subsequent layers
        flt_mult = 1
        for l in range(1, n):
            flt_mult_prev = flt_mult
            flt_mult = min(2**l, 8)
            model.add_module(f'conv_{l}', self.make_block(self.num_filters * flt_mult_prev,
                                                           self.num_filters * flt_mult,
                                                           k, stride, 1, bias, self.norm_layer, True))

        # One more layer with stride=1
        flt_mult_prev = flt_mult
        flt_mult = min(2**n, 8)
        model.add_module(f'conv_{n}', self.make_block(self.num_filters * flt_mult_prev,
                                                       self.num_filters * flt_mult,
                                                       k, 1, 1, bias, self.norm_layer, True))
        # Final output layer
        model.add_module('conv_out', self.make_block(self.num_filters * flt_mult, 1, k, 1, 1, bias, None, False))
        return model

    def forward(self, x):
        if self.use_noise and self.training:
            noise = torch.randn_like(x) * self.noise_sigma
            x = x + noise
        return self.net(x), None


#########################################################
# Improved Perceptual VGG19 for feature extraction
#########################################################
class PerceptualVGG19(nn.Module):
    def __init__(self, feature_layers, use_normalization=True, path=None):
        super(PerceptualVGG19, self).__init__()
        # Load VGG19
        # If you have a custom path to a pretrained model, load it here.
        # Otherwise, use torchvision's pretrained weights.
        if path is not None:
            print(f'Loading pre-trained VGG19 model from {path}')
            model = models.vgg19(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 8 * 8, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 40),
            )
            model.load_state_dict(torch.load(path))
        else:
            model = models.vgg19(pretrained=True)
        model.eval()

        self.model = model
        self.feature_layers = feature_layers
        self.use_normalization = use_normalization

        # Normalization constants for ImageNet
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])

        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

        self.mean_tensor = None
        self.std_tensor = None

        # No gradient required for perceptual model
        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, x):
        """
        Normalize input from [-1, 1] to ImageNet's mean/std.
        """
        if not self.use_normalization:
            return x

        if self.mean_tensor is None or self.std_tensor is None or self.mean_tensor.shape != x.shape:
            self.mean_tensor = self.mean.view(1, 3, 1, 1).expand_as(x)
            self.std_tensor = self.std.view(1, 3, 1, 1).expand_as(x)

        # x is assumed in [-1,1], bring it to [0,1] first
        x = (x + 1) / 2.0
        x = (x - self.mean_tensor) / self.std_tensor
        return x

    def extract_features(self, x):
        """
        Pass x through VGG19 and extract features from the specified layers.
        """
        features = []
        h = x
        max_layer = max(self.feature_layers)

        for i in range(max_layer + 1):
            h = self.model.features[i](h)
            if i in self.feature_layers:
                # Flatten features for easy usage
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)

        # Concatenate all chosen features
        features = torch.cat(features, dim=1)
        return features

    def forward(self, x):
        x = self.normalize(x)
        features = self.extract_features(x)
        return None, features
