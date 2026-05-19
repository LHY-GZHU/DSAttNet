import torch
from torch import nn
import torch.nn.functional as F


def global_median_pooling(x):  

    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled 


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1)) 
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)

        avg_out = self.fc1(avg_pool)
        avg_out = F.relu(avg_out, inplace=True) 
        avg_out = self.fc2(avg_out)
        avg_out = torch.sigmoid(avg_out)

        
        max_out = self.fc1(max_pool)
        max_out = F.relu(max_out, inplace=True) 
        max_out = self.fc2(max_out) 
        max_out = torch.sigmoid(max_out) 

        median_out = self.fc1(median_pool) 
        median_out = F.relu(median_out, inplace=True)
        median_out = self.fc2(median_out)
        median_out = torch.sigmoid(median_out)

        out = avg_out + max_out + median_out
        return out


class DWAM(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super(DWAM , self).__init__()

        self.C = in_channels
        self.O = out_channels
        assert in_channels == out_channels, "Input and output channels must be the same"
        self.channel_attention = ChannelAttention(input_channels=in_channels,
                                                  internal_neurons=in_channels // channel_attention_reduce)

        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

        self.depth_convs = nn.ModuleList([

            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.pointwise_conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.channel_attention(inputs)
        inputs = channel_att_vec * inputs

        initial_out = self.initial_depth_conv(inputs)

        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        spatial_att = self.pointwise_conv(spatial_out)
        out = spatial_att * inputs
        out = self.pointwise_conv(out)
        return out


if __name__ == '__main__':
    # 假设输入数据
    batch_size = 4
    channels = 16
    height = 64
    width = 64
    input_tensor = torch.randn(batch_size, channels, height, width).cuda()

    cpca_block = DWAM (in_channels=16, out_channels=16, channel_attention_reduce=4).cuda()

    output_tensor = cpca_block(input_tensor)

    print(f"Output shape: {output_tensor.shape}")
