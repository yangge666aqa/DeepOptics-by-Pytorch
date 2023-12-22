import torch
from torch import nn


class Res_Unet(nn.Module):
    def __init__(self, controlled_network_args):
        super(Res_Unet, self).__init__()
        self.network_args = controlled_network_args
        net_num = self.network_args['net_num']
        depth = self.network_args['depth']
        input_channel = self.network_args['image_channels']
        network_input = self.network_args['network_input']
        filter_root = self.network_args['filter_root']
        self.inner_block_res_down = nn.ModuleList()
        self.inner_block_res_up = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.remove_first_long_connection = self.network_args['remove_first_long_connection']
        self.up1 = nn.ModuleList()
        self.up_conv1 = nn.ModuleList()
        self.up_conv2 = nn.ModuleList()
        # 将无参数激活函数定义好
        self.act2 = nn.ELU()
        self.Maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        final_output = self.network_args['final_output_channels']

        for cur_net_num in range(net_num):
            for cur_depth in range(depth):
                output_channel = 2 ** cur_depth * filter_root
                index = cur_net_num * depth + cur_depth
                self.inner_block_res_down.append(nn.Conv2d(in_channels=input_channel,
                                                           out_channels=output_channel,
                                                           kernel_size=1, padding_mode='replicate'))

                # # First Conv2D Block with Conv2D, BN and activation

                self.conv2.append(nn.Sequential(
                    nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                              kernel_size=7 if cur_depth == 0 else 3, padding_mode='replicate',
                              padding=(3, 3) if cur_depth == 0 else (1, 1)),
                    nn.BatchNorm2d(num_features=output_channel),
                    nn.ELU(),
                    nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3,
                              padding_mode='replicate', padding=(1, 1)),
                    nn.BatchNorm2d(num_features=output_channel)))

                # # Second Conv2D block with Conv2D and BN only

                input_channel = output_channel

            for cur_depth in range(depth - 2, -1, -1):
                output_channel = 2 ** cur_depth * filter_root
                up_sampling_size1 = network_input[0] // int(2 ** cur_depth)
                up_sampling_size2 = network_input[1] // int(2 ** cur_depth)
                index = cur_net_num * (depth + 1) + cur_depth

                self.up1.insert(0, nn.Upsample(size=(up_sampling_size1, up_sampling_size2), mode='bilinear'))

                self.up_conv1.insert(0, nn.Conv2d(in_channels=output_channel * 2, out_channels=output_channel,
                                                  kernel_size=1, padding_mode='replicate'))

                self.up_conv2.insert(0, nn.Sequential(
                    nn.Conv2d(in_channels=output_channel * 2, out_channels=output_channel,
                              kernel_size=3, padding_mode='replicate', padding=(1, 1)),
                    nn.BatchNorm2d(num_features=output_channel),
                    nn.ELU(),
                    nn.Conv2d(in_channels=output_channel, out_channels=output_channel,
                              kernel_size=3, padding_mode='replicate', padding=(1, 1)),
                    nn.BatchNorm2d(num_features=output_channel)))

                self.inner_block_res_up.insert(0, nn.Conv2d(in_channels=output_channel * 2,
                                                            out_channels=output_channel,
                                                            kernel_size=1, padding_mode='replicate'))

        self.final_conv = nn.Conv2d(in_channels=output_channel, out_channels=final_output,
                                    kernel_size=1, padding_mode='replicate', )
        self.final_act = controlled_network_args['final_activation']()  # Sigmoid


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        net_num = self.network_args['net_num']
        depth = self.network_args['depth']
        long_connection_store = {}
        for cur_net_num in range(net_num):
            for cur_depth in range(depth):
                index = cur_net_num * depth + cur_depth
                inner_block_res_down = self.inner_block_res_down[index](x)
                conv_block = self.conv2[index](x)
                inner_block_res_added = torch.add(inner_block_res_down, conv_block)
                act2 = self.act2(inner_block_res_added)
                # Max pooling
                if cur_depth < depth - 1:  # 0-5层有长连接，6显然没有（看论文）
                    if self.remove_first_long_connection:  # remove_first_long_connection = False
                        if cur_depth > 0:
                            long_connection_store[str(cur_depth)] = act2
                    else:
                        long_connection_store[str(cur_depth)] = act2
                    x = self.Maxpooling(act2)  # strides=2
                else:
                    x = act2  # 对于第7层(depth=6),不在进行maxpooling,而在下面使用up1进行上采样
            for cur_depth in range(depth - 2, -1, -1):
                index = cur_net_num * (depth + 1) + cur_depth
                inner_block_res_up = self.inner_block_res_up[index](x)
                up1 = self.up1[index](x)
                up_conv1 = self.up_conv1[index](up1)
                if self.remove_first_long_connection and cur_depth == 0:  # remove_first_long_connection = False
                    # Dispose the first long connection if `remove_first_long_connection` is True.
                    up_long_connection_concat = up_conv1
                else:
                    long_connection = long_connection_store[str(cur_depth)]  # 取出之前存储的长连接
                    up_long_connection_concat = torch.cat((up_conv1, long_connection), dim=1)
                # 先concatenate，再用1024个3*3*2048的卷积核，将输出限制于1024；
                up_conv_block = self.up_conv2[index](up_long_connection_concat)
                inner_block_res_added_up = torch.add(self.inner_block_res_up[index](up_long_connection_concat)
                                                     , up_conv_block)
                x = self.act2(inner_block_res_added_up)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x
