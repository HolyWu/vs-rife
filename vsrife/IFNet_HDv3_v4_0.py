import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True
        ),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat) + feat
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear")
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self, scale=1, ensemble=False):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        self.scale = scale
        self.ensemble = ensemble

    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid):
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        scale_list = [8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](torch.cat((img0, img1, timestep), 1), None, scale=scale_list[i])
                if self.ensemble:
                    f1, m1 = block[i](torch.cat((img1, img0, 1 - timestep), 1), None, scale=scale_list[i])
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            else:
                f0, m0 = block[i](torch.cat((warped_img0, warped_img1, timestep, mask), 1), flow, scale=scale_list[i])
                if i == 1 and f0[:, :2].abs().max() > 32 and f0[:, 2:4].abs().max() > 32:
                    for k in range(4):
                        scale_list[k] *= 2
                    flow, mask = block[0](torch.cat((img0, img1, timestep), 1), None, scale=scale_list[0])
                    warped_img0 = warp(img0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
                    warped_img1 = warp(img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
                    f0, m0 = block[i](
                        torch.cat((warped_img0, warped_img1, timestep, mask), 1), flow, scale=scale_list[i]
                    )
                if self.ensemble:
                    f1, m1 = block[i](
                        torch.cat((warped_img1, warped_img0, 1 - timestep, -mask), 1),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale_list[i],
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 + (-m1)) / 2
                flow = flow + f0
                mask = mask + m0
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
            warped_img1 = warp(img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch.sigmoid(mask_list[3])
        return merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])
