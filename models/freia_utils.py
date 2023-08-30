import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):

    def __init__(self,
                 in_channels_list,
                 mid_channels=1024, 
                 add=False, 
                 condition_dims=None):
        super().__init__()
        output_channels = input_channels = sum(in_channels_list)
        self.condition_dims = condition_dims
        if condition_dims:
            input_channels += condition_dims
        mid_channels = input_channels // 4
        self.conv1 = nn.Conv2d(input_channels, mid_channels, 3, 1, 1)
        self.norm = nn.LayerNorm(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, output_channels, 3, 1, 1)
        self.split_channels = in_channels_list
        self.add = add
        self.relu = nn.ReLU(True)

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x]

        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = torch.split(out, self.split_channels, dim=1)
        out = [
            F.interpolate(a, size=s.size()[-2:], mode='bilinear', align_corners=True)
            for s, a in zip(x, out)
        ]
        if self.add:
            if self.condition_dims:
                x[-1] = x[-1].split([self.split_channels[-1], self.condition_dims], dim=1)[0]
            out = [s+a for s, a in zip(x, out)]

        return out

class FusionSTNets(nn.Module):

    def __init__(self, in_channels_list, mid_channels=512, rescale=False, condition_dims=None):
        super(FusionSTNets, self).__init__()

        self.cross_convs = FusionNet(in_channels_list, mid_channels, add=True, condition_dims=condition_dims)
        self.conv_s = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 3, 1, 1) for in_channels in in_channels_list])
        self.conv_t = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 3, 1, 1) for in_channels in in_channels_list])
        self.rescale = rescale
        if rescale:
            self.scales_s = nn.Parameter(torch.ones([len(in_channels_list)]))
            self.scales_t = nn.Parameter(torch.ones([len(in_channels_list)]))

    def forward(self, xs):
        out = self.cross_convs(xs)
        s = [conv(x) for conv, x in zip(self.conv_s, out)]
        t = [conv(x) for conv, x in zip(self.conv_t, out)]
        if self.rescale:
            s = [x * scale for x, scale in zip(s, self.scales_s)]
            t = [x * scale for x, scale in zip(t, self.scales_t)]
        out = [torch.cat([ys, yt], dim=1) for ys, yt in zip(s, t)]

        return out

class FusionCouplingLayer(nn.Module):
    def __init__(self, dims_in, dims_c=[], subnet=FusionSTNets, subnet_args={},
                 clamp=5.):
        super(FusionCouplingLayer, self).__init__()
        channels_list = [dim_in[0] for dim_in in dims_in]

        self.split1 = [channels // 2 for channels in channels_list]
        self.split2 = [channels - channels // 2 for channels in channels_list]

        if len(dims_c) != 0:
            self.condition = True
            condition_dims = dims_c[0][0]
        else:
            self.condition = False
            condition_dims = None

        self.clamp = clamp

        self.subnet_1 = subnet(self.split1, condition_dims=condition_dims, **subnet_args)
        self.subnet_2 = subnet(self.split2, condition_dims=condition_dims, **subnet_args)

    def e(self, s):
        if self.clamp > 0:
            return torch.exp(self.log_e(s))
        else:
            return torch.exp(s)

    def log_e(self, s):
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(s / self.clamp)
        else:
            return s

    def forward(self, x_list, c=[], rev=False, jac=True):
        x_spilt_list = list()
        for x, s1, s2 in zip(x_list, self.split1, self.split2):
            x_spilt_list.append(x.split([s1, s2], dim=1))
        x1_list = [x[0] for x in x_spilt_list]
        x2_list = [x[1] for x in x_spilt_list]

        if not rev:
            if self.condition:
                x2_list[-1] = torch.cat([x2_list[-1], c[0]], dim=1)
            st2_list = self.subnet_2(x2_list)
            s2_list = list()
            t2_list = list()
            for st, split in zip(st2_list, self.split1):
                s, t = st.split([split, split], dim=1)
                s2_list.append(s)
                t2_list.append(t)

            y1_list = list()
            for x1, s2, t2, split in zip(x1_list, s2_list, t2_list, self.split1):
                y1_list.append(self.e(s2) * x1[:, :split] + t2)

            if self.condition:
                y1_list[-1] = torch.cat([y1_list[-1], c[0]], dim=1)
            st1_list = self.subnet_1(y1_list)
            s1_list = list()
            t1_list = list()
            for st, split in zip(st1_list, self.split2):
                s, t = st.split([split, split], dim=1)
                s1_list.append(s)
                t1_list.append(t)

            y2_list = list()
            for x2, s1, t1, split in zip(x2_list, s1_list, t1_list, self.split2):
                y2_list.append(self.e(s1) * x2[:, :split] + t1)

        else:  # names of x and y are swapped!
            if self.condition:
                x1_list[-1] = torch.cat([x1_list[-1], c[0]], dim=1)
            st1_list = self.subnet_1(x1_list)
            s1_list = list()
            t1_list = list()
            for st, split in zip(st1_list, self.split2):
                s, t = st.split([split, split], dim=1)
                s1_list.append(s)
                t1_list.append(t)

            y2_list = list()
            for x2, s1, t1 in zip(x2_list, s1_list, t1_list):
                y2_list.append((x2 - t1) / self.e(s1))

            if self.condition:
                y2_list[-1] = torch.cat([y2_list[-1], c[0]], dim=1)
            st2_list = self.subnet_2(y2_list)
            s2_list = list()
            t2_list = list()
            for st, split in zip(st2_list, self.split2):
                s, t = st.split([split, split], dim=1)
                s2_list.append(s)
                t2_list.append(t)

            y1_list = list()
            for x1, s2, t2 in zip(x1_list, s2_list, t2_list):
                y1_list.append((x1 - t2) / self.e(s2))

        y_list = list()
        for y1, y2 in zip(y1_list, y2_list):
            y = torch.cat([y1, y2], dim=1)
            y_list.append(torch.clamp(y, -1e6, 1e6))

        self.jac = []
        for s1, s2 in zip(s1_list, s2_list):
            self.jac.append(torch.sum(self.log_e(s1), dim=(1, 2, 3)) + torch.sum(self.log_e(s2), dim=(1, 2, 3)))

        # return y_list, self.jac
        return y_list, sum(self.jac)

    def log_jacobian(self, x, rev=False):
        return self.jac

    def output_dims(self, input_dims):
        return input_dims
