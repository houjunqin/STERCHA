import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtV2_Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        return identity + self.drop_path(x)


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECABasicBlock(nn.Module):
    def __init__(self, inplanes, planes, num_nodes, k_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        self.ln1 = nn.LayerNorm(num_nodes)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, bias=True)
        self.ln2 = nn.LayerNorm(num_nodes)
        self.eca = eca_layer(planes, k_size)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out = self.eca(out)
        return out + residual


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_nodes, stride=1, k_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        self.ln1 = nn.LayerNorm(num_nodes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.ln2 = nn.LayerNorm(num_nodes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=True)
        self.ln3 = nn.LayerNorm(num_nodes)
        self.gelu = nn.GELU()
        self.eca = eca_layer(planes * 4, k_size)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.gelu(out)
        out = self.drop(out)

        out = self.conv3(out)
        out = self.ln3(out)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.eca(out)
        return out + residual


class ConvCAnext(nn.Module):
    def __init__(self, inplanes, planes, num_nodes, stride=1, k_size=3):
        super().__init__()
        self.dwconv = nn.Conv1d(inplanes, inplanes, kernel_size=7, padding=3, groups=inplanes)
        self.norm = nn.LayerNorm(num_nodes)
        self.pwconv1 = nn.Conv1d(inplanes, 4 * inplanes, kernel_size=1)
        self.grn = nn.LayerNorm(num_nodes)
        self.pwconv2 = nn.Conv1d(4 * inplanes, inplanes, kernel_size=1)

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        self.ln1 = nn.LayerNorm(num_nodes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.ln2 = nn.LayerNorm(num_nodes)
        self.conv3 = nn.Conv1d(planes, inplanes, kernel_size=1, bias=True)
        self.ln3 = nn.LayerNorm(num_nodes)

        self.gelu = nn.GELU()
        self.eca = eca_layer(inplanes, k_size)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.pwconv1(out)
        out = self.grn(out)
        out = self.gelu(out)
        out = self.pwconv2(out)

        out = self.conv1(out)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.gelu(out)

        out = self.conv3(out)
        out = self.ln3(out)
        out = self.eca(out)

        return out + residual


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        return self.relu(out + x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        hidden = self.fc2(self.drop(self.act(self.fc1(x))))
        return hidden + x


class STERCHA(nn.Module):
    def __init__(self, model_args):
        super().__init__()

        self.num_nodes = model_args.get("num_nodes", 8809)

        self.input_features = model_args.get("input_feature_idx", [0])
        self.output_features = model_args.get("output_feature_idx", [0])

        # derive dimensions from feature index lists, not from config integers
        self.input_dim = len(self.input_features)
        self.output_dim = len(self.output_features)

        self.input_time_steps = model_args.get("input_len", 12)
        self.output_time_steps = model_args.get("horizon", 12)
        self.num_layers = model_args.get("num_layers", 4)
        self.encoder_module = model_args.get("encoder_module", "MLP")

        self.time_of_day_size = model_args.get("sampling_freq_day", 95)
        self.day_of_week_size = model_args.get("day_of_week_size", 7)

        self.input_embedding_dim = model_args.get("input_embedding_dim", 512)
        self.manin_embedding_dim = model_args.get("manin_embedding_dim", 48)
        self.bathy_embedding_dim = model_args.get("bathy_embedding_dim", 48)
        self.tod_embedding_dim = model_args.get("tod_embedding_dim", 32)
        self.dow_embedding_dim = model_args.get("dow_embedding_dim", 0)
        self.spatial_embedding_dim = model_args.get("spatial_embedding_dim", 128)

        self.model_dim = (
            (self.input_embedding_dim if self.input_embedding_dim > 0 else 0)
            + (self.tod_embedding_dim if self.tod_embedding_dim > 0 else 0)
            + (self.dow_embedding_dim if self.dow_embedding_dim > 0 else 0)
            + (self.spatial_embedding_dim if self.spatial_embedding_dim > 0 else 0)
            + (self.bathy_embedding_dim if self.bathy_embedding_dim > 0 else 0)
            + (self.manin_embedding_dim if self.manin_embedding_dim > 0 else 0)
        )

        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)

        if self.bathy_embedding_dim > 0:
            self.batlnr = nn.Linear(1, self.bathy_embedding_dim)
            nn.init.xavier_uniform_(self.batlnr.weight)
            nn.init.constant_(self.batlnr.bias, 0)

        if self.manin_embedding_dim > 0:
            self.manlnr = nn.Linear(1, self.manin_embedding_dim)
            nn.init.xavier_uniform_(self.manlnr.weight)
            nn.init.constant_(self.manlnr.bias, 0)

        if self.tod_embedding_dim > 0:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.tod_embedding_dim))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.dow_embedding_dim > 0:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.dow_embedding_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        if self.input_embedding_dim > 0:
            self.input_ts_emb_layer = nn.Conv1d(
                in_channels=self.input_dim * self.input_time_steps,
                out_channels=self.input_embedding_dim,
                kernel_size=1,
                bias=True,
            )

        self.regression_layer = nn.Conv2d(
            in_channels=self.model_dim,
            out_channels=self.output_time_steps * self.output_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        
        match self.encoder_module:
            case "MLP":
                self.encoder = nn.Sequential(
                    *[MultiLayerPerceptron(self.model_dim, self.model_dim) for _ in range(self.num_layers)]
                )
            case "ResNet":
                self.encoder = nn.Sequential(
                    *[ResBlock1D(self.model_dim, self.model_dim) for _ in range(self.num_layers)]
                )
            case "ECABasicBlock":
                self.encoder = nn.Sequential(
                    *[ECABasicBlock(self.model_dim, self.model_dim, self.num_nodes) for _ in range(self.num_layers)]
                )
            case "ECABottleneck":
                self.encoder = nn.Sequential(
                    *[ECABottleneck(self.model_dim, self.model_dim // 4, self.num_nodes) for _ in range(self.num_layers)]
                )
            case "ConvNeXtV2_Block":
                self.encoder = nn.Sequential(
                    *[ConvNeXtV2_Block(self.model_dim) for _ in range(self.num_layers)]
                )
            case "ConvCAnext":
                self.encoder = nn.Sequential(
                    *[ConvCAnext(self.model_dim, self.model_dim, self.num_nodes) for _ in range(self.num_layers)]
                )
            case _:
                raise ValueError(f"Unknown encoder_module: {self.encoder_module}")

    def _expand_static_feature(self, feat: torch.Tensor, batch_size: int, t: int) -> torch.Tensor:
        if feat.dim() == 1:
            feat = feat.view(1, 1, -1, 1)
        elif feat.dim() == 2:
            if feat.shape[-1] == 1:
                feat = feat.unsqueeze(0).unsqueeze(0)
            else:
                feat = feat.unsqueeze(-1).unsqueeze(1)
        elif feat.dim() == 3:
            feat = feat.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected static feature shape: {feat.shape}")

        if feat.shape[-1] != 1:
            raise ValueError(f"Static feature must have last dim=1, got {feat.shape}")

        return feat.expand(batch_size, t, -1, -1).contiguous()

    def _encode(self, history_data, tod, dow, bathymetry, manning):
        batch_size, t, num_nodes, _ = history_data.shape
        features = []

        if self.bathy_embedding_dim > 0:
            with torch.autocast(device_type=history_data.device.type, enabled=False):
                bath = self._expand_static_feature(bathymetry.float(), batch_size, t)[:, -1]
                bat_emb = self.batlnr(bath)
            bat_emb = bat_emb.transpose(1, 2).contiguous().to(history_data.dtype)
            features.append(bat_emb)

        if self.manin_embedding_dim > 0:
            with torch.autocast(device_type=history_data.device.type, enabled=False):
                mann = self._expand_static_feature(manning.float(), batch_size, t)[:, -1]
                man_emb = self.manlnr(mann)
            man_emb = man_emb.transpose(1, 2).contiguous().to(history_data.dtype)
            features.append(man_emb)

        if self.tod_embedding_dim > 0:
            tod_idx = tod[:, -1, :].long().clamp_(0, self.time_of_day_size - 1)
            time_in_day_emb = self.time_in_day_emb[tod_idx].transpose(1, 2).contiguous()
            features.append(time_in_day_emb)

        if self.dow_embedding_dim > 0:
            dow_idx = dow[:, -1, :].long().clamp_(0, self.day_of_week_size - 1)
            day_in_week_emb = self.day_in_week_emb[dow_idx].transpose(1, 2).contiguous()
            features.append(day_in_week_emb)

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).contiguous()
            features.append(spatial_emb)

        if self.input_embedding_dim > 0:
            input_data = history_data.transpose(1, 2).contiguous()   # [B,N,T,C]
            input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).contiguous()
            time_series_emb = self.input_ts_emb_layer(input_data)     # [B,EL,N]
            features.append(time_series_emb)

        timefeatures = torch.cat(features, dim=1)                     # [B,E,N]
        hidden = self.encoder(timefeatures).unsqueeze(-1)             # [B,E,N,1]
        return hidden

    def _decode_current(self, hidden, batch_size, num_nodes):
        pred = self.regression_layer(hidden)                          # [B,L*Cout,N,1]
        pred = pred.squeeze(-1)                                       # [B,L*Cout,N]
        pred = pred.view(batch_size, self.output_time_steps, self.output_dim, num_nodes)
        pred = pred.permute(0, 1, 3, 2).contiguous()                 # [B,L,N,Cout]
        return pred

    def forward(
        self,
        history_data: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        bathymetry: torch.Tensor,
        manning: torch.Tensor,
        mode: str = "current",
        rollout_steps: int | None = None,
    ) -> torch.Tensor:
        if mode == "current":
            batch_size, _, num_nodes, _ = history_data.shape
            hidden = self._encode(history_data, tod, dow, bathymetry, manning)
            return self._decode_current(hidden, batch_size, num_nodes)

        if mode == "rollout":
            steps = self.output_time_steps if rollout_steps is None else rollout_steps
            return self.rollout(history_data, tod, dow, bathymetry, manning, steps=steps)

        raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def rollout(
        self,
        history_data: torch.Tensor,
        tod: torch.Tensor,
        dow: torch.Tensor,
        bathymetry: torch.Tensor,
        manning: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """
        Autoregressive rollout.
        Assumes the model predicts [B, L, N, C], but uses only the first step
        each iteration for recursion.
        """
        cur_x = history_data.clone()
        cur_tod = tod.clone()
        cur_dow = dow.clone()

        preds = []
        batch_size, _, num_nodes, _ = history_data.shape

        for _ in range(steps):
            hidden = self._encode(cur_x, cur_tod, cur_dow, bathymetry, manning)
            pred_block = self._decode_current(hidden, batch_size, num_nodes)   # [B,L,N,C]
            pred_1 = pred_block[:, 0:1]                                        # [B,1,N,C]
            preds.append(pred_1)

            # slide input window
            if pred_1.shape[-1] != cur_x.shape[-1]:
                # only replace the leading output channels, keep the rest if needed
                next_frame = cur_x[:, -1:].clone()
                next_frame[..., :pred_1.shape[-1]] = pred_1
            else:
                next_frame = pred_1

            cur_x = torch.cat([cur_x[:, 1:], next_frame], dim=1)

            # shift tod/dow by one step if already provided as sequences
            cur_tod = torch.cat([cur_tod[:, 1:], cur_tod[:, -1:]], dim=1)
            cur_dow = torch.cat([cur_dow[:, 1:], cur_dow[:, -1:]], dim=1)

        return torch.cat(preds, dim=1)   # [B,steps,N,C]