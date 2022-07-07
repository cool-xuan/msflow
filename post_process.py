import numpy as np
import torch
import torch.nn.functional as F

def post_process(c, size_list, outputs_list):
    print('Multi-scale sizes:', size_list)
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]
    for l, outputs in enumerate(outputs_list):
        # output = torch.tensor(output, dtype=torch.double)
        outputs = torch.cat(outputs, 0)
        logp_maps[l] = F.interpolate(outputs.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
        output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prob_map = torch.exp(output_norm) # convert to probs in range [0:1]
        prop_maps[l] = F.interpolate(prob_map.unsqueeze(1),
                size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
    
    logp_map = sum(logp_maps)
    logp_map-= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prop_map_mul = torch.exp(logp_map)
    anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
    batch = anomaly_score_map_mul.shape[0]
    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)
    anomaly_score = np.mean(
        anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)

    prop_map_add = sum(prop_maps)
    prop_map_add = prop_map_add.detach().cpu().numpy()
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()