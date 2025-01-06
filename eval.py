"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import enum
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import seaborn
import matplotlib.pyplot as plt
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy

# Enable need_weights on MultiheadedAttention block
def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap

# Hook used to log the attention weights
attention_hook = []
class AttentionHeatmapHook:
    def __init__(self):
        global attention_hook
        attention_hook = []

    def __call__(self, module, module_input, module_output):
        global attention_hook
        # TODO just capture a single sample for now
        if len(attention_hook) == 0:
            attn, attn_weights = module_output
            print(attn_weights.shape)
            if attn_weights == None:
                print('attn_weights not enabled, ensure need_weights=True is set')
                return
            # (batch, heads, target_seq_len, source_seq_len) 
            sample = attn_weights[0].detach().numpy()
            print('heatmap_hook', sample)
            attention_hook.append(sample)

            # graph the heatmap
            fig, axes = plt.subplots(len(sample), 1, figsize=(15,10))
            for i, ax in enumerate(axes.flat):
                seaborn.heatmap(sample[i], ax=ax)

            plt.tight_layout()
            plt.show()

    def clear(self):
        global attention_hook
        attention_hook = []

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-D', '--debug', is_flag=True)
def main(checkpoint, output_dir, device, debug):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    print(workspace)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    if debug:
        if isinstance(policy, DiffusionTransformerLowdimPolicy):
            print('adding attention heatmap hook')

            # attach hook to debug the attention
            # NOTE extracting attention from last decoder layer, could also investigate other layers 
            heatmap_hook = AttentionHeatmapHook()
            patch_attention(policy.model.decoder.layers[-1].multihead_attn)
            policy.model.decoder.layers[-1].multihead_attn.register_forward_hook(heatmap_hook)
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()


