import sys
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import torch
import hydra
import dill
from copy import deepcopy
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

def load_model(path):
    payload = torch.load(open(path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # diffusion model
    model: BaseImagePolicy
    model = workspace.model

    return model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage: merge.py <checkpoint_A> <checkpoint_B>')
        sys.exit(1)

    model_a = load_model(sys.argv[1])
    print(model_a)

    model_b = load_model(sys.argv[2])
    print(model_b)

    if model_a.state_dict().keys() != model_b.state_dict().keys():
        print('models do not have the same architecture')
        sys.exit(1)

    # perform the merge
    #merged_model = deepcopy(model_a)
    #for param_merged, param_a, param_b in zip(merged_model.parameters(), model_a.parameters(), model_b.parameters()):
    #    # TODO any rounding errors here?
    #    param_merged.data = (param_a.data + param_b.data) / 2.0

