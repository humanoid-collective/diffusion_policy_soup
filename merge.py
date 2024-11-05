import sys
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import torch
import hydra
import dill
import pathlib
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

    return workspace, payload

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('usage: merge.py <checkpoint_A> <checkpoint_B> <output_path>')
        sys.exit(1)

    output_path = pathlib.Path(sys.argv[3])

    workspace_a, payload = load_model(sys.argv[1])
    model_a: BaseImagePolicy = workspace_a.model

    # print(model_a)
    # print(model_a.state_dict().keys())

    workspace_b, _ = load_model(sys.argv[2])
    model_b: BaseImagePolicy = workspace_b.model
    # print(model_b)

    if model_a.state_dict().keys() != model_b.state_dict().keys():
        print('models do not have the same architecture')
        sys.exit(1)

    for param in model_a.state_dict().keys():
        shape_a = model_a.state_dict()[param].shape 
        shape_b = model_b.state_dict()[param].shape
        if shape_a != shape_b:
            print(f"shape mismatch at {param} - a:{shape_a}, b:{shape_b}")
            sys.exit(1)

    # perform the merge
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    new_state_dict = {}
    for param in state_a:
        # TODO is this proper way to mutate the state dict?
        # TODO any rounding errors here?
        new_state_dict[param] = (state_a[param] + state_b[param]) / 2.0

    workspace_a.model.load_state_dict(new_state_dict)

    # save the merged model
    # TODO use the workspace.save_checkpoint for this
    # workspace_a.state_dicts
    # new_payload = {
    #     'cfg': payload['cfg'],
    #     'state_dicts': state_a,
    #     'pickles': payload['pickles'],
    # }
    # torch.save(new_payload, output_path.open('wb'), pickle_module=dill)
    workspace_a.save_checkpoint(path=output_path)

