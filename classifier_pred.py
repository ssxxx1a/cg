import argparse
import os

import blobfile as bf
import torch 
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
def cond_fn(x, t, y=None):
    assert y is not None
    classifier_scale=10.0
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


args = create_argparser().parse_args()
classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
# classifier.load_state_dict(
#     dist_util.load_state_dict(args.classifier_path, map_location="cpu")
# )
classifier.to('cuda')
if args.classifier_use_fp16:
    classifier.convert_to_fp16()
classifier.eval()
x=torch.rand(size=(1,3,32,32)).to('cuda')
indices = list(range(1000))[::-1]
t = torch.tensor([indices[0]] * x.shape[0], device='cuda')
y = torch.randint(low=0, high=NUM_CLASSES, size=(1,), device='cuda')
print(cond_fn(x,t,y).size())