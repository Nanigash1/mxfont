import argparse
from pathlib import Path
import torch
from utils import refine, save_tensor_to_image
from datasets import get_test_loader
from models import Generator
from sconf import Config
from train import setup_transforms

def eval_ckpt():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", help="path to weight to evaluate.pth")
    parser.add_argument("--result_dir", help="path to save the result file")
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="mxfont/cfgs/defaults.yaml")
    cfg.argv_update(left_argv)
    img_dir = Path(args.result_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    trn_transform, val_transform = setup_transforms(cfg)

    g_kwargs = cfg.get('g_args', {})
    gen = Generator(1, cfg.C, 1, **g_kwargs).cuda()

    weight = torch.load(args.weight)
    if "generator_ema" in weight:
        weight = weight["generator_ema"]
        
    # Handle shape mismatch manually
    model_dict = gen.state_dict()
    gen.load_state_dict(weight)
    # pretrained_dict = {k: v for k, v in weight.items() if k in model_dict and v.size() == model_dict[k].size()}
    # model_dict.update(pretrained_dict)
    # gen.load_state_dict(model_dict) 

    test_dset, test_loader = get_test_loader(cfg, val_transform)

    for batch in test_loader:
        style_imgs = batch["style_imgs"].cuda()
        char_imgs = batch["source_imgs"].unsqueeze(1).cuda()

        out = gen.gen_from_style_char(style_imgs, char_imgs)
        fonts = batch["fonts"]
        chars = batch["chars"]

        for image, font, char in zip(refine(out), fonts, chars):
            (img_dir / font).mkdir(parents=True, exist_ok=True)
            prefix = 'upper_' if char.isupper() else 'lower_'
            path = img_dir / font / f"{prefix}{char}.png"
            save_tensor_to_image(image, path)

if __name__ == "__main__":
    eval_ckpt()
