import os
import gc
import argparse
from PIL import Image
import torch
from torch.nn.functional import interpolate

from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--min_size", type=int, default=256)
parser.add_argument("--edm_steps", type=int, default=12)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--ae_dtype", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
parser.add_argument("--diff_dtype", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
parser.add_argument("--no_llava", action='store_true', default=True)
parser.add_argument("--loading_half_params", action='store_true', default=True)
parser.add_argument("--use_tile_vae", action='store_true', default=True)
parser.add_argument("--encoder_tile_size", type=int, default=128)
parser.add_argument("--decoder_tile_size", type=int, default=32)
parser.add_argument("--load_8bit_llava", action='store_true', default=True)
parser.add_argument("--upfront_llava_cpu", action='store_true', default=False,
                    help='Load LLaVA on CPU to reduce GPU residency (slower).')
parser.add_argument("--swap_model_cpu", action='store_true', default=False,
                    help='Move SUPIR model back to CPU between images to free VRAM.')
parser.add_argument("--clear_llava_after_caption", action='store_true', default=True,
                    help='Unload LLaVA after caption generation to free memory.')
parser.add_argument("--edm_safety_steps", type=int, default=2,
                    help='Extra safety steps; keep small.')
parser.add_argument("--a_prompt", type=str, default='')
parser.add_argument("--n_prompt", type=str, default='')
parser.add_argument("--color_fix_type", type=str, default='Wavelet', choices=["None","AdaIn","Wavelet"])
args = parser.parse_args()
print(args)
use_llava = not args.no_llava

# ----------------- device mapping -----------------
if torch.cuda.is_available():
    ndev = torch.cuda.device_count()
    if ndev >= 2:
        SUPIR_device = torch.device('cuda:0')
        LLaVA_device = torch.device('cuda:1')
    else:
        SUPIR_device = torch.device('cuda:0')
        LLaVA_device = torch.device('cuda:0')
else:
    raise RuntimeError('CUDA required by this script')

# ----------------- utilities -----------------
def free_gpu():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

# Move model helper that avoids accidental grad mode
def move_model(model, device):
    return model.to(device)

# ----------------- load SUPIR model (small footprint defaults) -----------------
model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign=args.SUPIR_sign)
# prefer fp16
if args.loading_half_params:
    try:
        model = model.half()
    except Exception:
        pass
# enforce dtypes used by model internals
model.ae_dtype = convert_dtype(args.ae_dtype)
model.model.dtype = convert_dtype(args.diff_dtype)
# init tiled VAE with small tiles
if args.use_tile_vae:
    model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)

# Optionally keep model on CPU between images to reduce steady VRAM
if args.swap_model_cpu:
    model = move_model(model, torch.device('cpu'))
else:
    model = move_model(model, SUPIR_device)

# ----------------- load LLaVA lazily -----------------
llava_agent = None
if use_llava:
    # allow loading on CPU to reduce GPU memory pressure if requested
    llava_dev = torch.device('cpu') if args.upfront_llava_cpu else LLaVA_device
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=llava_dev, load_8bit=args.load_8bit_llava, load_4bit=False)

os.makedirs(args.save_dir, exist_ok=True)

# speed/precision contexts
amp_dtype = torch.float16 if args.diff_dtype == 'fp16' else torch.bfloat16 if args.diff_dtype == 'bf16' else torch.float32

# ----------------- main loop -----------------
file_list = sorted([f for f in os.listdir(args.img_dir) if os.path.isfile(os.path.join(args.img_dir, f))])
for img_pth in file_list:
    img_name = os.path.splitext(img_pth)[0]
    pil_img = Image.open(os.path.join(args.img_dir, img_pth)).convert('RGB')

    # 1) Create small 512px image for denoise+caption. Do this first to keep heavy ops small.
    # Use PIL resizing to keep CPU memory usage low then convert to tensor only when needed.
    small_pil = pil_img.resize((512, 512), resample=Image.LANCZOS)

    # Bring model to GPU if swap mode
    if args.swap_model_cpu:
        model = move_model(model, SUPIR_device)
        free_gpu()

    # run fast denoise on small image
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            # convert to tensor for model denoise
            LQ_small, h1, w1 = PIL2Tensor(small_pil, upsacle=1, min_size=512, fix_resize=512)
            LQ_small = LQ_small.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
            clean_imgs = model.batchify_denoise(LQ_small)
            # move single clean image to CPU PIL for captioning and to reduce GPU hold
            clean_PIL_img = Tensor2PIL(clean_imgs[0].cpu(), h1, w1)
            # free immediate small tensors
            del LQ_small, clean_imgs
    free_gpu()

    # 2) Caption with LLaVA if enabled. Runs on CPU device if llava_agent was loaded on CPU.
    captions = ['']
    if use_llava and llava_agent is not None:
        try:
            captions = llava_agent.gen_image_caption([clean_PIL_img])
        except Exception:
            captions = ['']
        # optionally unload LLaVA to free memory if it sits on GPU
        if args.clear_llava_after_caption and hasattr(llava_agent, 'cleanup'):
            try:
                llava_agent.cleanup()
            except Exception:
                pass
        # dereference
        if args.clear_llava_after_caption:
            del llava_agent
            llava_agent = None
            free_gpu()

    # 3) Create full-res LQ tensor just-in-time. Use lower min_size to save memory.
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
            LQ_img, h0, w0 = PIL2Tensor(pil_img, upsacle=args.upscale, min_size=args.min_size)
            LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

            # Ensure batch size 1
            LQ_img = LQ_img[:1]

            # 4) Sampling. Keep steps small and move outputs to CPU immediately.
            samples = model.batchify_sample(
                LQ_img,
                captions,
                num_steps=args.edm_steps,
                restoration_scale=getattr(args, 's_stage1', -1),
                s_churn=getattr(args, 's_churn', 5),
                s_noise=getattr(args, 's_noise', 1.01),
                cfg_scale=getattr(args, 's_cfg', 4.0),
                control_scale=getattr(args, 's_stage2', 1.0),
                seed=args.seed,
                num_samples=args.num_samples,
                p_p=getattr(args, 'a_prompt', ''),
                n_p=getattr(args, 'n_prompt', ''),
                color_fix_type=getattr(args, 'color_fix_type', 'Wavelet'),
                use_linear_CFG=getattr(args, 'linear_CFG', True),
                use_linear_control_scale=getattr(args, 'linear_s_stage2', False),
                cfg_scale_start=getattr(args, 'spt_linear_CFG', 1.0),
                control_scale_start=getattr(args, 'spt_linear_s_stage2', 0.0)
            )

            # Save outputs as soon as they are on CPU. Do not keep GPU copies.
            for idx, sample in enumerate(samples):
                try:
                    out = sample.cpu()
                except Exception:
                    out = sample
                Tensor2PIL(out, h0, w0).save(os.path.join(args.save_dir, f"{img_name}_{idx}.png"))
                # free
                try:
                    del out
                except Exception:
                    pass

            # cleanup per-image
            del samples, LQ_img
    free_gpu()

    # move model back to CPU between images if requested to release steady VRAM
    if args.swap_model_cpu:
        model = move_model(model, torch.device('cpu'))
        free_gpu()

print('Done')
