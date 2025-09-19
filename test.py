import torch.cuda
import argparse
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
from llava.llava_agent import LLavaAgent
from SUPIR.CKPT_PTH import LLAVA_MODEL_PATH
from SUPIR.utils.model_fetch import get_model
import os

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# hyperparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--edm_steps", type=int, default=50)
parser.add_argument("--s_stage1", type=int, default=-1)
parser.add_argument("--s_churn", type=int, default=5)
parser.add_argument("--s_noise", type=float, default=1.003)
parser.add_argument("--s_cfg", type=float, default=4.0)
parser.add_argument("--s_stage2", type=float, default=1.)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--a_prompt", type=str,
                    default='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                            'hyper sharpness, perfect without deformations.')
parser.add_argument("--n_prompt", type=str,
                    default='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                            'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, over-smooth')
parser.add_argument("--color_fix_type", type=str, default='Wavelet', choices=["None", "AdaIn", "Wavelet"])
parser.add_argument("--linear_CFG", action='store_true', default=True)
parser.add_argument("--linear_s_stage2", action='store_true', default=False)
parser.add_argument("--spt_linear_CFG", type=float, default=1.0)
parser.add_argument("--spt_linear_s_stage2", type=float, default=0.)
parser.add_argument("--weight_dtype", type=str, default="bf16", choices=['fp32', 'bf16', 'fp16'])
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
parser.add_argument("--load_4bit_llava", action='store_true', default=False)
parser.add_argument("--sampler", type=str, default="DPMPP2M")
args = parser.parse_args()
print(args)
use_llava = not args.no_llava

# load SUPIR
model_cfg = "options/SUPIR_v0_tiled.yaml" if args.use_tile_vae else "options/SUPIR_v0.yaml"
model = create_SUPIR_model(model_cfg, weight_dtype=args.weight_dtype, supir_sign=args.SUPIR_sign, 
                          device=SUPIR_device, sampler=args.sampler)

if args.use_tile_vae:
    model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)

# load LLaVA
if use_llava:
    llava_path = get_model('liuhaotian/llava-v1.5-7b')
    llava_agent = LLavaAgent(llava_path, device=LLaVA_device, load_8bit=args.load_8bit_llava, 
                            load_4bit=args.load_4bit_llava)
else:
    llava_agent = None

os.makedirs(args.save_dir, exist_ok=True)
for img_pth in os.listdir(args.img_dir):
    img_name = os.path.splitext(img_pth)[0]

    LQ_ips = Image.open(os.path.join(args.img_dir, img_pth))
    LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upscale=args.upscale, min_size=args.min_size)
    LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

    # step 1: Pre-denoise for LLaVA, resize to 512
    LQ_img_512, h1, w1 = PIL2Tensor(LQ_ips, upscale=args.upscale, min_size=args.min_size, do_fix_resize=512)
    LQ_img_512 = LQ_img_512.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    clean_imgs = model.batchify_denoise(LQ_img_512)
    clean_PIL_img = Tensor2PIL(clean_imgs[0], h1, w1)

    # step 2: LLaVA
    if use_llava:
        captions = llava_agent.gen_image_caption([clean_PIL_img])
    else:
        captions = ['']
    print(captions)

    # step 3: Diffusion Process
    samples = model.batchify_sample(LQ_img, captions, num_steps=args.edm_steps, restoration_scale=args.s_stage1, 
                                   s_churn=args.s_churn, s_noise=args.s_noise, cfg_scale=args.s_cfg, 
                                   control_scale=args.s_stage2, seed=args.seed, num_samples=args.num_samples, 
                                   p_p=args.a_prompt, n_p=args.n_prompt, color_fix_type=args.color_fix_type,
                                   use_linear_cfg=args.linear_CFG, use_linear_control_scale=args.linear_s_stage2,
                                   cfg_scale_start=args.spt_linear_CFG, control_scale_start=args.spt_linear_s_stage2,
                                   sampler_cls=f"sgm.modules.diffusionmodules.sampling.RestoreDPMPP2MSampler")
    # save
    for _i, sample in enumerate(samples):
        Tensor2PIL(sample, h0, w0).save(f'{args.save_dir}/{img_name}_{_i}.png')