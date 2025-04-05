import argparse
import calendar
import matplotlib.pyplot as plt
import os
import time
import torch
import torchaudio
import warnings
import wandb
from torch import inference_mode
import re             # <--- Added Import
import pathlib        # <--- Added Import

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.ddim_inversion import ddim_inversion, text2image_ldm_stable
from models import load_model
from utils import set_reproducability, load_audio, get_spec


HF_TOKEN = None  # Needed for stable audio open. You can leave None when not using it


# --- Added Helper Function ---
def sanitize_filename(name):
    """Removes invalid characters and limits length for file/directory names."""
    # Remove characters invalid for Windows/Linux paths
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    # Replace sequences of whitespace with a single underscore
    name = re.sub(r'\s+', '_', name)
    # Remove leading/trailing dots or underscores
    name = name.strip('._')
    # Limit length (optional, but good practice)
    max_len = 100 # Adjust as needed
    if len(name) > max_len:
        name = name[:max_len]
    return name if name else "untitled" # Return "untitled" if sanitization results in empty string
# --- End Helper Function ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run text-based audio editing.')
    parser.add_argument("--device_num", type=int, default=0, help="GPU device number")
    parser.add_argument('-s', "--seed", type=int, default=None, help="GPU device number")
    parser.add_argument("--model_id", type=str, choices=["cvssp/audioldm-s-full-v2",
                                                         "cvssp/audioldm-l-full",
                                                         "cvssp/audioldm2",
                                                         "cvssp/audioldm2-large",
                                                         "cvssp/audioldm2-music",
                                                         'declare-lab/tango-full-ft-audio-music-caps',
                                                         'declare-lab/tango-full-ft-audiocaps',
                                                         "stabilityai/stable-audio-open-1.0"
                                                         ],
                        default="cvssp/audioldm2-music", help='Audio diffusion model to use')

    parser.add_argument("--init_aud", type=str, required=True, help='Audio to invert and extract PCs from')
    parser.add_argument("--cfg_src", type=float, nargs='+', default=[3],
                        help='Classifier-free guidance strength for forward process')
    parser.add_argument("--cfg_tar", type=float, nargs='+', default=[12],
                        help='Classifier-free guidance strength for reverse process')
    parser.add_argument("--num_diffusion_steps", type=int, default=200,
                        help="Number of diffusion steps. TANGO and AudioLDM2 are recommended to be used with 200 steps"
                             ", while AudioLDM is recommeneded to be used with 100 steps")
    parser.add_argument("--target_prompt", type=str, nargs='+', default=[""], required=True,
                        help="Prompt to accompany the reverse process. Should describe the wanted edited audio.")
    parser.add_argument("--source_prompt", type=str, nargs='+', default=[""],
                        help="Prompt to accompany the forward process. Should describe the original audio.")
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help="Negative prompt to accompany the inversion and generation process")
    parser.add_argument("--tstart", type=int, nargs='+', default=[100],
                        help="Diffusion timestep to start the reverse process from. Controls editing strength.")
    parser.add_argument("--results_path", type=str, default="results", help="path to dump results")

    parser.add_argument("--cutoff_points", type=float, nargs='*', default=None)
    parser.add_argument("--mode", default="ours", choices=['ours', 'ddim'],
                        help="Run our editing or DDIM inversion based editing.")
    parser.add_argument("--fix_alpha", type=float, default=0.1)

    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true', default=True)

    args = parser.parse_args()
    args.eta = 1.
    args.numerical_fix = True
    args.test_rand_gen = False

    # Check HF Token requirement AFTER parsing args
    if args.model_id == "stabilityai/stable-audio-open-1.0":
        # Allow reading from environment variable OR the hardcoded one
        hf_token_to_use = os.environ.get("HF_TOKEN", HF_TOKEN)
        if hf_token_to_use is None:
             raise ValueError("HF_TOKEN is required for stabilityai/stable-audio-open-1.0 model. "
                              "Set it in this script or as an environment variable (HF_TOKEN).")
    else:
        hf_token_to_use = None # Not needed for other models


    print("Setting up device and reproducibility...")
    set_reproducability(args.seed, extreme=False)
    device = f"cuda:{args.device_num}"
    try:
        torch.cuda.set_device(args.device_num)
    except Exception as e:
        raise RuntimeError(f"Failed to set CUDA device {args.device_num}. Is it available? Error: {e}")

    model_id = args.model_id
    cfg_scale_src = args.cfg_src
    cfg_scale_tar = args.cfg_tar

    # --- WANDB Setup (if used) ---
    sanitized_model_name_wandb = sanitize_filename(model_id.split('/')[-1])
    sanitized_audio_stem_wandb = sanitize_filename(pathlib.Path(args.init_aud).stem)
    default_wandb_name = f"{sanitized_model_name_wandb}_{sanitized_audio_stem_wandb}_{int(time.time())}" # Add timestamp

    if not args.wandb_disable:
        print("Initializing wandb...")
        try:
            wandb.login(key='') # Add key or rely on env var/login
            wandb_run = wandb.init(project="AudInv", entity='', config={}, # Add your entity
                                   name=args.wandb_name if args.wandb_name is not None else default_wandb_name,
                                   group=args.wandb_group,
                                   mode='online', # Explicitly online unless disabled
                                   settings=wandb.Settings(_disable_stats=True))
            wandb.config.update(args)
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}. Disabling wandb.")
            args.wandb_disable = True
            wandb_run = None
    else:
         print("wandb logging is disabled.")
         wandb_run = None


    eta = args.eta
    if len(args.tstart) != len(args.target_prompt):
        if len(args.tstart) == 1: args.tstart *= len(args.target_prompt)
        else: raise ValueError("T-start amount and target prompt amount don't match.")
    args.tstart = torch.tensor(args.tstart, dtype=torch.int)
    skip = args.num_diffusion_steps - args.tstart

    # --- Load Model and Audio ---
    print(f"Loading model: {model_id}...")
    ldm_stable = load_model(model_id, device, args.num_diffusion_steps, token=hf_token_to_use)
    print(f"Loading audio: {args.init_aud}...")
    try:
        # --- Pass duration from load_audio if returned ---
        load_result = load_audio(args.init_aud, ldm_stable.get_fn_STFT(), device=device,
                                      stft=('stable-audio' not in model_id), model_sr=ldm_stable.get_sr())

        # Unpack results based on what load_audio returns
        if len(load_result) == 3:
             x0, sr, duration = load_result
        elif len(load_result) == 2: # If load_audio was simplified and only returns x0, sr
             x0, sr = load_result
             duration = x0.shape[-1] / sr # Calculate duration manually if needed
             print("Manually calculated duration from loaded waveform.")
        else:
            raise ValueError(f"Unexpected number of return values from load_audio: {len(load_result)}")

    except FileNotFoundError:
         print(f"ERROR: Input audio file not found: {args.init_aud}")
         raise # Re-raise the error to stop the script
    except Exception as e:
         print(f"ERROR: Failed to load audio file '{args.init_aud}'. Error: {e}")
         # Consider if the error from load_audio might be due to format (handled by RIFF fix now)
         raise # Re-raise

    torch.cuda.empty_cache()

    # --- Inversion Process ---
    print("Starting inversion process (VAE encode)...")
    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

        if args.mode == "ddim":
            print("Performing DDIM inversion...")
            if len(cfg_scale_src) > 1: raise ValueError("DDIM only supports one cfg_scale_src")
            wT = ddim_inversion(ldm_stable, w0, args.source_prompt, cfg_scale_src[0],
                                num_inference_steps=args.num_diffusion_steps, skip=skip[0].item()) # Pass scalar skip
        else: # Ours
            print("Performing forward process (diffusion)...")
            wt, zs, wts, extra_info = inversion_forward_process(
                ldm_stable, w0, etas=eta, prompts=args.source_prompt, cfg_scales=cfg_scale_src,
                prog_bar=True, num_inference_steps=args.num_diffusion_steps,
                cutoff_points=args.cutoff_points, numerical_fix=args.numerical_fix, duration=duration
            )

        # --- Path Generation (MODIFIED) ---
        results_base_path = pathlib.Path(args.results_path)
        model_folder_name = sanitize_filename(model_id.split('/')[-1])
        original_audio_stem = pathlib.Path(args.init_aud).stem
        sanitized_audio_folder_name = sanitize_filename(original_audio_stem)

        base_save_dir = results_base_path / model_folder_name / sanitized_audio_folder_name
        os.makedirs(base_save_dir, exist_ok=True)

        output_base_filename = f"{sanitized_audio_folder_name}-edited"
        output_suffix = ".wav"
        version = 1
        save_full_path_wave = base_save_dir / f"{output_base_filename}{output_suffix}"

        while save_full_path_wave.exists():
            version += 1
            save_full_path_wave = base_save_dir / f"{output_base_filename}_V{version}{output_suffix}"

        save_full_path_spec = save_full_path_wave.with_suffix(".png")
        save_full_path_origwave = base_save_dir / f"{sanitized_audio_folder_name}-orig{output_suffix}"

        print(f"Output base directory: {base_save_dir}")
        print(f"Output WAV will be saved to: {save_full_path_wave}")
        # --- End Path Generation ---

        # --- Reverse/Generation Process ---
        print("Starting reverse/generation process (diffusion)...")
        if args.mode == "ours":
            w0_gen, _ = inversion_reverse_process(
                ldm_stable, xT=wts if not args.test_rand_gen else torch.randn_like(wts),
                tstart=args.tstart, fix_alpha=args.fix_alpha, etas=eta,
                prompts=args.target_prompt, neg_prompts=args.target_neg_prompt,
                cfg_scales=cfg_scale_tar, prog_bar=True,
                # Ensure correct slicing based on skip tensor shape/values
                zs=zs[:int(args.num_diffusion_steps - min(skip.cpu()).item())] if not args.test_rand_gen else torch.randn_like(zs[:int(args.num_diffusion_steps - min(skip.cpu()).item())]),
                cutoff_points=args.cutoff_points, duration=duration, extra_info=extra_info
            )
        else:  # ddim
            if skip.any() != 0: warnings.warn("Partial DDIM inversion.", RuntimeWarning)
            if len(cfg_scale_tar) > 1: raise ValueError("DDIM only supports one cfg_scale_tar")
            if len(args.source_prompt) > 1: raise ValueError("DDIM only supports one source_prompt")
            if len(args.target_prompt) > 1: raise ValueError("DDIM only supports one target_prompt")
            w0_gen = text2image_ldm_stable(
                ldm_stable, args.target_prompt, args.num_diffusion_steps,
                cfg_scale_tar[0], wT, skip=skip[0].item() # Pass scalar skip
            )

    # --- VAE Decode and Prepare for Saving ---
    print("Decoding generated audio (VAE decode)...")
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0_gen)
        if 'stable-audio' not in model_id:
            if x0_dec.dim() < 4: x0_dec = x0_dec[None, :, :, :]
            with torch.no_grad():
                # Assuming decode_to_mel returns waveform for non-stable models
                audio = ldm_stable.decode_to_mel(x0_dec)
                orig_audio = ldm_stable.decode_to_mel(x0)
        else:
            # Stable audio case returns waveform directly from vae_decode
            audio = x0_dec.detach().cpu() # Keep batch dim for consistency if needed later
            orig_audio = x0.detach().cpu()
            # Get specs for stable-audio if needed (e.g., for logging)
            try:
                x0_dec_spec = get_spec(x0_dec.cpu(), ldm_stable.get_fn_STFT())
                x0_orig_spec = get_spec(x0.unsqueeze(0).cpu(), ldm_stable.get_fn_STFT())
                if x0_dec_spec.dim() < 4: x0_dec_spec = x0_dec_spec.unsqueeze(0) # Add batch/channel if needed
                if x0_orig_spec.dim() < 4: x0_orig_spec = x0_orig_spec.unsqueeze(0)
            except Exception as e:
                 print(f"Warning: Failed to generate spectrograms for stable-audio: {e}")
                 x0_dec_spec = None
                 x0_orig_spec = None


    # --- Saving Outputs ---
    print(f"Saving outputs to {base_save_dir}...")
    try:
        # Ensure audio tensors are on CPU before saving
        audio_to_save = audio.squeeze(0).cpu() # Remove potential batch dim
        orig_audio_to_save = orig_audio.squeeze(0).cpu() # Remove potential batch dim

        # Save generated audio
        torchaudio.save(str(save_full_path_wave), audio_to_save, sample_rate=sr)
        print(f"Saved generated audio to {save_full_path_wave}")

        # Save original audio copy
        if not save_full_path_origwave.exists():
            torchaudio.save(str(save_full_path_origwave), orig_audio_to_save, sample_rate=sr)
            print(f"Saved original audio copy to {save_full_path_origwave}")

        # Save spectrogram image (Optional)
        try:
            spec_to_save = None
            if 'stable-audio' in model_id and x0_dec_spec is not None:
                 spec_to_save = x0_dec_spec.squeeze(0).squeeze(0).numpy() # Remove batch/channel
            elif x0_dec is not None: # Attempt basic spec saving for others if x0_dec is latent/spec
                 # This might need adjustment based on what x0_dec actually holds
                 try: spec_to_save = x0_dec.squeeze(0).squeeze(0).cpu().numpy()
                 except: pass # Ignore if squeezing/numpy fails

            if spec_to_save is not None:
                 # Handle potential transpose (same logic as before)
                if spec_to_save.shape[0] > spec_to_save.shape[1]: spec_to_save = spec_to_save.T
                plt.imsave(str(save_full_path_spec), spec_to_save)
                print(f"Saved spectrogram to {save_full_path_spec}")

        except Exception as e:
            print(f"Warning: Could not save spectrogram image '{save_full_path_spec}': {e}")

    except Exception as e:
         print(f"ERROR saving audio files: {e}")


    # --- WANDB Logging (Adjusted) ---
    if not args.wandb_disable and wandb_run:
        print("Logging results to wandb...")
        try:
            final_caption = save_full_path_wave.stem
            logging_dict = {}
            # Prepare numpy arrays for wandb logging
            orig_np = orig_audio.squeeze().cpu().numpy()
            gen_np = audio.squeeze().cpu().numpy()

            # Log audio
            logging_dict['orig_audio'] = wandb.Audio(orig_np, caption=f'{sanitized_audio_folder_name}-orig', sample_rate=sr)
            logging_dict['gen_audio'] = wandb.Audio(gen_np, caption=final_caption, sample_rate=sr)

            # Prepare and log spectrograms if available
            def prep_spec_for_wandb(spec_tensor):
                 if spec_tensor is None or not isinstance(spec_tensor, torch.Tensor): return None
                 spec = spec_tensor.squeeze().cpu().numpy()
                 if spec.ndim > 2: spec = spec[0] # Take first channel if multichannel
                 if spec.shape[0] > spec.shape[1]: spec = spec.T # Transpose if needed
                 return spec

            orig_spec_np = prep_spec_for_wandb(x0_orig_spec if 'stable-audio' in model_id else x0)
            gen_spec_np = prep_spec_for_wandb(x0_dec_spec if 'stable-audio' in model_id else x0_dec)

            if orig_spec_np is not None:
                 logging_dict['orig_spec'] = wandb.Image(orig_spec_np, caption=f'{sanitized_audio_folder_name}-orig_spec')
            if gen_spec_np is not None:
                 logging_dict['gen_spec'] = wandb.Image(gen_spec_np, caption=f'{final_caption}_spec')

            wandb.log(logging_dict)
            print("Logged artifacts to wandb.")
        except Exception as e:
             print(f"Warning: Failed to log to wandb: {e}")

    if wandb_run:
        wandb_run.finish()

    print("Processing finished.")