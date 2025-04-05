
[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.23.5-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.23.5/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7.1+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.7.1)
[![Notebook](https://img.shields.io/badge/notebook-7.0.6+-green?logo=jupyter&logoColor=white)](https://pypi.org/project/notebook/7.0.6)
[![torch](https://img.shields.io/badge/torch-2.0.0+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-2.0.1+-green?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![diffusers](https://img.shields.io/badge/diffusers-0.22.0+-green)](https://github.com/huggingface/diffusers/) <!-- Check specific needed version -->
[![transformers](https://img.shields.io/badge/transformers-1.35.0+-green)](https://github.com/huggingface/transformers/) <!-- Check specific needed version -->
[![PyQt6](https://img.shields.io/badge/PyQt6-Required%20for%20UI-orange)](https://pypi.org/project/PyQt6/)
[![Colorama](https://img.shields.io/badge/Colorama-Required%20for%20UI-yellow)](https://pypi.org/project/Colorama/)
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]
[![CC BY 4.0][cc-by-shield]][cc-by]

<!-- omit in toc -->
# Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion [ICML 2025] - Germanized UI Version

### [Project page](https://HilaManor.github.io/AudioEditing) | [Arxiv](https://arxiv.org/abs/2402.10009) | [Text-Based Space](https://huggingface.co/spaces/hilamanor/audioEditing)

This repository contains the official code release for ***Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion (Germanized added audio+text-audio conversion without stable audio)***, along with a graphical user interface provided by Germanized.

### known bug/tweak (ITS NOT FROZEN)
the progress in the terminal or gui wont show untill it is done or has encountered an error I have spent nights trying to figure this out to no avail but the app still works 

<!-- omit in toc -->
## Table of Contents

- [Change Log](#change-log)
- [Requirements](#requirements)
- [Germanized UI Version](#germanized-ui-version)
  - [How it Works](#how-it-works)
  - [Troubleshooting](#troubleshooting)
- [Original Command-Line Usage](#original-command-line-usage)
  - [Text-Based Editing](#text-based-editing)
  - [Unsupervised Editing](#unsupervised-editing)
  - [SDEdit](#sdedit)
- [Evaluation](#evaluation)
- [MedleyMDPrompts](#medleymdprompts)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Change Log

**2024-10-12**: Added support for text-based editing using [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)!

- Wasn't trained only on music, so results might vary. You can change the loaded checkpoint to finetuned models on music in `models.py:StableAudWrapper`.
- You need to [accept the model's license](https://huggingface.co/stabilityai/stable-audio-open-1.0), then insert your Hugging Face token in `main_run.py:HF_TOKEN`.
- Needs Diffusers >= 0.30.
- Recommended to use with `src_cfg` of 1.

**2024-09-09**: Added a wrapper for a face-images unconditional LDM model (trained on CelebAHQ), relevant for unsupervised editing. Additionally, moved to PyTorch >= 2.2, Diffusers >= 0.26 to accommodate security concerns. The version tested in the paper is still reachable in the `paper_code` branch.

*(Previous Change Logs preserved)*

## Requirements

Install the core requirements:
```bash
python -m pip install -r requirements.txt
```

**For the UI Version:**

You also need PyQt6 and colorama:
```bash
python -m pip install PyQt6 colorama
```

**IMPORTANT:** This project **requires a CUDA-enabled GPU** and a correctly configured environment (PyTorch with CUDA support, matching NVIDIA drivers). The UI includes checks for this and will disable processing if CUDA is unavailable.

## Germanized UI Version

A graphical user interface (`UI.py` or `gui_launcher.py`), created by Germanized, is provided for easier interaction with the text-based editing functionality (`main_run.py`).

![UI Screenshot Placeholder](link_to_screenshot.png) <!-- Optional: Add a screenshot of the UI here -->

### How it Works

1.  **Launch:** Run the UI script from your terminal within the `code` directory (or wherever `UI.py` and `main_run.py` reside):
    ```bash
    cd path/to/your/code/directory
    python UI.py
    ```
2.  **Splash Screen:** A brief loading splash screen is displayed.
3.  **Dependencies Check:** The UI automatically checks:
    *   If PyTorch and Torchaudio are installed and importable.
    *   If a CUDA-enabled GPU is detected by PyTorch (`torch.cuda.is_available()`).
    *   Processing is disabled, and warnings are shown if prerequisites are not met.
4.  **Upload Audio:** Click "Upload Audio" to select a `.wav` or `.mp3` file using the file dialog. The selected filename and audio length (if readable) will be displayed.
5.  **Select Model:** Choose the desired audio diffusion model from the dropdown list. Note that selecting "Stable Audio Open 1.0" will prompt a reminder to set the `HF_TOKEN` variable in the `main_run.py` script.
6.  **Enter Prompt:** Type your text-based editing instructions into the large text box (e.g., "Remove background noise", "Add reverb", "Change speaker gender").
7.  **Process:** Click "Process Audio".
    *   The UI constructs the necessary command-line arguments based on your inputs and the detected CUDA device number.
    *   It launches `main_run.py` as a background subprocess using the same Python interpreter.
    *   **Live Output (in Terminal):** You will see the real-time STDOUT and STDERR output from `main_run.py` printed **in the terminal window where you launched the UI**. This output is color-coded (requires `colorama`) to distinguish stdout (usually green) from stderr (usually red). This is crucial for seeing model loading progress, diffusion progress bars (printed to stderr), and detailed error messages.
    *   **GUI Status Label:** The status label at the bottom of the UI provides overall status updates ("Processing starting...", "Status: Processing complete!", or error messages). On error, it includes a summary of the collected standard output and standard error from the subprocess for easier diagnosis and copying.
    *   **GUI Progress Bar:** The progress bar attempts to track the diffusion steps by parsing percentage values (`XX%|`) printed by `main_run.py` to the standard error stream (stderr). Its accuracy depends on the `tqdm` output format and system buffering, but it provides a visual indication during the diffusion phase.
8.  **Output:**
    *   Upon successful completion, the UI status indicates completion, and the final progress bar reaches 100%.
    *   The edited audio file (e.g., `your_audio_name-edited.wav`, `your_audio_name-edited_V2.wav`, etc.) and potentially other files (like spectrogram `.png` and `orig.wav` copy) will be saved according to the logic within the modified `main_run.py`. The default structure is `output/<model_name>/<sanitized_audio_name>/`.
    *   The UI then attempts to open the main `output/` directory using the system's default file explorer (`os.startfile` on Windows, `open` on macOS, `xdg-open` on Linux).

### Troubleshooting

*   **"main_run.py not found" Error:** Make sure you launch `UI.py` from the same directory where `main_run.py` is located using `cd` first in your terminal.
*   **Processing Hangs/Stuck:** If the UI says "Processing..." and the terminal output stops (especially after "Loading model/audio"), this usually indicates `main_run.py` has frozen.
    *   **Monitor VRAM:** Use `nvidia-smi -l 1` in another terminal. Check if GPU memory usage hits the maximum. This is the most common cause (Out of Memory). Try a smaller model or shorter audio clip.
    *   **Run Manually:** Copy the "Equivalent manual command" printed in the terminal (add quotes around file paths and prompts!) and run it directly. Add `print()` statements inside `main_run.py` to pinpoint the hang location.
    *   **`CUDA_LAUNCH_BLOCKING=1`:** Set this environment variable before running manually (`set CUDA_LAUNCH_BLOCKING=1` on Windows cmd, `$env:CUDA_LAUNCH_BLOCKING=1` on PowerShell, `export CUDA_LAUNCH_BLOCKING=1` on Linux/macOS). This makes CUDA errors synchronous and might reveal a more specific error traceback in the terminal at the exact point of failure.
*   **`SyntaxError: expected 'except' or 'finally' block`:** This indicates an indentation error within a `try...except` block in the `UI.py` script itself. Ensure you have the latest version of the script and that no unintended indentation changes occurred (e.g., from mixing tabs and spaces). Using an IDE (like VS Code) can help find these.
*   **`CUDA error: unknown error`:** Follow the troubleshooting steps above, especially setting `CUDA_LAUNCH_BLOCKING=1` and checking VRAM and driver/CUDA/PyTorch compatibility.

## Original Command-Line Usage

The core functionalities can also be used directly via the command line as described in the original paper's repository.

*(Keep the original sections below for Text-Based Editing, Unsupervised Editing, SDEdit, Evaluation, etc., unchanged, just ensure section headers match the original)*

### Text-Based Editing
```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_run.py --cfg_tar <target_cfg_strength> --cfg_src <source_cfg_strength> --init_aud <input_audio_path> --target_prompt <description of the wanted edited signal> --tstart <edit from timestep> --model_id <model_name> --results_path <path to dump results> --device_num <cuda_device_index>
```
- You can supply a source prompt that describes the original audio by using `--source_prompt`.
- `tstart` is set to `100` by default. Edit strength decreases as `tstart` increases.
- Use `python main_run.py --help` for all options.
- Use `--mode ddim` for DDIM inversion (requires `tstart` == `num_diffusion_steps`).

### Unsupervised Editing
First extract the PCs for your wanted timesteps:
```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_pc_extract_inv.py --init_aud <input_audio_path> --model_id <model_name> --results_path <path to dump results> --drift_start <start timestep> --drift_end <end timestep> --n_evs <amount of evs> --device_num <cuda_device_index>
```
Then apply the PCs:
```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_pc_apply_drift.py --extraction_path <path to .pt file> --drift_start <apply start> --drift_end <apply end> --amount <edit strength> --evs <ev nums> --device_num <cuda_device_index>
```
- Use `python main_pc_extract_inv.py --help` and `python main_pc_apply_drift.py --help` for options.

### SDEdit
```bash
CUDA_VISIBLE_DEVICES=<gpu_num> python main_run_sdedit.py --cfg_tar <target_cfg_strength> --init_aud <input_audio_path> --target_prompt <description> --tstart <edit from timestep> --model_id <model_name> --results_path <path> --device_num <cuda_device_index>
```
- Use `python main_run_sdedit.py --help` for options.

## Evaluation
... (original content) ...

## MedleyMDPrompts
... (original content) ...

## Citation
... (original content) ...

## Acknowledgements

*(Original acknowledgements from the base repository)*

GUI development and integration by Germanized.

AudioLDM2 is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa]. Therefore, using the weights of AudioLDM2 (the default) and code originating in the `code/audioldm` folder is under the same license.
The weights of StableAudioOpen are licensed under Stability AI's Community License.
The rest of the code (inversion, PCs computation) is licensed under an MIT license.
The Germanized UI (`UI.py`) is also licensed under the MIT License.

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

The evaluation code adapts code from differently licensed repos:

- FAD is from [microsoft/fadtk](https://github.com/microsoft/fadtk), under MIT License.
- LPAPS is adapted from [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), under BSD-2-Clause License.
- CLAP's weights are under CC0-1.0 License, from [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- CLAP's processing code is adapted from [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft), under MIT License.

Our *MedleyMDPrompts* dataset is licensed under CC-BY-4.0 License.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

