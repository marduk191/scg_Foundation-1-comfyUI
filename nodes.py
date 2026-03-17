"""
SCG Foundation-1 Sample Generator – ComfyUI nodes.

Provides four nodes:
  1. SCG Foundation-1 Loader     – downloads / loads the model
  2. SCG Foundation-1 Sampler    – generates audio with full parameter control
  3. SCG Foundation-1 Prompt Builder  – structured prompt construction
  4. SCG Foundation-1 Random Prompt   – weighted random prompt generation
"""

import math
import os
import random as _random
import string
import wave
import struct

import numpy as np
import torch
import folder_paths

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

from .model_manager import load_model, unload_model
from .tags import (
    INSTRUMENT_FAMILIES, ALL_SUBFAMILIES, TIMBRE_TAGS, SPATIAL_TAGS,
    WAVE_TECH_TAGS, STYLE_TAGS, BAND_TAGS,
    FX_REVERB, FX_DELAY, FX_DISTORTION, FX_MODULATION,
    STRUCTURE_TAGS, SPEED_TAGS, RHYTHM_TAGS, CONTOUR_TAGS, DENSITY_TAGS,
    KEYS, SCALES, BARS_OPTIONS, BPM_OPTIONS,
    generate_random_prompt,
)

LOG_PREFIX = "[SCG Foundation-1]"
BEATS_PER_BAR = 4
CATEGORY = "audio/SCG Foundation-1"

SAMPLER_TYPES = [
    "dpmpp-3m-sde",
    "dpmpp-2m-sde",
    "k-heun",
    "k-lms",
    "k-dpmpp-2s-ancestral",
    "k-dpm-2",
    "k-dpm-fast",
]


def _calculate_timing(bars, bpm, sample_rate, downsampling_ratio):
    """
    Calculate the exact sample counts for loop-accurate generation.

    Returns:
        (clip_samples, seconds_total_int, target_samples)
        - clip_samples: exact sample count for the musical duration
        - seconds_total_int: integer seconds for the conditioner
        - target_samples: padded sample count aligned to downsampling_ratio
    """
    clip_seconds = (60.0 / float(bpm)) * float(BEATS_PER_BAR) * float(bars)
    clip_samples = int(round(clip_seconds * sample_rate))
    seconds_total_int = int(math.ceil(clip_samples / sample_rate))
    target_samples = int(seconds_total_int * sample_rate)

    if downsampling_ratio > 0 and (target_samples % downsampling_ratio) != 0:
        target_samples += downsampling_ratio - (target_samples % downsampling_ratio)

    return clip_samples, seconds_total_int, target_samples


def _apply_fade_out(audio, fade_samples=661):
    """Apply a short fade-out (~15 ms at 44100 Hz) to avoid clicks at the loop point."""
    if audio.shape[-1] <= fade_samples:
        return audio
    fade = torch.linspace(1.0, 0.0, fade_samples, device=audio.device, dtype=audio.dtype)
    audio[..., -fade_samples:] *= fade
    return audio


def _save_wav_to_temp(waveform, sample_rate):
    """
    Save a waveform tensor to a temp WAV file for UI preview.
    Returns (filename, subfolder, type) for the ComfyUI audio UI payload.
    Uses the standard-library wave module to avoid extra dependencies.
    """
    temp_dir = folder_paths.get_temp_directory()
    rand_id = "".join(_random.choice(string.ascii_lowercase) for _ in range(10))
    filename = f"foundation1_{rand_id}.wav"
    full_path = os.path.join(temp_dir, filename)

    if waveform.dim() == 3:
        wav = waveform[0]
    elif waveform.dim() == 1:
        wav = waveform.unsqueeze(0)
    else:
        wav = waveform

    wav = wav.clamp(-1.0, 1.0).cpu().float()
    channels, n_samples = wav.shape

    int16_data = (wav.numpy().T * 32767).astype(np.int16)

    with wave.open(full_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

    return filename, "", "temp"


# ---------------------------------------------------------------------------
#  Node 1: Loader
# ---------------------------------------------------------------------------

class SCGFoundation1Loader:
    """Downloads (if needed) and loads the Foundation-1 model into memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("F1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, device, keep_model_loaded):
        try:
            import stable_audio_tools  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "\n\n"
                "═" * 64 + "\n"
                "  stable-audio-tools is NOT installed.\n\n"
                "  It must be installed with --no-deps or it will\n"
                "  break your ComfyUI environment.\n\n"
                "  Run ONE of:\n"
                "    python install.py          (from this node's folder)\n"
                "    pip install stable-audio-tools --no-deps\n"
                "═" * 64
            )
        model, config = load_model(device=device)
        return ((model, config, keep_model_loaded),)


# ---------------------------------------------------------------------------
#  Node 2: Sampler
# ---------------------------------------------------------------------------

class SCGFoundation1Sampler:
    """
    Generates audio samples using Foundation-1.
    Automatically calculates loop-accurate timing from BPM / bars and
    appends key / scale / bars / BPM to the prompt.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("F1_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Synth Lead, Warm, Bright, Melody",
                }),
                "bars": ([4, 8], {"default": 8}),
                "bpm": ([100, 110, 120, 128, 130, 140, 150], {"default": 128}),
                "key": (KEYS, {"default": "C"}),
                "scale": (SCALES, {"default": "minor"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "steps": ("INT", {"default": 75, "min": 1, "max": 500, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 25.0, "step": 0.5}),
                "sampler_type": (SAMPLER_TYPES, {"default": "dpmpp-3m-sde"}),
                "sigma_min": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 2.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "autoplay": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                }),
                "init_audio": ("AUDIO",),
                "init_noise_level": ("FLOAT", {
                    "default": 0.9, "min": 0.01, "max": 5.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def generate(self, model, prompt, bars, bpm, key, scale, seed, steps,
                 cfg_scale, sampler_type, sigma_min, sigma_max, autoplay,
                 negative_prompt="", init_audio=None, init_noise_level=0.9):

        model_obj, model_config, keep_loaded = model

        sample_rate = model_config.get("sample_rate", 44100)
        max_sample_size = model_config.get("sample_size", 882000)
        downsampling_ratio = 2048
        pretransform_cfg = model_config.get("model", {}).get("pretransform", {}).get("config", {})
        if pretransform_cfg:
            downsampling_ratio = pretransform_cfg.get("downsampling_ratio", 2048)

        clip_samples, seconds_total, target_samples = _calculate_timing(
            bars, bpm, sample_rate, downsampling_ratio
        )
        target_samples = min(target_samples, max_sample_size)

        full_prompt = f"{prompt}, {key} {scale}, {bars} bars, {bpm} BPM"
        print(f"{LOG_PREFIX} Prompt: {full_prompt}")
        print(f"{LOG_PREFIX} Timing: {bars} bars @ {bpm} BPM → {clip_samples} samples "
              f"({clip_samples / sample_rate:.2f}s), generation target: {target_samples} samples")

        device = next(model_obj.parameters()).device

        conditioning = [{
            "prompt": full_prompt,
            "seconds_start": 0,
            "seconds_total": float(seconds_total),
        }]

        neg_cond = None
        if negative_prompt and negative_prompt.strip():
            neg_cond = [{
                "prompt": negative_prompt.strip(),
                "seconds_start": 0,
                "seconds_total": float(seconds_total),
            }]

        init_audio_tuple = None
        if init_audio is not None:
            init_sr = init_audio.get("sample_rate", sample_rate)
            init_wav = init_audio["waveform"]
            if init_wav.dim() == 3:
                init_wav = init_wav[0]
            init_audio_tuple = (init_sr, init_wav)

        if model_management is not None:
            model_management.throw_exception_if_processing_interrupted()

        try:
            from stable_audio_tools.inference.generation import generate_diffusion_cond
        except ImportError:
            raise RuntimeError(
                f"{LOG_PREFIX} stable-audio-tools is required. "
                "Install with: pip install stable-audio-tools"
            )

        sampler_kwargs = {
            "sampler_type": sampler_type,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
        }

        if seed == -1:
            seed = np.random.randint(0, 2**31 - 1)

        print(f"{LOG_PREFIX} Generating with {steps} steps, cfg={cfg_scale}, sampler={sampler_type}")

        output = generate_diffusion_cond(
            model_obj,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            negative_conditioning=neg_cond,
            batch_size=1,
            sample_size=target_samples,
            seed=seed,
            device=str(device),
            init_audio=init_audio_tuple,
            init_noise_level=init_noise_level if init_audio_tuple else 1.0,
            **sampler_kwargs,
        )

        if model_management is not None:
            model_management.throw_exception_if_processing_interrupted()

        # Trim to exact loop length and apply fade-out
        if output.shape[-1] > clip_samples:
            output = output[..., :clip_samples]
        output = _apply_fade_out(output, fade_samples=min(661, clip_samples // 4))

        # Normalise to [-1, 1]
        peak = torch.max(torch.abs(output))
        if peak > 0:
            output = output / peak
        output = output.clamp(-1.0, 1.0)

        # Ensure [B, C, S] shape
        if output.dim() == 2:
            output = output.unsqueeze(0)

        audio_dict = {
            "waveform": output.cpu().float(),
            "sample_rate": sample_rate,
        }

        # Temp file for UI preview
        fname, subfolder, ftype = _save_wav_to_temp(output, sample_rate)
        ui_payload = {"audio": [{
            "filename": fname,
            "subfolder": subfolder,
            "type": ftype,
            "autoplay": autoplay,
        }]}

        print(f"{LOG_PREFIX} Generation complete. Output shape: {output.shape}")

        if not keep_loaded:
            unload_model()

        return {"result": (audio_dict,), "ui": ui_payload}


# ---------------------------------------------------------------------------
#  Node 3: Prompt Builder
# ---------------------------------------------------------------------------

class SCGFoundation1PromptBuilder:
    """
    Builds a structured Foundation-1 prompt from individual tag components.
    Each dropdown corresponds to a layer in the model's conditioning hierarchy.
    """

    @classmethod
    def INPUT_TYPES(cls):
        family_opts = ["None"] + INSTRUMENT_FAMILIES
        sub_opts = ["None"] + ALL_SUBFAMILIES
        timbre_opts = ["None"] + sorted(TIMBRE_TAGS)
        spatial_opts = ["None"] + SPATIAL_TAGS
        band_opts = ["None"] + BAND_TAGS
        wave_opts = ["None"] + WAVE_TECH_TAGS
        style_opts = ["None"] + STYLE_TAGS

        return {
            "required": {
                "instrument_family": (family_opts, {"default": "Synth"}),
                "sub_family": (sub_opts, {"default": "Synth Lead"}),
                "timbre_1": (timbre_opts, {"default": "Warm"}),
                "timbre_2": (timbre_opts, {"default": "None"}),
                "timbre_3": (timbre_opts, {"default": "None"}),
                "structure": (STRUCTURE_TAGS, {"default": "Melody"}),
                "speed": (SPEED_TAGS, {"default": "None"}),
                "density": (DENSITY_TAGS, {"default": "None"}),
                "contour": (CONTOUR_TAGS, {"default": "None"}),
                "rhythm": (RHYTHM_TAGS, {"default": "None"}),
            },
            "optional": {
                "spatial": (spatial_opts, {"default": "None"}),
                "band": (band_opts, {"default": "None"}),
                "wave_tech": (wave_opts, {"default": "None"}),
                "style": (style_opts, {"default": "None"}),
                "fx_reverb": (FX_REVERB, {"default": "None"}),
                "fx_delay": (FX_DELAY, {"default": "None"}),
                "fx_distortion": (FX_DISTORTION, {"default": "None"}),
                "fx_modulation": (FX_MODULATION, {"default": "None"}),
                "additional_tags": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Extra comma-separated tags",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, instrument_family, sub_family, timbre_1, timbre_2, timbre_3,
              structure, speed, density, contour, rhythm,
              spatial="None", band="None", wave_tech="None", style="None",
              fx_reverb="None", fx_delay="None", fx_distortion="None",
              fx_modulation="None", additional_tags=""):

        parts = []

        for val in (instrument_family, sub_family):
            if val and val != "None":
                parts.append(val)

        for val in (timbre_1, timbre_2, timbre_3):
            if val and val != "None":
                parts.append(val)

        for val in (spatial, band, wave_tech, style):
            if val and val != "None":
                parts.append(val)

        for val in (fx_reverb, fx_delay, fx_distortion, fx_modulation):
            if val and val != "None":
                parts.append(val)

        for val in (structure, speed, density, contour, rhythm):
            if val and val != "None":
                parts.append(val)

        if additional_tags and additional_tags.strip():
            for tag in additional_tags.split(","):
                tag = tag.strip()
                if tag:
                    parts.append(tag)

        prompt = ", ".join(parts)
        print(f"{LOG_PREFIX} Built prompt: {prompt}")
        return (prompt,)


# ---------------------------------------------------------------------------
#  Node 4: Random Prompt
# ---------------------------------------------------------------------------

class SCGFoundation1RandomPrompt:
    """
    Generates a random structured prompt using Foundation-1's tag vocabulary.
    Supports Simple (coherent, single-family) and Experimental (richer, with
    optional timbre mixing) modes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        families = ["Random"] + INSTRUMENT_FAMILIES
        return {
            "required": {
                "mode": (["Simple", "Experimental"], {"default": "Simple"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "family_override": (families, {"default": "Random"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = CATEGORY

    def generate_prompt(self, mode, seed, family_override="Random"):
        if seed >= 0:
            _random.seed(seed)

        prompt = generate_random_prompt(
            mode=mode.lower(),
            family_override=family_override if family_override != "Random" else None,
        )

        print(f"{LOG_PREFIX} Random prompt ({mode}): {prompt}")
        return (prompt,)


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "SCGFoundation1Loader": SCGFoundation1Loader,
    "SCGFoundation1Sampler": SCGFoundation1Sampler,
    "SCGFoundation1PromptBuilder": SCGFoundation1PromptBuilder,
    "SCGFoundation1RandomPrompt": SCGFoundation1RandomPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCGFoundation1Loader": "SCG Foundation-1 Loader",
    "SCGFoundation1Sampler": "SCG Foundation-1 Sample Generator",
    "SCGFoundation1PromptBuilder": "SCG Foundation-1 Prompt Builder",
    "SCGFoundation1RandomPrompt": "SCG Foundation-1 Random Prompt",
}
