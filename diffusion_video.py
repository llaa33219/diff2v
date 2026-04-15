import os
import random
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import importlib
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

gr = importlib.import_module("gradio")


MODELS_DIR = Path("./models")
RESOLUTIONS = {
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1024x1024": (1024, 1024),
    "832x1216": (832, 1216),
    "640x960": (640, 960),
}


def is_sd3_model(filename):
    name = filename.lower()
    return (
        "sd3" in name
        or "sd35" in name
        or "sd-3" in name
        or "stable-diffusion-3" in name
    )


def is_sdxl_model(filename):
    name = filename.lower()
    return "xl" in name


def get_available_models():
    if not MODELS_DIR.exists():
        return []
    return sorted([f.name for f in MODELS_DIR.glob("*.safetensors")])


def _enable_vae_memory_opts(pipe):
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()


def load_pipeline(model_filename):
    diffusers = importlib.import_module("diffusers")
    StableDiffusionPipeline = diffusers.StableDiffusionPipeline
    StableDiffusionXLPipeline = diffusers.StableDiffusionXLPipeline
    StableDiffusion3Pipeline = diffusers.StableDiffusion3Pipeline

    use_rocm = torch.cuda.is_available()
    dtype = torch.float16 if use_rocm else torch.float32
    device = "cuda" if use_rocm else "cpu"

    model_path = MODELS_DIR / model_filename

    if is_sd3_model(model_filename):
        pipe = StableDiffusion3Pipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            text_encoder_3=None,
            tokenizer_3=None,
        )
    elif is_sdxl_model(model_filename):
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
        )
    else:
        pipe = StableDiffusionPipeline.from_single_file(
            str(model_path),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipe = pipe.to(device)
    _enable_vae_memory_opts(pipe)

    try:
        torch.set_float32_matmul_precision("high")
        # torch.compile disabled - causes hangs on ROCm UNet compilation
        # if hasattr(pipe, "unet") and pipe.unet is not None:
        #     pipe.unet = torch.compile(pipe.unet, mode="default")
    except Exception as exc:
        print(f"[WARN] torch.compile failed: {exc}")

    return pipe


def _apply_scheduler(
    pipe, scheduler_select, default_scheduler_class_name, default_scheduler_cfg
):
    diffusers = importlib.import_module("diffusers")

    if scheduler_select == "Default":
        if default_scheduler_class_name and default_scheduler_cfg is not None:
            try:
                scheduler_cls = getattr(diffusers, default_scheduler_class_name)
                pipe.scheduler = scheduler_cls.from_config(
                    deepcopy(default_scheduler_cfg)
                )
            except Exception as exc:
                print(f"[WARN] Failed restoring default scheduler: {exc}")
        return

    try:
        current_cfg = pipe.scheduler.config
        if scheduler_select == "DPM++ 2M Karras":
            pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
                current_cfg,
                use_karras_sigmas=True,
                timestep_spacing="trailing",
            )
        elif scheduler_select == "DPM++ SDE":
            pipe.scheduler = diffusers.DPMSolverSDEScheduler.from_config(current_cfg)
        elif scheduler_select == "Euler A":
            pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
                current_cfg
            )
        elif scheduler_select == "LCM":
            pipe.scheduler = diffusers.LCMScheduler.from_config(current_cfg)
    except Exception as exc:
        print(f"[WARN] Failed applying scheduler '{scheduler_select}': {exc}")


PIPE = {"pipe": None}
PIPE_MODEL = {"name": None}
PIPE_DEFAULT_SCHEDULER_CLASS = {"name": None}
PIPE_DEFAULT_SCHEDULER_CFG = {"cfg": None}


def _prepare_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for old_image in output_path.glob("img_*.png"):
        old_image.unlink(missing_ok=True)

    preview_image = output_path / "preview_last.png"
    preview_image.unlink(missing_ok=True)

    old_video = output_path / "output.mp4"
    old_video.unlink(missing_ok=True)
    return output_path


def _batch_decode_latents(latents_list, decode_last_n=None):
    if not latents_list:
        return [], 0

    pipe = PIPE["pipe"]
    if pipe is None or getattr(pipe, "vae", None) is None:
        raise RuntimeError("Pipeline/VAE is not initialized.")

    if decode_last_n and decode_last_n > 0 and len(latents_list) > decode_last_n:
        selected = latents_list[-decode_last_n:]
    else:
        selected = latents_list

    chunk_size = 2
    all_pil = []
    for start in range(0, len(selected), chunk_size):
        chunk = selected[start : start + chunk_size]
        latent_batch = torch.cat(chunk, dim=0)
        with torch.no_grad():
            decoded = pipe.vae.decode(
                latent_batch / pipe.vae.config.scaling_factor
            ).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded_np = decoded.permute(0, 2, 3, 1).cpu().float().numpy()
            all_pil.extend(pipe.image_processor.numpy_to_pil(decoded_np))
        del latent_batch, decoded
        torch.cuda.empty_cache()

    return all_pil, len(selected)


def _apply_crossfade(frames_np, hold_frames, crossfade_frames):
    result_frames = []
    for idx in range(len(frames_np)):
        for _ in range(hold_frames):
            result_frames.append(frames_np[idx])
        if idx < len(frames_np) - 1:
            for t in range(1, crossfade_frames + 1):
                alpha = t / (crossfade_frames + 1)
                blended = (
                    (1 - alpha) * frames_np[idx] + alpha * frames_np[idx + 1]
                ).astype(frames_np.dtype)
                result_frames.append(blended)
    return np.array(result_frames)


def _encode_video_ffmpeg(frames_np, output_path, fps=10, use_hardware=True):
    n_frames, height, width, _ = frames_np.shape

    if use_hardware:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vaapi_device",
            "/dev/dri/renderD129",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "hevc_vaapi",
            "-qp",
            "24",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate(input=frames_np.tobytes())
    if process.returncode != 0 and use_hardware:
        print("[WARN] HW encode failed, retrying software encode:")
        print(stderr.decode("utf-8", errors="ignore"))
        return _encode_video_ffmpeg(frames_np, output_path, fps, use_hardware=False)
    if process.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg encode failed: {err_msg}")

    return str(output_path), ("hardware" if use_hardware else "software")


def generate_diffusion_video(
    model_select,
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    seed,
    random_seed,
    resolution,
    scheduler_select,
    decode_last_n,
    output_dir,
    progress=None,
):
    if progress is None:
        progress = gr.Progress(track_tqdm=False)

    if not model_select:
        return (
            None,
            None,
            "Error: No model selected. Put .safetensors files in /models/",
        )

    models = get_available_models()
    if model_select not in models:
        return None, None, f"Error: Model '{model_select}' not found in /models/"

    if PIPE["pipe"] is None or PIPE_MODEL["name"] != model_select:
        progress(0.02, desc="Loading model...")
        PIPE["pipe"] = load_pipeline(model_select)
        PIPE_MODEL["name"] = model_select
        PIPE_DEFAULT_SCHEDULER_CLASS["name"] = PIPE["pipe"].scheduler.__class__.__name__
        PIPE_DEFAULT_SCHEDULER_CFG["cfg"] = deepcopy(PIPE["pipe"].scheduler.config)

    current_pipe = PIPE["pipe"]
    if current_pipe is None:
        return None, None, "Error: Pipeline failed to initialize."

    _apply_scheduler(
        current_pipe,
        scheduler_select,
        PIPE_DEFAULT_SCHEDULER_CLASS["name"],
        PIPE_DEFAULT_SCHEDULER_CFG["cfg"],
    )

    output_path = _prepare_output_dir(output_dir)
    width, height = RESOLUTIONS[resolution]

    decode_last_n_raw = int(decode_last_n)
    steps = int(steps)
    guidance_scale = float(guidance_scale)

    if random_seed:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = int(seed)

    decode_last_n = 0 if decode_last_n_raw == 0 else max(1, decode_last_n_raw)

    generator = torch.Generator(device=current_pipe.device).manual_seed(seed)
    collected_latents = []

    def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        collected_latents.append(callback_kwargs["latents"].clone())
        progress(
            0.05 + ((step_index + 1) / max(steps, 1)) * 0.75,
            desc=f"Denoising step {step_index + 1}/{steps}",
        )
        return callback_kwargs

    with torch.inference_mode():
        current_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="latent",
        )

    progress(0.82, desc="Batch decoding selected latents...")
    decoded_pil, decoded_count = _batch_decode_latents(collected_latents, decode_last_n)
    if not decoded_pil:
        return None, None, "Error: No decoded frames were produced."

    start_index = (
        len(collected_latents) - decoded_count
        if decode_last_n and decode_last_n > 0
        else 0
    )
    for i, pil_img in enumerate(decoded_pil):
        pil_img.save(output_path / f"img_{start_index + i:04d}.png")

    frames_np = np.stack(
        [np.asarray(img.convert("RGB"), dtype=np.uint8) for img in decoded_pil],
        axis=0,
    )

    base_fps = 10
    num_steps = max(1, len(frames_np))
    hold_frames = max(
        1, round((1.0 if num_steps < 30 else num_steps / 30.0) * base_fps)
    )
    crossfade_frames = max(3, hold_frames // 2)

    progress(0.9, desc="Applying crossfade transitions...")
    expanded_frames_np = _apply_crossfade(frames_np, hold_frames, crossfade_frames)

    progress(0.96, desc="Encoding video with FFmpeg...")
    video_path = output_path / "output.mp4"
    encoded_path, encode_mode = _encode_video_ffmpeg(
        expanded_frames_np,
        video_path,
        fps=base_fps,
        use_hardware=True,
    )

    preview_path = output_path / "preview_last.png"
    decoded_pil[-1].save(preview_path)

    progress(1.0, desc="Done")
    decode_desc = (
        f"all {decoded_count} steps"
        if decode_last_n == 0
        else f"last {decoded_count} steps"
    )
    status = (
        f"Done. Seed: {seed}, decoded {decode_desc}, encoded {len(expanded_frames_np)} frames "
        f"({encode_mode}) to '{encoded_path}'."
    )
    return str(preview_path), str(video_path), status


def build_ui():
    available = get_available_models()

    with gr.Blocks(title="Diffusion to Video (RX 7600 XT)") as demo:
        gr.Markdown("# Diffusion → Step Images → MP4\n*AMD RX 7600 XT (ROCm)*")

        with gr.Row():
            with gr.Column():
                model_select = gr.Dropdown(
                    choices=available,
                    value=available[0] if available else None,
                    label="Model (.safetensors in /models/)",
                    interactive=True,
                )
                if not available:
                    gr.Markdown("⚠️ No .safetensors files found in `/models/`")

                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="A cinematic shot of a futuristic city at sunset",
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    placeholder="blurry, low quality, artifacts",
                )
                steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=30,
                    step=1,
                    label="Steps",
                )
                guidance_scale = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=7.5,
                    step=0.1,
                    label="Guidance Scale",
                )
                seed = gr.Number(label="Seed", value=42, precision=0)
                random_seed = gr.Checkbox(label="Random Seed", value=True)
                resolution = gr.Dropdown(
                    choices=list(RESOLUTIONS.keys()),
                    value="512x512",
                    label="Resolution",
                )
                scheduler_select = gr.Dropdown(
                    choices=[
                        "Default",
                        "DPM++ 2M Karras",
                        "DPM++ SDE",
                        "Euler A",
                        "LCM",
                    ],
                    value="DPM++ 2M Karras",
                    label="Scheduler",
                )
                decode_last_n = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Decode Last N Steps (0 = All)",
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./output",
                )
                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                preview = gr.Image(label="Last Step Preview", type="filepath")
                video = gr.Video(label="Output MP4")
                status = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=generate_diffusion_video,
            inputs=[
                model_select,
                prompt,
                negative_prompt,
                steps,
                guidance_scale,
                seed,
                random_seed,
                resolution,
                scheduler_select,
                decode_last_n,
                output_dir,
            ],
            outputs=[preview, video, status],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
