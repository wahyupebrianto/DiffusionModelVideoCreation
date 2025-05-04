import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from torch import autocast
from tqdm import tqdm

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    v0_np = v0.cpu().numpy() if isinstance(v0, torch.Tensor) else v0
    v1_np = v1.cpu().numpy() if isinstance(v1, torch.Tensor) else v1

    dot = np.dot(v0_np.flatten() / np.linalg.norm(v0_np), v1_np.flatten() / np.linalg.norm(v1_np))
    if np.abs(dot).item() > DOT_THRESHOLD:
        result = (1 - t) * v0_np + t * v1_np
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        result = s0 * v0_np + s1 * v1_np
    return torch.from_numpy(result).to(v0.device).unsqueeze(0)

def generate_video(
    prompt="a surreal landscape with floating islands",
    output_dir="outputs",
    num_frames=100, #120,
    steps_per_transition=20,
    num_inference_steps=50,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=42,
    model_id="CompVis/stable-diffusion-v1-4",
    scheduler_type="LMS",
    device="cuda"
):
    """
    Generates a video by interpolating between random latent vectors.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    # Select scheduler
    if scheduler_type == "LMS":
        scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    # Encode prompt
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    # Generate initial latent
    latent_shape = (1, pipe.unet.config.in_channels, height // 8, width // 8)
    latents_a = torch.randn(latent_shape, device=device)

    frame_idx = 0
    pbar = tqdm(total=num_frames, desc="Generating frames")
    while frame_idx < num_frames:
        latents_b = torch.randn(latent_shape, device=device)
        for step in range(steps_per_transition):
            t = step / steps_per_transition
            latents_interp = slerp(t, latents_a.squeeze(0), latents_b.squeeze(0)).to(device)
            with autocast(device):
                image = pipe(prompt=prompt, latents=latents_interp, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
            image.save(os.path.join(output_dir, f"frame_{frame_idx:05d}.png"))
            frame_idx += 1
            pbar.update(1)
            if frame_idx >= num_frames:
                break
        latents_a = latents_b
    pbar.close()
    print(f"Video frames saved to {output_dir}")

if __name__ == "__main__":
    import fire
    fire.Fire(generate_video)
