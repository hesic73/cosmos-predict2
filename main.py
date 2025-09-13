import os
import glob
import math
from pathlib import Path
from multiprocessing import get_context
from typing import List
from loguru import logger
from imaginaire.utils.io import save_image_or_video

# File stem -> prompt mapping
PROMPT_MAPPING = {
    "lab8": "water the flower",
    "lab9": "Pour the candies from the spoon into the plate.",
    "lab10": "put the pot lid onto the pot",
    "lab11": "put the spoon into the pan",
    "lab14": "Sweep the trash directly into the white dustpan with the red brush.",
    "lab16": "pour the tomato in the pan into the white plate",
    "lab18": "put the book inside the bookshelf",
}


def setup_pipeline(model_size: str, device: str):
    """
    Create the Cosmos2 Video2World pipeline on the given device.
    All heavy imports are inside so CUDA_VISIBLE_DEVICES is already set in the worker.
    """
    from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
    from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

    logger.info(
        f"Loading Cosmos2 pipeline on device: {device}, model_size: {model_size}")
    config = get_cosmos_predict2_video2world_pipeline(model_size=model_size)
    # Disable guardrail to match your previous usage
    if hasattr(config, "guardrail_config") and hasattr(config.guardrail_config, "enabled"):
        config.guardrail_config.enabled = False

    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=get_cosmos_predict2_video2world_checkpoint(
            model_size=model_size),
    )
    pipe.to(device)
    return pipe


def generate_videos_for_image(
    pipe,
    image_path: str,
    prompt: str,
    output_dir: str,
    num_videos: int,
    gpu_tag: str,
    base_seed: int,
):
    """
    Run inference multiple times for one image and save videos.
    Heavy I/O import is inside to respect CUDA masking.
    """

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{gpu_tag} Processing {image_path} | prompt: {prompt}")

    for video_idx in range(num_videos):
        # Generate unique seed for each video: base_seed + image_hash + video_idx
        image_hash = hash(image_path) % 10000  # Simple hash to differentiate images
        unique_seed = base_seed + image_hash + video_idx * 1000
        logger.info(f"{gpu_tag} Generating video {video_idx + 1}/{num_videos} with seed {unique_seed}")
        result = pipe(input_path=image_path, prompt=prompt, seed=unique_seed)
        output_file = os.path.join(output_dir, f"video_{video_idx:03d}.mp4")
        save_image_or_video(result, output_file)
        logger.success(f"{gpu_tag} Saved video to: {output_file}")


def _worker(
    visible_gpu: int,
    image_paths: List[str],
    output_dir: str,
    model_size: str,
    num_videos: int,
    base_seed: int,
):
    """
    Single-GPU worker process.
    IMPORTANT: Set CUDA_VISIBLE_DEVICES BEFORE importing torch/imaginaire.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)

    import numpy as np
    import torch

    # Each GPU worker gets a different base seed to ensure diversity
    worker_seed = base_seed + visible_gpu * 100000
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

    gpu_tag = f"[GPU-{visible_gpu}]"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        # Only one visible device in this process => index 0
        torch.cuda.set_device(0)
        logger.info(
            f"{gpu_tag} Using cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        logger.info(f"{gpu_tag} Using CPU")

    pipe = setup_pipeline(model_size=model_size, device=device)

    for image_path in image_paths:
        name = Path(image_path).stem
        prompt = PROMPT_MAPPING.get(name)
        if prompt is None:
            logger.warning(
                f"{gpu_tag} No prompt mapping for '{name}', skipping.")
            continue

        img_output_dir = os.path.join(output_dir, name)
        generate_videos_for_image(
            pipe=pipe,
            image_path=image_path,
            prompt=prompt,
            output_dir=img_output_dir,
            num_videos=num_videos,
            gpu_tag=gpu_tag,
            base_seed=worker_seed,
        )

    logger.success(f"{gpu_tag} Done.")


def main(
    input_dir: str,
    output_dir: str,
    model_size: str = "2B",
    gpus: str = "0",       # e.g., "0,1,2,3"
    num_videos: int = 3,
    seed: int = 42,
):
    """
    Multi-GPU batch inference for Cosmos2 Video2World.
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    logger.info(f"Found {len(png_files)} PNG files in {input_dir}")
    if not png_files:
        logger.warning("No images found. Exit.")
        return

    gpu_ids = [int(x) for x in gpus.split(",") if x.strip()]
    num_workers = max(1, len(gpu_ids))
    logger.info(f"Using {num_workers} worker(s): GPUs {gpu_ids}")

    if num_workers == 1:
        # Single-GPU path; no subprocesses
        _worker(
            visible_gpu=gpu_ids[0],
            image_paths=png_files,
            output_dir=output_dir,
            model_size=model_size,
            num_videos=num_videos,
            base_seed=seed,
        )
        logger.success(f"All done! Results saved in {output_dir}")
        return

    # Evenly split images across workers
    per = math.ceil(len(png_files) / num_workers)
    chunks = [png_files[i * per: min((i + 1) * per, len(png_files))]
              for i in range(num_workers)]

    ctx = get_context("spawn")
    procs = []
    for visible_gpu, paths in zip(gpu_ids, chunks):
        if not paths:
            continue
        p = ctx.Process(
            target=_worker,
            args=(visible_gpu, paths, output_dir,
                  model_size, num_videos, seed),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    logger.success(f"All done! Results saved in {output_dir}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
