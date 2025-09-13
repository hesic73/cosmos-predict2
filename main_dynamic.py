import os
import glob
import math
import time
from pathlib import Path
from multiprocessing import get_context, Queue, Event
from queue import Empty
from typing import List, Tuple
from loguru import logger
from imaginaire.utils.io import save_image_or_video
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

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
    logger.info(f"Loading Cosmos2 pipeline on device: {device}, model_size: {model_size}")
    config = get_cosmos_predict2_video2world_pipeline(model_size=model_size)
    # Disable guardrail to match your previous usage
    if hasattr(config, "guardrail_config") and hasattr(config.guardrail_config, "enabled"):
        config.guardrail_config.enabled = False

    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=get_cosmos_predict2_video2world_checkpoint(model_size=model_size),
    )
    pipe.to(device)
    return pipe


def gpu_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    model_size: str,
    base_seed: int,
):
    """
    GPU worker that continuously processes tasks from the queue.
    Each task is a single video generation job.
    """
    # Set CUDA device BEFORE importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import numpy as np
    import torch
    
    # Set unique seed for this GPU worker
    worker_seed = base_seed + gpu_id * 100000
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    
    gpu_tag = f"[GPU-{gpu_id}]"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Only one visible device in this process
        logger.info(f"{gpu_tag} Using cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        logger.info(f"{gpu_tag} Using CPU")
    
    # Initialize pipeline once per worker
    pipe = setup_pipeline(model_size=model_size, device=device)
    logger.success(f"{gpu_tag} Pipeline loaded and ready")
    
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            # Get task from queue with timeout
            task = task_queue.get(timeout=1.0)
            if task is None:  # Poison pill
                break
                
            image_path, prompt, output_file, seed = task
            
            logger.info(f"{gpu_tag} Processing {Path(image_path).stem} -> {Path(output_file).name} (seed: {seed})")
            
            # Generate video
            start_time = time.time()
            result = pipe(input_path=image_path, prompt=prompt, seed=seed)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            save_image_or_video(result, output_file)
            
            elapsed = time.time() - start_time
            processed_count += 1
            
            logger.success(f"{gpu_tag} Completed {Path(output_file).name} in {elapsed:.1f}s (total: {processed_count})")
            
            # Report completion
            result_queue.put((gpu_id, output_file, True, None))
            
        except Empty:
            continue  # Timeout, check stop_event and continue
        except Exception as e:
            logger.error(f"{gpu_tag} Error processing task: {e}")
            if 'task' in locals():
                result_queue.put((gpu_id, task[2] if len(task) > 2 else "unknown", False, str(e)))
    
    logger.info(f"{gpu_tag} Worker finished. Processed {processed_count} videos.")


def create_all_tasks(
    image_paths: List[str], 
    output_dir: str, 
    num_videos: int, 
    base_seed: int
) -> List[Tuple[str, str, str, int]]:
    """
    Create all video generation tasks.
    Returns list of (image_path, prompt, output_file, seed) tuples.
    """
    tasks = []
    
    for image_path in image_paths:
        name = Path(image_path).stem
        prompt = PROMPT_MAPPING.get(name)
        
        if prompt is None:
            logger.warning(f"No prompt mapping for '{name}', skipping.")
            continue
        
        img_output_dir = os.path.join(output_dir, name)
        
        for video_idx in range(num_videos):
            # Generate unique seed for each video
            image_hash = hash(image_path) % 10000
            unique_seed = base_seed + image_hash + video_idx * 1000
            output_file = os.path.join(img_output_dir, f"video_{video_idx:03d}.mp4")
            
            tasks.append((image_path, prompt, output_file, unique_seed))
    
    return tasks


def main(
    input_dir: str,
    output_dir: str,
    model_size: str = "2B",
    gpus: str = "0",       # e.g., "0,1,2,3"
    num_videos: int = 3,
    seed: int = 42,
):
    """
    Dynamic multi-GPU batch inference for Cosmos2 Video2World.
    Uses a task queue to distribute individual video generation jobs across GPUs.
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
    num_workers = len(gpu_ids)
    logger.info(f"Using {num_workers} GPU worker(s): {gpu_ids}")
    
    # Create all tasks
    all_tasks = create_all_tasks(png_files, output_dir, num_videos, seed)
    total_tasks = len(all_tasks)
    logger.info(f"Created {total_tasks} video generation tasks")
    
    if total_tasks == 0:
        logger.warning("No tasks to process. Exit.")
        return
    
    # Create queues and synchronization objects
    ctx = get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    stop_event = ctx.Event()
    
    # Fill task queue
    for task in all_tasks:
        task_queue.put(task)
    
    # Start GPU workers
    workers = []
    for gpu_id in gpu_ids:
        worker = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, stop_event, model_size, seed)
        )
        worker.start()
        workers.append(worker)
    
    logger.info(f"Started {len(workers)} GPU workers")
    
    # Monitor progress
    completed_tasks = 0
    failed_tasks = 0
    start_time = time.time()
    
    try:
        while completed_tasks + failed_tasks < total_tasks:
            try:
                gpu_id, output_file, success, error = result_queue.get(timeout=30.0)
                
                if success:
                    completed_tasks += 1
                    logger.info(f"Progress: {completed_tasks}/{total_tasks} completed")
                else:
                    failed_tasks += 1
                    logger.error(f"Task failed on GPU {gpu_id}: {output_file} - {error}")
                    
            except Empty:
                logger.warning("No results received in 30s, checking worker status...")
                # Check if workers are still alive
                alive_workers = [w for w in workers if w.is_alive()]
                if not alive_workers:
                    logger.error("All workers died!")
                    break
                    
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        # Signal workers to stop
        stop_event.set()
        
        # Add poison pills to ensure workers exit
        for _ in workers:
            task_queue.put(None)
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker.pid}")
                worker.terminate()
                worker.join()
    
    elapsed = time.time() - start_time
    
    logger.success(f"Completed {completed_tasks}/{total_tasks} tasks in {elapsed:.1f}s")
    if failed_tasks > 0:
        logger.warning(f"{failed_tasks} tasks failed")
    
    logger.success(f"Results saved in {output_dir}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
