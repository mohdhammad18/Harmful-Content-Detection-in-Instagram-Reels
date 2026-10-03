import os
import uuid
import random
import asyncio
import json

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from reels_agent import ReelsAgent, save_session, SESSION_FILE
from logger import logger
from classifier import classify_video, VideoCaptioningModel
from tqdm.auto import tqdm
import time
import datetime

# --- Fast laptop defaults (raise for longer / more coverage runs) ---
# Throughput is still ~1 reel at a time: download + local Qwen caption + 3 OpenAI calls.
WAIT_FOR_SECONDS = 3
MAX_REELS = 20
NUM_EXPERIMENTS = 1
# SAVE_VIDEOS: 0 = delete mp4 after classify (fastest disk); 1 = keep IMPLICIT/BORDERLINE only;
# 2 = keep every download (uses more disk; JSON still always saved).
SAVE_VIDEOS = 2
# Lower = faster Qwen, slightly rougher captions (try 2–4).
CAPTION_FPS = 4

start_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H_%M_%S')

async def experiment():
    logger.info("Initializing video captioning model...")
    video_captioning_model = VideoCaptioningModel()
    video_captioning_model.video_fps = CAPTION_FPS

    if os.path.exists(SESSION_FILE):
        logger.debug(f"Using existing session file: {SESSION_FILE}")
    else:
        logger.info(f"No session file found at {SESSION_FILE}. Starting login flow...")
        await save_session()

    for experiment_idx in tqdm(range(NUM_EXPERIMENTS), desc="Experiments"):
        async with ReelsAgent(headless=True) as agent:
            data = []
            reel = await agent.current_reel()
            logger.info(f"Current reel: {reel}")

            for i in tqdm(range(MAX_REELS), desc="Feed Progress"):
                try:
                    reel = await reel.seek_next()

                    if reel is None:
                        reel = await agent.current_reel()

                    logger.info(f"Reel under consideration: {reel}")

                    path = await reel.download(f"tmp/{uuid.uuid4()}.mp4")
                    logger.debug(f"Downloaded to {path}")

                    # should be in a different tab here, but just in case, bring idle tab to front so we don't accumulate watch time while deciding
                    await agent._idle_page.bring_to_front()

                    result = await classify_video(video_captioning_model=video_captioning_model, video_path=path)

                    verdict = result.get("verdict", "")
                    if hasattr(verdict, "value"):
                        verdict = verdict.value
                    verdict_str = str(verdict) if verdict is not None else ""

                    logger.info(f"Classifier verdict: {verdict_str}")

                    should_engage = verdict_str in ("BORDERLINE", "IMPLICIT_SEXUAL")
                    in_dataset = should_engage

                    if SAVE_VIDEOS == 0 or (SAVE_VIDEOS == 1 and not should_engage):
                        os.remove(path)
                        os.remove(path.replace(".mp4", ".info.json"))
                        path_saved = ""
                    else:
                        path_saved = path

                    if should_engage:
                        await reel.play(WAIT_FOR_SECONDS)
                        logger.debug(f"Finished watching reel for {WAIT_FOR_SECONDS} seconds.")

                        if random.random() < 0.5: 
                            liked = await reel.like()
                            logger.debug("Liked!" if liked else "Like button not found.")
                        
                    else:
                        logger.warning("Skipping this reel.")
                    
                    data.append({
                        "url": reel.url,
                        "result": result,
                        "path": path_saved or path,
                        "dataset": in_dataset
                    })

                except Exception as e:
                    logger.error(f"Error processing reel: {e}")
                    data.append({
                        "url": reel.url if 'reel' in locals() else "N/A",
                        "result": {"verdict": "ERROR", "error": str(e)},
                        "path": "N/A",
                        "dataset": False
                    })
                    continue

                with open(f"experiments/trajectory/agent_evolution_{start_timestamp}_{experiment_idx}.json", "w") as f:
                    json.dump(data, f, indent=2)


if __name__ == "__main__":
    asyncio.run(experiment())
