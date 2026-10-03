# Harmful Content Detection in Instagram Reels

This is a research internship project with Prof. Animesh Mukherjee at the Department of Computer Science and Engineering, IIT Kharagpur. It watches an Instagram Reels feed, describes each video, and decides whether the reel is safe, implicitly sexual, or explicitly sexual. The same pipeline was used to review more than 370 videos.

A reel is not judged from the caption alone. Qwen2.5-VL-3B watches the frames and writes a structured description: posture, clothing, setting, physical interaction, on-screen text, and whether anything explicit is visible. GPT-4o-mini then reads that description in three passes. The first pass looks for sexual cues, euphemisms, and how the clip escalates. The second pass argues the other way and lists innocent readings. The third pass gives the category, a confidence score, and the reasons for and against it. The labels are SAFE, IMPLICIT_SEXUAL, and EXPLICIT_SEXUAL.

Moderation also uses annotated cases. A few-shot captioner is shown a safe example and an implicit example, each with a written description, before it describes the reel under review. The new clip is matched against those category examples so the description stays consistent with cases a person has already labelled. That is the retrieval step: similar-category matching, not a fresh guess on every video.

## How a run works

Playwright opens Instagram and keeps the logged-in tab in the background so merely inspecting a reel does not add watch time. The agent reads the current reel URL, steps to the next one, and downloads the video with yt-dlp using the browser cookies. Qwen captions the file. The three-stage classifier returns a verdict. Reels marked implicit or borderline can be played for a few seconds. Everything else is skipped. Results are written locally as JSON and are not part of this repository.

The captioning weights loaded by default are `huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated`, with the processor from `Qwen/Qwen2.5-VL-3B-Instruct`. The language-model stage is `gpt-4o-mini`, called through the OpenAI client. A Gemini key can stand in for that client: if `GEMINI_API_KEY` is set and `OPENAI_API_KEY` is not, the Gemini key is used as the OpenAI key.

## Layout

`main.py` runs the feed loop for a small batch of reels. `reels_agent.py` is the Playwright session: navigation, download, play, and like. `classifier.py` holds Qwen captioning, the few-shot captioner, and the three language-model stages. `experiment.py` is for offline runs on saved videos. `evaluate.ipynb` and `train.ipynb` are the evaluation and training notes. `logger.py` sets up logging.

Annotated example videos used by the few-shot captioner stay on the machine that runs the model. They are named `few_shot_examples/safe.mp4` and `few_shot_examples/implicit.mp4`. They are not uploaded here.

## Setup

Python 3.11 or newer. A CUDA GPU or Apple Silicon is recommended for the 3B vision model. CPU works and is slow.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and put in your own key. Do not commit `.env`.

```
GEMINI_API_KEY=your-gemini-key
QWEN_DEVICE=cuda:0
```

Use `mps` on Apple Silicon or `cpu` if there is no GPU. Set `SKIP_QWEN=1` to skip the vision model while testing the rest of the loop.

The first launch of `python main.py` opens Instagram so you can log in. The session file it writes is local and is ignored by git. Later runs reuse that file.

## What is not in this repository

Live API keys, the Instagram session, downloaded reels, classification dumps, debug screenshots, and the remote-machine setup used for GPU jobs. Those stay off the public project.
