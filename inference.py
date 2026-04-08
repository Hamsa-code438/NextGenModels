import os
import random
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO

# ── Environment variables (as required by hackathon checklist) ──────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://your-active-url.com")
MODEL_NAME   = os.getenv("MODEL_NAME",   "stabilityai/sdxl-turbo")
HF_TOKEN     = os.getenv("HF_TOKEN")          # optional – no default

# ── Model setup ─────────────────────────────────────────────────────────────
device      = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    use_auth_token=HF_TOKEN,
)
pipe = pipe.to(device)

MAX_SEED       = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory environment state ──────────────────────────────────────────────
env_state = {
    "prompt": "",
    "seed":   0,
    "image":  None,
    "step":   0,
}

# ── Request schemas ──────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    prompt: Optional[str] = "A beautiful landscape"
    seed:   Optional[int] = None

class StepRequest(BaseModel):
    prompt:              Optional[str]   = None
    negative_prompt:     Optional[str]   = ""
    width:               Optional[int]   = 1024
    height:              Optional[int]   = 1024
    guidance_scale:      Optional[float] = 0.0
    num_inference_steps: Optional[int]   = 2

# ── Helper ───────────────────────────────────────────────────────────────────
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment state — required by OpenEnv."""
    seed = request.seed if request.seed is not None else random.randint(0, MAX_SEED)
    env_state.update({
        "prompt": request.prompt,
        "seed":   seed,
        "image":  None,
        "step":   0,
    })
    return {
        "status":      "reset successful",
        "prompt":      env_state["prompt"],
        "seed":        env_state["seed"],
        "observation": None,
        "step":        0,
    }


@app.post("/step")
def step(request: StepRequest = StepRequest()):
    """Run one inference step and return the generated image."""
    prompt = request.prompt or env_state["prompt"]
    seed   = env_state["seed"]

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=request.negative_prompt,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]

    env_state["image"] = image
    env_state["step"] += 1

    return {
        "status":      "step completed",
        "step":        env_state["step"],
        "observation": image_to_base64(image),
        "reward":      1.0,
        "done":        True,
    }


@app.get("/validate")
def validate():
    """Health-check endpoint for openenv validate."""
    return {
        "status":   "ok",
        "model":    MODEL_NAME,
        "device":   device,
        "api_base": API_BASE_URL,
    }


@app.get("/")
def root():
    return {"message": "NextGenModels inference server is running."}


# ── Local dev entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
