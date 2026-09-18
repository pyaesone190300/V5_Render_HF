# V5 Myanmar Voice Telegram Bot — Render + Hugging Face

GitHub stores the bot code. Hugging Face stores the large VoxCPM2 model.

Default HF repo: `pyaesone190300/VoxCPM2`

Render downloads the model at startup with `huggingface_hub`.

V5 settings are unchanged:
- inference_timesteps=10
- cfg_value=2.0
- retry_badcase=False
- max_len=1000

Render Environment Variables:
- `BOT_TOKEN` = Telegram BotFather token
- `HF_REPO_ID` = pyaesone190300/VoxCPM2
- `HF_TOKEN` = only if the HF repo is private

Keep `model.safetensors` OUT of GitHub.

The worker is CPU-only (`device="cpu"`, `optimize=False`) and serializes TTS generation with a lock. This is slower than a T4 GPU but keeps resource use predictable.

For production, use persistent storage if available so the several-GB HF model does not need to be downloaded after every restart.
