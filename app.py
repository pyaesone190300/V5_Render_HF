import asyncio
import os
import tempfile
import time
from pathlib import Path

import soundfile as sf
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, Message
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")

HF_REPO_ID = os.getenv(
    "HF_REPO_ID",
    "pyaesone190300/VoxCPM2"
)

HF_TOKEN = os.getenv("HF_TOKEN") or None

BASE_DIR = Path("/app")
MODEL_DIR = BASE_DIR / "model"
REFERENCE_WAV = BASE_DIR / "vvipvoice_v5_ref.wav"


# ============================================================
# V5 REFERENCE PROMPT
# ============================================================

PROMPT_TEXT = """ဒေါ်ခင်ခင်ညိုက နူးငယ်ရဲ့ အဒေါ်ပါ။လူကတော့ နည်းနည်းလေး sexyကျတယ်။ နူးငယ်ရဲ့ အဒေါ်ဆိုပေမယ့် အမေ့ညီမ အငယ်ဆုံးဆိုတော့ နူးငယ်နဲ့ကတော့ အသက်သိပ်မကွာပါဘူး။ နူးငယ်အတွက်က အဒေါ်ဆိုလည်းဟုတ်၊ သူငယ်ချင်းဆိုလည်းဟုတ်ဆိုတော့ တူမနှစ်ယောက်က ပြောမနာဆိုမနာပါ။"""


# ============================================================
# TELEGRAM
# ============================================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# User တစ်ယောက် generate လုပ်နေချိန်မှာ
# နောက် user generate မလုပ်နိုင်အောင် lock
tts_lock = asyncio.Lock()

model = None


# ============================================================
# DOWNLOAD MODEL
# ============================================================

def download_model():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"⬇️ Downloading {HF_REPO_ID} ...")

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN
    )

    print("✅ Hugging Face model ready")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    global model

    print("🧠 Loading VoxCPM2...")

    model = VoxCPM.from_pretrained(
        str(MODEL_DIR),
        load_denoiser=False,
        local_files_only=True,
        optimize=False,
        device="cpu"
    )

    print("Loaded VoxCPM2Model")
    print("✅ VoxCPM2 loaded")

    # --------------------------------------------------------
    # IMPORTANT:
    # VoxCPM object မှာ .device attribute မရှိနိုင်ပါ။
    # ဒါကြောင့် model.device မသုံးပါ။
    # --------------------------------------------------------

    try:
        device = next(model.parameters()).device
        print(f"Device: {device}")
    except (StopIteration, AttributeError):
        print("Device: cpu")

    # --------------------------------------------------------
    # Sample rate
    # --------------------------------------------------------

    try:
        sample_rate = model.tts_model.sample_rate
        print(f"Sample rate: {sample_rate}")
    except AttributeError:
        sample_rate = 16000
        print(
            "⚠️ model.tts_model.sample_rate မတွေ့ပါ။ "
            f"Fallback sample rate: {sample_rate}"
        )

    return model


# ============================================================
# GET SAMPLE RATE
# ============================================================

def get_sample_rate():
    """
    VoxCPM version မတူရင် sample_rate location
    ပြောင်းနိုင်တာကြောင့် safe fallback ထားထားပါတယ်။
    """

    try:
        return model.tts_model.sample_rate
    except AttributeError:
        pass

    try:
        return model.sample_rate
    except AttributeError:
        pass

    # Safe fallback
    return 16000


# ============================================================
# GENERATE V5 VOICE
# ============================================================

async def generate_v5(text, out):
    async with tts_lock:

        print("🎙️ Generating V5 voice...")

        wav = await asyncio.to_thread(
            model.generate,
            text=text,

            # Reference voice
            prompt_wav_path=str(REFERENCE_WAV),
            prompt_text=PROMPT_TEXT,

            # Keep reference explicitly
            reference_wav_path=str(REFERENCE_WAV),

            # Inference
            inference_timesteps=10,
            cfg_value=2.0,

            # Error handling
            retry_badcase=False,

            # Max length
            max_len=1000
        )

        sample_rate = get_sample_rate()

        sf.write(
            out,
            wav,
            sample_rate
        )

        print(
            f"✅ Voice generated "
            f"(sample_rate={sample_rate})"
        )


# ============================================================
# /START
# ============================================================

@dp.message(CommandStart())
async def start(message: Message):

    await message.answer(
        "🎙️ <b>V5 Myanmar Voice Bot</b>\n\n"
        "မြန်မာစာပို့ပါ။\n"
        "V5 Voice နဲ့ အသံထုတ်ပေးပါမယ်။\n\n"
        "⏳ CPU server ဖြစ်လို့ အချိန်ကြာနိုင်ပါတယ်။",
        parse_mode="HTML"
    )


# ============================================================
# TEXT HANDLER
# ============================================================

@dp.message(F.text)
async def text_handler(message: Message):

    text = message.text.strip()

    if not text:
        return

    # Character limit
    if len(text) > 1000:
        await message.answer(
            "❌ စာသား 1000 characters ထက် မကျော်ပါနဲ့။"
        )
        return

    # --------------------------------------------------------
    # Generate status
    # --------------------------------------------------------

    status = await message.answer(
        "⏳ <b>V5 အသံထုတ်နေပါတယ်...</b>",
        parse_mode="HTML"
    )

    out = None

    try:

        # ----------------------------------------------------
        # Temporary WAV
        # ----------------------------------------------------

        fd, out = tempfile.mkstemp(
            suffix=".wav",
            prefix="v5_"
        )

        os.close(fd)

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        start_time = time.time()

        await generate_v5(
            text,
            out
        )

        elapsed = time.time() - start_time

        # ----------------------------------------------------
        # Send audio
        # ----------------------------------------------------

        await message.answer_audio(
            FSInputFile(out),
            caption=(
                f"🎙️ V5 Myanmar Voice\n"
                f"⏱️ {elapsed:.1f}s"
            )
        )

        # Delete processing message
        try:
            await status.delete()
        except Exception:
            pass

    except Exception as e:

        print(
            "❌ Generation error:",
            repr(e)
        )

        error_text = str(e)

        # Telegram HTML မပျက်အောင် basic escape
        error_text = (
            error_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        try:
            await status.edit_text(
                "❌ <b>Error ဖြစ်ပါတယ်။</b>\n\n"
                f"<code>{error_text[:1500]}</code>",
                parse_mode="HTML"
            )
        except Exception:
            pass

    finally:

        # ----------------------------------------------------
        # Remove temporary WAV
        # ----------------------------------------------------

        if out:
            try:
                os.remove(out)
            except OSError:
                pass


# ============================================================
# MAIN
# ============================================================

async def main():

    # --------------------------------------------------------
    # BOT TOKEN
    # --------------------------------------------------------

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN မတွေ့ပါ။"
        )

    # --------------------------------------------------------
    # REFERENCE WAV
    # --------------------------------------------------------

    if (
        not REFERENCE_WAV.exists()
        or REFERENCE_WAV.stat().st_size == 0
    ):
        raise FileNotFoundError(
            f"V5 reference file မတွေ့ပါ: "
            f"{REFERENCE_WAV}"
        )

    print("========================================")
    print("🚀 Starting V5 Myanmar Voice Bot")
    print("========================================")

    # --------------------------------------------------------
    # Download model
    # --------------------------------------------------------

    download_model()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    load_model()

    print("========================================")
    print("🤖 Telegram V5 CPU Bot starting...")
    print("========================================")

    # --------------------------------------------------------
    # Telegram polling
    # --------------------------------------------------------

    await dp.start_polling(bot)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
