import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import soundfile as sf

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    FSInputFile,
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from aiogram.types import ContentType

from huggingface_hub import snapshot_download
from voxcpm import VoxCPM


# =========================================================
# CONFIG
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")

HF_REPO_ID = os.getenv(
    "HF_REPO_ID",
    "pyaesone190300/VoxCPM2"
)

HF_TOKEN = os.getenv("HF_TOKEN") or None

BASE_DIR = Path("/app")

MODEL_DIR = BASE_DIR / "model"

# =========================================================
# ANNA - ORIGINAL V5 VOICE
# =========================================================

ANNA_VOICE = BASE_DIR / "vvipvoice_v5_ref.wav"

ANNA_NAME = "Anna"

PROMPT_TEXT = """ဒေါ်ခင်ခင်ညိုက နူးငယ်ရဲ့ အဒေါ်ပါ။လူကတော့ နည်းနည်းလေး sexyကျတယ်။ နူးငယ်ရဲ့ အဒေါ်ဆိုပေမယ့် အမေ့ညီမ အငယ်ဆုံးဆိုတော့ နူးငယ်နဲ့ကတော့ အသက်သိပ်မကွာပါဘူး။ နူးငယ်အတွက်က အဒေါ်ဆိုလည်းဟုတ်၊ သူငယ်ချင်းဆိုလည်းဟုတ်ဆိုတော့ တူမနှစ်ယောက်က ပြောမနာဆိုမနာပါ။"""


# =========================================================
# CUSTOM VOICE DIRECTORY
# =========================================================

CUSTOM_VOICE_DIR = BASE_DIR / "custom_voices"
CUSTOM_VOICE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# BOT
# =========================================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

model = None

# Prevent multiple TTS generations at the same time
tts_lock = asyncio.Lock()

# Current selected voice per user
# user_id -> "anna" / "custom"
user_voice = {}


# =========================================================
# KEYBOARD
# =========================================================

def main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="👩 Anna"),
                KeyboardButton(text="🎤 Custom Voice"),
            ]
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


# =========================================================
# DOWNLOAD MODEL
# =========================================================

def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Downloading {HF_REPO_ID} ...")

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
    )

    print("✅ Hugging Face model ready")


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():
    global model

    print("🧠 Loading VoxCPM2...")

    model = VoxCPM.from_pretrained(
        str(MODEL_DIR),
        load_denoiser=False,
        local_files_only=True,
        optimize=False,
        device="cpu",
    )

    print("Loaded VoxCPM2Model")
    print("✅ VoxCPM2 loaded")

    try:
        device = next(model.parameters()).device
        print(f"Device: {device}")
    except (StopIteration, AttributeError):
        print("Device: cpu")

    try:
        sample_rate = model.tts_model.sample_rate
        print(f"Sample rate: {sample_rate}")
    except AttributeError:
        print(
            "⚠️ model.tts_model.sample_rate မတွေ့ပါ။ "
            "Fallback sample rate: 16000"
        )


# =========================================================
# SAMPLE RATE
# =========================================================

def get_sample_rate():
    try:
        return model.tts_model.sample_rate
    except AttributeError:
        pass

    try:
        return model.sample_rate
    except AttributeError:
        pass

    return 16000


# =========================================================
# USER CUSTOM VOICE PATH
# =========================================================

def get_custom_voice_path(user_id: int) -> Path:
    return CUSTOM_VOICE_DIR / f"{user_id}.wav"


def has_custom_voice(user_id: int) -> bool:
    return get_custom_voice_path(user_id).exists()


# =========================================================
# GET CURRENT VOICE
# =========================================================

def get_user_voice(user_id: int):
    selected = user_voice.get(user_id, "anna")

    if selected == "custom":
        custom_path = get_custom_voice_path(user_id)

        if custom_path.exists():
            return custom_path, "Custom Voice"

        # Custom Voice မရှိတော့ရင် Anna ပြန်သုံး
        user_voice[user_id] = "anna"

    return ANNA_VOICE, ANNA_NAME


# =========================================================
# TTS GENERATION
# =========================================================

async def generate_voice(
    text: str,
    reference_wav: Path,
    out: Path,
):
    async with tts_lock:

        print("🎙️ Generating voice...")
        print(f"Reference: {reference_wav}")

        wav = await asyncio.to_thread(
            model.generate,
            text=text,
            reference_wav_path=str(reference_wav),
            inference_timesteps=10,
            cfg_value=2.0,
            retry_badcase=False,
            max_len=1000,
        )

        sample_rate = get_sample_rate()

        sf.write(
            str(out),
            wav,
            sample_rate,
        )

        print(
            f"✅ Voice generated "
            f"(sample_rate={sample_rate})"
        )


# =========================================================
# /START
# =========================================================

@dp.message(CommandStart())
async def start_handler(message: Message):

    user_id = message.from_user.id

    # Every /start begins with Anna
    user_voice[user_id] = "anna"

    await message.answer(
        "🎙️ <b>VoxCPM2 Myanmar Voice Bot</b>\n\n"
        "👩 <b>Current Voice: Anna</b>\n\n"
        "Available Voices:\n"
        "👩 Anna — Original V5\n"
        "🎤 Custom Voice — Upload your own voice\n\n"
        "စာပို့လိုက်ရင် လက်ရှိရွေးထားတဲ့ Voice နဲ့ "
        "အသံအဖြစ် ပြောင်းပေးပါမယ်။",
        parse_mode="HTML",
        reply_markup=main_keyboard(),
    )


# =========================================================
# ANNA BUTTON
# =========================================================

@dp.message(F.text == "👩 Anna")
async def anna_handler(message: Message):

    user_id = message.from_user.id

    user_voice[user_id] = "anna"

    await message.answer(
        "👩 <b>Anna</b> ကို ရွေးထားပါတယ်။\n\n"
        "🎙️ Original V5 Voice\n"
        "စာပို့လိုက်ပါ။",
        parse_mode="HTML",
        reply_markup=main_keyboard(),
    )


# =========================================================
# CUSTOM VOICE BUTTON
# =========================================================

@dp.message(F.text == "🎤 Custom Voice")
async def custom_voice_handler(message: Message):

    user_id = message.from_user.id

    if has_custom_voice(user_id):

        user_voice[user_id] = "custom"

        await message.answer(
            "🎤 <b>Custom Voice</b> ကို ရွေးထားပါတယ်။\n\n"
            "လက်ရှိသိမ်းထားတဲ့ Custom Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "Voice အသစ်ပြောင်းချင်ရင် "
            "<b>audio / voice file</b> အသစ်တစ်ခု ပို့ပါ။",
            parse_mode="HTML",
            reply_markup=main_keyboard(),
        )

    else:

        await message.answer(
            "🎤 <b>Custom Voice</b>\n\n"
            "အသုံးပြုချင်တဲ့ Voice / Audio file ကို "
            "ပို့ပေးပါ။\n\n"
            "ဥပမာ -\n"
            "• .wav\n"
            "• .mp3\n"
            "• Telegram Voice\n"
            "• .m4a",
            parse_mode="HTML",
            reply_markup=main_keyboard(),
        )


# =========================================================
# SAVE TELEGRAM VOICE
# =========================================================

async def save_custom_voice(
    message: Message,
    user_id: int,
):

    custom_path = get_custom_voice_path(user_id)

    temp_original = None
    temp_converted = None

    try:

        file_id = None

        # Telegram Voice
        if message.voice:
            file_id = message.voice.file_id

        # Telegram Audio
        elif message.audio:
            file_id = message.audio.file_id

        # Telegram Document
        elif message.document:
            file_id = message.document.file_id

        else:
            return False

        telegram_file = await bot.get_file(file_id)

        suffix = ".dat"

        if message.voice:
            suffix = ".ogg"

        elif message.audio:

            if message.audio.file_name:
                suffix = Path(
                    message.audio.file_name
                ).suffix or ".mp3"

        elif message.document:

            if message.document.file_name:
                suffix = Path(
                    message.document.file_name
                ).suffix or ".wav"

        temp_original = (
            Path(tempfile.gettempdir())
            / f"voice_{user_id}{suffix}"
        )

        await bot.download_file(
            telegram_file.file_path,
            destination=str(temp_original),
        )

        # -------------------------------------------------
        # Try ffmpeg normalization
        # -------------------------------------------------

        temp_converted = (
            Path(tempfile.gettempdir())
            / f"voice_{user_id}_converted.wav"
        )

        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            str(temp_original),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-sample_fmt",
            "s16",
            str(temp_converted),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        return_code = await process.wait()

        if return_code == 0 and temp_converted.exists():

            shutil.copy2(
                temp_converted,
                custom_path,
            )

        else:

            # If ffmpeg isn't available,
            # try direct WAV copy
            shutil.copy2(
                temp_original,
                custom_path,
            )

        # -------------------------------------------------
        # Validate audio
        # -------------------------------------------------

        try:
            data, samplerate = sf.read(
                str(custom_path),
                always_2d=False,
            )

            print(
                f"✅ Custom Voice saved: "
                f"user={user_id}, "
                f"sample_rate={samplerate}"
            )

        except Exception as e:

            print(
                f"❌ Audio validation failed: {e}"
            )

            if custom_path.exists():
                custom_path.unlink()

            return False

        # Automatically select Custom Voice
        user_voice[user_id] = "custom"

        return True

    except Exception as e:

        print(
            f"❌ Custom Voice save error: {e}"
        )

        return False

    finally:

        if temp_original and temp_original.exists():

            try:
                temp_original.unlink()
            except Exception:
                pass

        if temp_converted and temp_converted.exists():

            try:
                temp_converted.unlink()
            except Exception:
                pass


# =========================================================
# VOICE / AUDIO / DOCUMENT HANDLER
# =========================================================

@dp.message(
    F.voice | F.audio | F.document
)
async def voice_upload_handler(message: Message):

    user_id = message.from_user.id

    # Only audio-like files
    if message.document:

        filename = (
            message.document.file_name or ""
        ).lower()

        allowed = (
            ".wav",
            ".mp3",
            ".m4a",
            ".ogg",
            ".flac",
            ".aac",
        )

        if not filename.endswith(allowed):

            await message.answer(
                "❌ ဒီ file က audio file မဟုတ်ပါဘူး။\n\n"
                "🎤 .wav / .mp3 / .m4a / .ogg / .flac "
                "စတဲ့ audio file ပို့ပါ။"
            )

            return

    await message.answer(
        "⏳ <b>Custom Voice ကို သိမ်းနေပါတယ်...</b>",
        parse_mode="HTML",
    )

    success = await save_custom_voice(
        message,
        user_id,
    )

    if success:

        await message.answer(
            "✅ <b>Custom Voice သိမ်းပြီးပါပြီ။</b>\n\n"
            "🎤 Current Voice: <b>Custom Voice</b>\n\n"
            "အခု စာပို့လိုက်ရင် ဒီ Voice နဲ့ "
            "အသံထုတ်ပေးပါမယ်။",
            parse_mode="HTML",
            reply_markup=main_keyboard(),
        )

    else:

        await message.answer(
            "❌ Custom Voice သိမ်းလို့မရပါဘူး။\n\n"
            "WAV / MP3 / M4A / Telegram Voice "
            "တစ်ခုခုနဲ့ ပြန်ပို့ကြည့်ပါ။"
        )


# =========================================================
# TEXT → TTS
# =========================================================

@dp.message(F.text)
async def text_handler(message: Message):

    text = message.text.strip()

    # Ignore buttons
    if text in (
        "👩 Anna",
        "🎤 Custom Voice",
    ):
        return

    if not text:
        return

    if len(text) > 1000:

        await message.answer(
            "❌ စာသားက 1000 characters ထက်မကျော်ရပါ။"
        )

        return

    user_id = message.from_user.id

    reference_wav, voice_name = get_user_voice(
        user_id
    )

    if not reference_wav.exists():

        await message.answer(
            "❌ Voice file မတွေ့ပါဘူး။\n\n"
            "Anna file ကို စစ်ပေးပါ။"
        )

        return

    status = await message.answer(
        f"🎙️ <b>{voice_name}</b>\n"
        f"⏳ အသံထုတ်နေပါတယ်...",
        parse_mode="HTML",
    )

    temp_output = Path(
        tempfile.gettempdir()
    ) / (
        f"tts_{user_id}_"
        f"{message.message_id}.wav"
    )

    try:

        await generate_voice(
            text=text,
            reference_wav=reference_wav,
            out=temp_output,
        )

        await status.edit_text(
            f"🎙️ <b>{voice_name}</b>\n"
            f"✅ အသံထုတ်ပြီးပါပြီ။",
            parse_mode="HTML",
        )

        await message.answer_audio(
            audio=FSInputFile(
                str(temp_output)
            ),
            title=f"{voice_name} Voice",
        )

    except Exception as e:

        print(
            f"❌ TTS Error: {e}"
        )

        await status.edit_text(
            "❌ အသံထုတ်တဲ့အချိန် Error ဖြစ်သွားပါတယ်။\n\n"
            f"<code>{str(e)[:500]}</code>",
            parse_mode="HTML",
        )

    finally:

        if temp_output.exists():

            try:
                temp_output.unlink()
            except Exception:
                pass


# =========================================================
# STARTUP
# =========================================================

async def main():

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN environment variable မတွေ့ပါ။"
        )

    if not ANNA_VOICE.exists():
        raise FileNotFoundError(
            f"Anna voice file မတွေ့ပါ: {ANNA_VOICE}"
        )

    print("🚀 Starting VoxCPM2 Telegram Bot...")

    # Download model
    await asyncio.to_thread(
        download_model
    )

    # Load model
    await asyncio.to_thread(
        load_model
    )

    print("✅ Bot is ready")

    await dp.start_polling(
        bot
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())
