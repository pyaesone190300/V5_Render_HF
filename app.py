import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

import soundfile as sf

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

from huggingface_hub import snapshot_download
from voxcpm import VoxCPM


# =========================================================
# CONFIG
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

HF_REPO_ID = os.getenv(
    "HF_REPO_ID",
    "pyaesone190300/VoxCPM2"
)

HF_TOKEN = os.getenv("HF_TOKEN") or None

BASE_DIR = Path("/app")

MODEL_DIR = BASE_DIR / "model"

ANNA_VOICE = BASE_DIR / "vvipvoice_v5_ref.wav"

CUSTOM_VOICE_DIR = BASE_DIR / "custom_voices"

ACCESS_FILE = BASE_DIR / "access.json"

CUSTOM_VOICE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# ANNA
# =========================================================

ANNA_NAME = "Anna"

PROMPT_TEXT = """ဒေါ်ခင်ခင်ညိုက နူးငယ်ရဲ့ အဒေါ်ပါ။လူကတော့ နည်းနည်းလေး sexyကျတယ်။ နူးငယ်ရဲ့ အဒေါ်ဆိုပေမယ့် အမေ့ညီမ အငယ်ဆုံးဆိုတော့ နူးငယ်နဲ့ကတော့ အသက်သိပ်မကွာပါဘူး။ နူးငယ်အတွက်က အဒေါ်ဆိုလည်းဟုတ်၊ သူငယ်ချင်းဆိုလည်းဟုတ်ဆိုတော့ တူမနှစ်ယောက်က ပြောမနာဆိုမနာပါ။"""


# =========================================================
# BOT
# =========================================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

model = None

tts_lock = asyncio.Lock()

access_lock = asyncio.Lock()


# =========================================================
# USER VOICE MEMORY
# =========================================================

# user_id -> "anna" / "custom"

user_voice = {}


# =========================================================
# ACCESS DATABASE
# =========================================================

def default_access():
    return {
        "approved": [],
        "pending": [],
        "rejected": [],
    }


def load_access():

    if not ACCESS_FILE.exists():
        return default_access()

    try:

        with open(
            ACCESS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        data.setdefault("approved", [])
        data.setdefault("pending", [])
        data.setdefault("rejected", [])

        return data

    except Exception as e:

        print(
            f"⚠️ access.json load error: {e}"
        )

        return default_access()


def save_access(data):

    temp_file = ACCESS_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_file.replace(
        ACCESS_FILE
    )


def is_owner(user_id: int):

    return (
        OWNER_ID != 0
        and user_id == OWNER_ID
    )


def is_approved(user_id: int):

    if is_owner(user_id):
        return True

    data = load_access()

    return str(user_id) in data["approved"]


def is_pending(user_id: int):

    data = load_access()

    return str(user_id) in data["pending"]


# =========================================================
# ACCESS REQUEST
# =========================================================

async def request_access(
    message: Message
):

    user = message.from_user

    user_id = user.id

    if is_owner(user_id):

        return True

    if is_approved(user_id):

        return True

    async with access_lock:

        data = load_access()

        uid = str(user_id)

        if uid not in data["pending"]:

            data["pending"].append(uid)

        # If previously rejected, remove rejected state
        if uid in data["rejected"]:

            data["rejected"].remove(uid)

        save_access(data)

    # -----------------------------------------------------
    # OWNER REQUEST
    # -----------------------------------------------------

    username = (
        f"@{user.username}"
        if user.username
        else "No username"
    )

    first_name = (
        user.first_name
        or "Unknown"
    )

    owner_text = (
        "🔐 <b>New Access Request</b>\n\n"
        f"👤 Name: <b>{first_name}</b>\n"
        f"🔹 Username: {username}\n"
        f"🆔 User ID: <code>{user_id}</code>\n\n"
        "ဒီ User ကို Bot အသုံးပြုခွင့်ပေးမလား?"
    )

    owner_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Approve",
                    callback_data=f"access:approve:{user_id}"
                ),
                InlineKeyboardButton(
                    text="❌ Reject",
                    callback_data=f"access:reject:{user_id}"
                ),
            ]
        ]
    )

    try:

        await bot.send_message(
            OWNER_ID,
            owner_text,
            parse_mode="HTML",
            reply_markup=owner_keyboard,
        )

    except Exception as e:

        print(
            f"❌ Cannot contact owner: {e}"
        )

    return False


# =========================================================
# MAIN MENU
# =========================================================

def main_menu(
    current_voice: str = "Anna"
):

    voice_text = (
        "👩 Anna"
        if current_voice == "Anna"
        else "🎤 Custom Voice"
    )

    return InlineKeyboardMarkup(
        inline_keyboard=[

            [
                InlineKeyboardButton(
                    text=(
                        f"👩 Anna"
                        f"{'  ✓' if current_voice == 'Anna' else ''}"
                    ),
                    callback_data="voice:anna"
                ),

                InlineKeyboardButton(
                    text=(
                        f"🎤 Custom Voice"
                        f"{'  ✓' if current_voice == 'Custom Voice' else ''}"
                    ),
                    callback_data="voice:custom"
                ),
            ],

            [
                InlineKeyboardButton(
                    text="ℹ️ How to Use",
                    callback_data="menu:help"
                )
            ]

        ]
    )


# =========================================================
# BACK BUTTON
# =========================================================

def back_menu():

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⬅️ Back",
                    callback_data="menu:main"
                )
            ]
        ]
    )


# =========================================================
# CUSTOM VOICE INSTRUCTION
# =========================================================

def custom_voice_menu():

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⬅️ Back",
                    callback_data="menu:main"
                )
            ]
        ]
    )


# =========================================================
# START
# =========================================================

@dp.message(CommandStart())
async def start_handler(
    message: Message
):

    user_id = message.from_user.id

    # -----------------------------------------------------
    # ACCESS CHECK
    # -----------------------------------------------------

    if not is_approved(user_id):

        allowed = await request_access(
            message
        )

        if not allowed:

            if is_pending(user_id):

                await message.answer(
                    "🔐 <b>Access Pending</b>\n\n"
                    "ဒီ Bot ကို အသုံးပြုရန် "
                    "Owner ရဲ့ ခွင့်ပြုချက်လိုအပ်ပါတယ်။\n\n"
                    "⏳ Owner ဆီကို Access Request ပို့ပြီးပါပြီ။\n"
                    "ခွင့်ပြုပြီးတာနဲ့ Bot ကို အသုံးပြုနိုင်ပါမယ်။",
                    parse_mode="HTML",
                )

            return

    # -----------------------------------------------------
    # APPROVED
    # -----------------------------------------------------

    user_voice[user_id] = "anna"

    await message.answer(
        "🎙️ <b>VoxCPM2 Myanmar Voice Bot</b>\n\n"
        "👋 မင်္ဂလာပါ။\n\n"
        "ဒီ Bot မှာ မြန်မာစာကို Voice အဖြစ် "
        "ပြောင်းနိုင်ပါတယ်။\n\n"
        "🎙️ <b>Current Voice: Anna</b>\n\n"
        "အောက်က Menu ကနေ Voice ရွေးပါ။",
        parse_mode="HTML",
        reply_markup=main_menu("Anna"),
    )


# =========================================================
# OWNER COMMAND
# =========================================================

@dp.message(Command("pending"))
async def pending_handler(
    message: Message
):

    user_id = message.from_user.id

    if not is_owner(user_id):

        return

    data = load_access()

    pending = data["pending"]

    if not pending:

        await message.answer(
            "📭 <b>Pending Request မရှိပါ။</b>",
            parse_mode="HTML"
        )

        return

    text = (
        "🔐 <b>Pending Access Requests</b>\n\n"
    )

    for uid in pending:

        text += (
            f"🆔 <code>{uid}</code>\n"
        )

    await message.answer(
        text,
        parse_mode="HTML"
    )


# =========================================================
# APPROVE / REJECT
# =========================================================

@dp.callback_query(
    F.data.startswith("access:")
)
async def access_callback(
    callback: CallbackQuery
):

    # Only owner can approve/reject
    if not is_owner(
        callback.from_user.id
    ):

        await callback.answer(
            "❌ Owner only",
            show_alert=True
        )

        return

    parts = callback.data.split(":")

    if len(parts) != 3:

        await callback.answer(
            "Invalid request",
            show_alert=True
        )

        return

    action = parts[1]

    target_id = int(parts[2])

    async with access_lock:

        data = load_access()

        target = str(target_id)

        # -------------------------------------------------
        # APPROVE
        # -------------------------------------------------

        if action == "approve":

            if target not in data["approved"]:

                data["approved"].append(
                    target
                )

            if target in data["pending"]:

                data["pending"].remove(
                    target
                )

            if target in data["rejected"]:

                data["rejected"].remove(
                    target
                )

            save_access(data)

            await callback.message.edit_text(
                callback.message.text
                + "\n\n"
                + "✅ <b>APPROVED</b>",
                parse_mode="HTML",
            )

            try:

                await bot.send_message(
                    target_id,
                    "🎉 <b>Access Approved!</b>\n\n"
                    "Owner က သင့်ကို Bot အသုံးပြုခွင့် "
                    "ပေးလိုက်ပါပြီ။\n\n"
                    "အခု Bot ကို စတင်အသုံးပြုနိုင်ပါပြီ။",
                    parse_mode="HTML",
                    reply_markup=main_menu("Anna"),
                )

            except Exception as e:

                print(
                    f"⚠️ Cannot notify user: {e}"
                )

            await callback.answer(
                "✅ User approved"
            )

            return

        # -------------------------------------------------
        # REJECT
        # -------------------------------------------------

        if action == "reject":

            if target in data["pending"]:

                data["pending"].remove(
                    target
                )

            if target not in data["rejected"]:

                data["rejected"].append(
                    target
                )

            save_access(data)

            await callback.message.edit_text(
                callback.message.text
                + "\n\n"
                + "❌ <b>REJECTED</b>",
                parse_mode="HTML",
            )

            try:

                await bot.send_message(
                    target_id,
                    "❌ <b>Access မရသေးပါ။</b>\n\n"
                    "Owner က ဒီအချိန်မှာ "
                    "Bot အသုံးပြုခွင့် မပေးထားသေးပါ။",
                    parse_mode="HTML",
                )

            except Exception as e:

                print(
                    f"⚠️ Cannot notify user: {e}"
                )

            await callback.answer(
                "❌ User rejected"
            )

            return


# =========================================================
# MENU CALLBACK
# =========================================================

@dp.callback_query(
    F.data == "menu:main"
)
async def menu_main(
    callback: CallbackQuery
):

    if not is_approved(
        callback.from_user.id
    ):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    user_id = callback.from_user.id

    selected = user_voice.get(
        user_id,
        "anna"
    )

    current = (
        "Anna"
        if selected == "anna"
        else "Custom Voice"
    )

    await callback.answer()

    await callback.message.edit_text(
        "🎙️ <b>VoxCPM2 Myanmar Voice Bot</b>\n\n"
        f"🎙️ <b>Current Voice:</b> {current}\n\n"
        "အသုံးပြုလိုတဲ့ Voice ကိုရွေးပါ။",
        parse_mode="HTML",
        reply_markup=main_menu(current),
    )


# =========================================================
# HOW TO USE
# =========================================================

@dp.callback_query(
    F.data == "menu:help"
)
async def menu_help(
    callback: CallbackQuery
):

    if not is_approved(
        callback.from_user.id
    ):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    await callback.answer()

    await callback.message.edit_text(
        "ℹ️ <b>အသုံးပြုနည်း</b>\n\n"
        "1️⃣ <b>Anna</b> ကိုရွေးပါ\n"
        "→ Original V5 Voice ကို အသုံးပြုပါမယ်။\n\n"
        "2️⃣ <b>Custom Voice</b> ကိုရွေးပါ\n"
        "→ ကိုယ်အသုံးပြုလိုတဲ့ Voice File ကို ပို့ပါ။\n\n"
        "3️⃣ Voice ရွေးပြီးရင်\n"
        "→ မြန်မာစာကို ပို့ပါ။\n\n"
        "4️⃣ Bot က ရွေးထားတဲ့ Voice နဲ့\n"
        "→ မြန်မာအသံအဖြစ် ပြန်ပေးပါမယ်။",
        parse_mode="HTML",
        reply_markup=back_menu(),
    )


# =========================================================
# ANNA
# =========================================================

@dp.callback_query(
    F.data == "voice:anna"
)
async def select_anna(
    callback: CallbackQuery
):

    if not is_approved(
        callback.from_user.id
    ):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    user_id = callback.from_user.id

    user_voice[user_id] = "anna"

    await callback.answer(
        "👩 Anna ကို ရွေးပြီးပါပြီ"
    )

    await callback.message.edit_text(
        "👩 <b>Anna</b>\n\n"
        "Original V5 Voice ကို အသုံးပြုနေပါတယ်။\n\n"
        "အခု မြန်မာစာပို့လိုက်ပါ။",
        parse_mode="HTML",
        reply_markup=main_menu("Anna"),
    )


# =========================================================
# CUSTOM VOICE
# =========================================================

@dp.callback_query(
    F.data == "voice:custom"
)
async def select_custom(
    callback: CallbackQuery
):

    if not is_approved(
        callback.from_user.id
    ):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    user_id = callback.from_user.id

    custom_path = (
        CUSTOM_VOICE_DIR
        / f"{user_id}.wav"
    )

    if custom_path.exists():

        user_voice[user_id] = "custom"

        await callback.answer(
            "🎤 Custom Voice ကို ရွေးပြီးပါပြီ"
        )

        await callback.message.edit_text(
            "🎤 <b>Custom Voice</b>\n\n"
            "လက်ရှိသိမ်းထားတဲ့ Custom Voice ကို "
            "အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။\n\n"
            "Voice အသစ်ပြောင်းချင်ရင် "
            "ဒီနေရာမှာ Audio / Voice file အသစ်ပို့ပါ။",
            parse_mode="HTML",
            reply_markup=custom_voice_menu(),
        )

    else:

        await callback.answer()

        await callback.message.edit_text(
            "🎤 <b>Custom Voice</b>\n\n"
            "ကိုယ်အသုံးပြုလိုတဲ့ အသံဖိုင်ကို "
            "ဒီနေရာမှာ ပို့ပေးပါ။\n\n"
            "📌 Supported formats:\n"
            "• Telegram Voice\n"
            "• WAV\n"
            "• MP3\n"
            "• M4A\n"
            "• OGG\n"
            "• FLAC\n\n"
            "ဖိုင်ရောက်လာတာနဲ့ Custom Voice အဖြစ် "
            "သိမ်းပေးပါမယ်။",
            parse_mode="HTML",
            reply_markup=custom_voice_menu(),
        )


# =========================================================
# CUSTOM VOICE PATH
# =========================================================

def get_custom_voice_path(
    user_id: int
):

    return (
        CUSTOM_VOICE_DIR
        / f"{user_id}.wav"
    )


def has_custom_voice(
    user_id: int
):

    return get_custom_voice_path(
        user_id
    ).exists()


# =========================================================
# SAVE CUSTOM VOICE
# =========================================================

async def save_custom_voice(
    message: Message,
    user_id: int
):

    custom_path = get_custom_voice_path(
        user_id
    )

    temp_original = None
    temp_converted = None

    try:

        file_id = None

        if message.voice:

            file_id = message.voice.file_id

        elif message.audio:

            file_id = message.audio.file_id

        elif message.document:

            file_id = message.document.file_id

        else:

            return False

        telegram_file = await bot.get_file(
            file_id
        )

        suffix = ".dat"

        if message.voice:

            suffix = ".ogg"

        elif message.audio:

            if message.audio.file_name:

                suffix = (
                    Path(
                        message.audio.file_name
                    ).suffix
                    or ".mp3"
                )

        elif message.document:

            if message.document.file_name:

                suffix = (
                    Path(
                        message.document.file_name
                    ).suffix
                    or ".wav"
                )

        temp_original = (
            Path(
                tempfile.gettempdir()
            )
            / f"voice_{user_id}{suffix}"
        )

        await bot.download_file(
            telegram_file.file_path,
            destination=str(
                temp_original
            ),
        )

        temp_converted = (
            Path(
                tempfile.gettempdir()
            )
            / f"voice_{user_id}_converted.wav"
        )

        process = (
            await asyncio.create_subprocess_exec(
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
        )

        return_code = await process.wait()

        if (
            return_code == 0
            and temp_converted.exists()
        ):

            shutil.copy2(
                temp_converted,
                custom_path,
            )

        else:

            shutil.copy2(
                temp_original,
                custom_path,
            )

        # Validate
        sf.read(
            str(custom_path),
            always_2d=False,
        )

        user_voice[user_id] = "custom"

        print(
            f"✅ Custom Voice saved: {user_id}"
        )

        return True

    except Exception as e:

        print(
            f"❌ Custom Voice Error: {e}"
        )

        if custom_path.exists():

            try:
                custom_path.unlink()
            except Exception:
                pass

        return False

    finally:

        if (
            temp_original
            and temp_original.exists()
        ):

            try:
                temp_original.unlink()
            except Exception:
                pass

        if (
            temp_converted
            and temp_converted.exists()
        ):

            try:
                temp_converted.unlink()
            except Exception:
                pass


# =========================================================
# AUDIO / VOICE UPLOAD
# =========================================================

@dp.message(
    F.voice | F.audio | F.document
)
async def voice_upload_handler(
    message: Message
):

    user_id = message.from_user.id

    if not is_approved(user_id):

        await message.answer(
            "🔐 <b>Access မရသေးပါ။</b>\n\n"
            "Owner ခွင့်ပြုချက်ရပြီးမှ "
            "Bot ကို အသုံးပြုနိုင်ပါတယ်။",
            parse_mode="HTML",
        )

        return

    # Check document extension
    if message.document:

        filename = (
            message.document.file_name
            or ""
        ).lower()

        allowed = (
            ".wav",
            ".mp3",
            ".m4a",
            ".ogg",
            ".flac",
            ".aac",
        )

        if not filename.endswith(
            allowed
        ):

            await message.answer(
                "❌ Audio file မဟုတ်ပါဘူး။\n\n"
                "WAV / MP3 / M4A / OGG / FLAC "
                "တစ်ခုခု ပို့ပါ။"
            )

            return

    status = await message.answer(
        "🎤 <b>Custom Voice</b>\n\n"
        "⏳ Voice ကို သိမ်းနေပါတယ်...",
        parse_mode="HTML",
    )

    success = await save_custom_voice(
        message,
        user_id
    )

    if success:

        await status.edit_text(
            "✅ <b>Custom Voice သိမ်းပြီးပါပြီ။</b>\n\n"
            "🎤 Current Voice: <b>Custom Voice</b>\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            parse_mode="HTML",
            reply_markup=main_menu(
                "Custom Voice"
            ),
        )

    else:

        await status.edit_text(
            "❌ <b>Custom Voice သိမ်းမရပါ။</b>\n\n"
            "WAV / MP3 / M4A / Telegram Voice "
            "နဲ့ ပြန်ပို့ကြည့်ပါ။",
            parse_mode="HTML",
            reply_markup=main_menu("Anna"),
        )


# =========================================================
# GET USER VOICE
# =========================================================

def get_user_voice(
    user_id: int
):

    selected = user_voice.get(
        user_id,
        "anna"
    )

    if selected == "custom":

        custom_path = get_custom_voice_path(
            user_id
        )

        if custom_path.exists():

            return (
                custom_path,
                "Custom Voice"
            )

        user_voice[user_id] = "anna"

    return (
        ANNA_VOICE,
        "Anna"
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
# TTS
# =========================================================

async def generate_voice(
    text: str,
    reference_wav: Path,
    out: Path,
    voice_name: str,
):

    async with tts_lock:

        print(
            f"🎙️ Generating: {voice_name}"
        )

        print(
            f"Reference: {reference_wav}"
        )

        # -------------------------------------------------
        # IMPORTANT
        #
        # Do NOT use Anna's PROMPT_TEXT for Custom Voice.
        # reference_wav_path is used for voice cloning.
        # -------------------------------------------------

        if voice_name == "Anna":

            wav = await asyncio.to_thread(
                model.generate,
                text=text,
                reference_wav_path=str(
                    reference_wav
                ),
                inference_timesteps=10,
                cfg_value=2.0,
                retry_badcase=False,
                max_len=1000,
            )

        else:

            wav = await asyncio.to_thread(
                model.generate,
                text=text,
                reference_wav_path=str(
                    reference_wav
                ),
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
            f"✅ Generated "
            f"sample_rate={sample_rate}"
        )


# =========================================================
# TEXT → SPEECH
# =========================================================

@dp.message(F.text)
async def text_handler(
    message: Message
):

    user_id = message.from_user.id

    if not is_approved(user_id):

        await message.answer(
            "🔐 <b>Access မရသေးပါ။</b>\n\n"
            "Owner ခွင့်ပြုချက်ရပြီးမှ "
            "Bot ကို အသုံးပြုနိုင်ပါတယ်။",
            parse_mode="HTML",
        )

        return

    text = message.text.strip()

    if not text:

        return

    if len(text) > 1000:

        await message.answer(
            "❌ စာသားက 1000 characters "
            "ထက်မကျော်ရပါ။"
        )

        return

    reference_wav, voice_name = (
        get_user_voice(user_id)
    )

    if not reference_wav.exists():

        await message.answer(
            "❌ Voice file မတွေ့ပါဘူး။\n\n"
            "Anna voice file ကို စစ်ပေးပါ။"
        )

        return

    status = await message.answer(
        f"🎙️ <b>{voice_name}</b>\n\n"
        "⏳ မြန်မာအသံထုတ်နေပါတယ်...",
        parse_mode="HTML",
    )

    output_file = (
        Path(
            tempfile.gettempdir()
        )
        / f"tts_{user_id}_{message.message_id}.wav"
    )

    try:

        await generate_voice(
            text=text,
            reference_wav=reference_wav,
            out=output_file,
            voice_name=voice_name,
        )

        await status.edit_text(
            f"🎙️ <b>{voice_name}</b>\n\n"
            "✅ အသံထွက်ပြီးပါပြီ။",
            parse_mode="HTML",
        )

        await message.answer_audio(
            audio=FSInputFile(
                str(output_file)
            ),
            title=f"{voice_name} Voice",
        )

    except Exception as e:

        print(
            f"❌ TTS Error: {e}"
        )

        await status.edit_text(
            "❌ TTS Error ဖြစ်သွားပါတယ်။\n\n"
            f"<code>{str(e)[:500]}</code>",
            parse_mode="HTML",
        )

    finally:

        if output_file.exists():

            try:
                output_file.unlink()
            except Exception:
                pass


# =========================================================
# STARTUP
# =========================================================

async def main():

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN မတွေ့ပါ။"
        )

    if OWNER_ID == 0:

        raise RuntimeError(
            "OWNER_ID မထည့်ရသေးပါ။"
        )

    if not ANNA_VOICE.exists():

        raise FileNotFoundError(
            f"Anna voice file မတွေ့ပါ: "
            f"{ANNA_VOICE}"
        )

    print(
        "🚀 Starting VoxCPM2 Telegram Bot..."
    )

    print(
        f"👑 Owner ID: {OWNER_ID}"
    )

    await asyncio.to_thread(
        download_model
    )

    await asyncio.to_thread(
        load_model
    )

    print(
        "✅ VoxCPM2 Telegram Bot READY"
    )

    await dp.start_polling(
        bot
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    asyncio.run(main())
