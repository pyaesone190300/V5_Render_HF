import os
import json
import asyncio
import subprocess
from pathlib import Path

import soundfile as sf
import torch
import numpy as np

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    FSInputFile,
)
from aiogram.client.default import DefaultBotProperties

from huggingface_hub import snapshot_download
from voxcpm import VoxCPM


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

MODEL_REPO = "pyaesone190300/VoxCPM2"

BASE_DIR = Path("/app")

MODEL_DIR = BASE_DIR / "model"

# ------------------------------------------------------------
# Anna
# ------------------------------------------------------------

ANNA_VOICE = BASE_DIR / "vvipvoice_v5_ref.wav"
ANNA_NAME = "Anna"

# ------------------------------------------------------------
# Fangyung
# ------------------------------------------------------------

FANGYUNG_VOICE = BASE_DIR / "Fangyung_vvipvoice.wav"
FANGYUNG_NAME = "Fangyung"

# ------------------------------------------------------------
# Htun
# ------------------------------------------------------------

HTUN_VOICE = BASE_DIR / "Htun_vvipvoice.wav"
HTUN_NAME = "Htun"

# ------------------------------------------------------------
# Phyo
# ------------------------------------------------------------

PHYO_VOICE = BASE_DIR / "Phyo_vvipvoice.wav"
PHYO_NAME = "Phyo"

# ------------------------------------------------------------
# Pyae
# ------------------------------------------------------------

PYAE_VOICE = BASE_DIR / "Pyae_vvipvoice.wav"
PYAE_NAME = "Pyae"

# ------------------------------------------------------------
# Zxee
# ------------------------------------------------------------

ZXEE_VOICE = BASE_DIR / "Zxee_vvipvoice.wav"
ZXEE_NAME = "Zxee"

# ------------------------------------------------------------
# Custom Voice
# ------------------------------------------------------------

CUSTOM_VOICE_DIR = BASE_DIR / "custom_voices"
CUSTOM_VOICE_DIR.mkdir(parents=True, exist_ok=True)

ACCESS_FILE = BASE_DIR / "access.json"


# ============================================================
# GLOBALS
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN မရှိပါ")

if not OWNER_ID:
    raise RuntimeError("OWNER_ID မရှိပါ")


bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(
        parse_mode=ParseMode.HTML
    )
)

dp = Dispatcher()

model = None

# User selected voice
# anna / fangyung / htun / phyo / pyae / zxee / custom
user_voice = {}

# TTS lock
tts_lock = asyncio.Lock()


# ============================================================
# ACCESS SYSTEM
# ============================================================

def load_access():
    if not ACCESS_FILE.exists():
        data = {
            "approved": [],
            "pending": []
        }

        ACCESS_FILE.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

        return data

    try:
        data = json.loads(
            ACCESS_FILE.read_text(
                encoding="utf-8"
            )
        )

        if "approved" not in data:
            data["approved"] = []

        if "pending" not in data:
            data["pending"] = []

        return data

    except Exception:
        return {
            "approved": [],
            "pending": []
        }


def save_access(data):
    ACCESS_FILE.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def is_owner(user_id: int):
    return user_id == OWNER_ID


def is_approved(user_id: int):
    if is_owner(user_id):
        return True

    data = load_access()

    return user_id in data.get("approved", [])


def is_pending(user_id: int):
    data = load_access()

    return user_id in data.get("pending", [])


# ============================================================
# CUSTOM VOICE
# ============================================================

def get_custom_voice_path(user_id: int):
    return CUSTOM_VOICE_DIR / f"{user_id}.wav"


# ============================================================
# VOICE MENU
# ============================================================

def main_menu(current="Anna"):

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"👩 Anna{'  ✓' if current == 'Anna' else ''}",
                    callback_data="voice:anna",
                ),
                InlineKeyboardButton(
                    text=f"🎙️ Fangyung{'  ✓' if current == 'Fangyung' else ''}",
                    callback_data="voice:fangyung",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=f"👦 Htun{'  ✓' if current == 'Htun' else ''}",
                    callback_data="voice:htun",
                ),
                InlineKeyboardButton(
                    text=f"👦 Phyo{'  ✓' if current == 'Phyo' else ''}",
                    callback_data="voice:phyo",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=f"👦 Pyae{'  ✓' if current == 'Pyae' else ''}",
                    callback_data="voice:pyae",
                ),
                InlineKeyboardButton(
                    text=f"👦 Zxee{'  ✓' if current == 'Zxee' else ''}",
                    callback_data="voice:zxee",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=f"🎤 Custom{'  ✓' if current == 'Custom Voice' else ''}",
                    callback_data="voice:custom",
                )
            ],
            [
                InlineKeyboardButton(
                    text="ℹ️ How to Use",
                    callback_data="menu:help",
                )
            ],
        ]
    )


# ============================================================
# START
# ============================================================

@dp.message(CommandStart())
async def start_handler(message: Message):

    uid = message.from_user.id

    # Default voice = Anna
    user_voice[uid] = "anna"

    if not is_approved(uid) and not is_owner(uid):

        if is_pending(uid):

            await message.answer(
                "⏳ <b>Access Request Pending</b>\n\n"
                "Owner က approve လုပ်ပေးဖို့ စောင့်ပေးပါ။"
            )

            return

        data = load_access()

        if uid not in data["pending"]:
            data["pending"].append(uid)
            save_access(data)

        await message.answer(
            "🔐 <b>Access Required</b>\n\n"
            "ဒီ Bot ကို အသုံးပြုဖို့ Owner Approval လိုအပ်ပါတယ်။\n\n"
            "⏳ Access request ပို့ပြီးပါပြီ။"
        )

        try:

            await bot.send_message(
                OWNER_ID,
                "🔔 <b>New Access Request</b>\n\n"
                f"👤 Name: {message.from_user.full_name}\n"
                f"🆔 ID: <code>{uid}</code>\n\n"
                "Approve / Reject လုပ်နိုင်ပါတယ်။",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="✅ Approve",
                                callback_data=f"access:approve:{uid}"
                            ),
                            InlineKeyboardButton(
                                text="❌ Reject",
                                callback_data=f"access:reject:{uid}"
                            ),
                        ]
                    ]
                )
            )

        except Exception as e:
            print("Owner notification error:", e)

        return

    await message.answer(
        "🎙️ <b>VoxCPM2 Voice Bot</b>\n\n"
        "အသံရွေးချယ်ပြီး မြန်မာစာပို့ပါ။\n\n"
        "👩 Anna — Original Voice\n"
        "🎙️ Fangyung — Natural Boy Voice\n"
        "👦 Htun — Htun Voice\n"
        "👦 Phyo — Phyo Voice\n"
        "👦 Pyae — Pyae Voice\n"
        "👦 Zxee — Zxee Voice\n"
        "🎤 Custom Voice — ကိုယ်ပိုင် Voice",
        reply_markup=main_menu("Anna")
    )


# ============================================================
# OWNER APPROVE
# ============================================================

@dp.callback_query(F.data.startswith("access:approve:"))
async def approve_user(callback: CallbackQuery):

    if not is_owner(callback.from_user.id):
        await callback.answer(
            "❌ Owner Only",
            show_alert=True
        )
        return

    try:
        uid = int(
            callback.data.split(":")[-1]
        )

    except Exception:
        await callback.answer(
            "❌ Invalid User ID",
            show_alert=True
        )
        return

    data = load_access()

    if uid not in data["approved"]:
        data["approved"].append(uid)

    if uid in data["pending"]:
        data["pending"].remove(uid)

    save_access(data)

    await callback.answer(
        "✅ Approved"
    )

    try:

        await bot.send_message(
            uid,
            "✅ <b>Access Approved!</b>\n\n"
            "VoxCPM2 Voice Bot ကို အသုံးပြုနိုင်ပါပြီ။\n\n"
            "အသံရွေးပြီး မြန်မာစာပို့ပါ။",
            reply_markup=main_menu("Anna")
        )

    except Exception as e:
        print("Approved user notification error:", e)

    try:

        await callback.message.edit_text(
            "✅ <b>User Approved</b>\n\n"
            f"User ID: <code>{uid}</code>"
        )

    except Exception:
        pass


# ============================================================
# OWNER REJECT
# ============================================================

@dp.callback_query(F.data.startswith("access:reject:"))
async def reject_user(callback: CallbackQuery):

    if not is_owner(callback.from_user.id):
        await callback.answer(
            "❌ Owner Only",
            show_alert=True
        )
        return

    try:
        uid = int(
            callback.data.split(":")[-1]
        )

    except Exception:
        await callback.answer(
            "❌ Invalid User ID",
            show_alert=True
        )
        return

    data = load_access()

    if uid in data["pending"]:
        data["pending"].remove(uid)

    if uid in data["approved"]:
        data["approved"].remove(uid)

    save_access(data)

    await callback.answer(
        "❌ Rejected"
    )

    try:

        await bot.send_message(
            uid,
            "❌ <b>Access Rejected</b>\n\n"
            "Owner မှ Access မပေးသေးပါ။"
        )

    except Exception as e:
        print("Reject notification error:", e)

    try:

        await callback.message.edit_text(
            "❌ <b>User Rejected</b>\n\n"
            f"User ID: <code>{uid}</code>"
        )

    except Exception:
        pass


# ============================================================
# ANNA SELECT
# ============================================================

@dp.callback_query(F.data == "voice:anna")
async def select_anna(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    if not ANNA_VOICE.exists():

        await callback.answer(
            "❌ Anna voice file မတွေ့ပါ",
            show_alert=True
        )

        return

    user_voice[uid] = "anna"

    await callback.answer(
        "👩 Anna ကို ရွေးပြီးပါပြီ"
    )

    try:

        await callback.message.edit_text(
            "👩 <b>Anna</b>\n\n"
            "Original Anna Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            reply_markup=main_menu("Anna")
        )

    except Exception:
        pass


# ============================================================
# FANGYUNG SELECT
# ============================================================

@dp.callback_query(F.data == "voice:fangyung")
async def select_fangyung(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    if not FANGYUNG_VOICE.exists():

        await callback.answer(
            "❌ Fangyung voice file မတွေ့ပါ",
            show_alert=True
        )

        return

    user_voice[uid] = "fangyung"

    await callback.answer(
        "🎙️ Fangyung ကို ရွေးပြီးပါပြီ"
    )

    try:

        await callback.message.edit_text(
            "🎙️ <b>Fangyung</b>\n\n"
            "Fangyung Natural Boy Voice ကို "
            "အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            reply_markup=main_menu("Fangyung")
        )

    except Exception:
        pass


# ============================================================
# HTUN SELECT
# ============================================================

@dp.callback_query(F.data == "voice:htun")
async def select_htun(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    if not HTUN_VOICE.exists():

        await callback.answer(
            "❌ Htun voice file မတွေ့ပါ",
            show_alert=True
        )

        return

    user_voice[uid] = "htun"

    await callback.answer(
        "👦 Htun ကို ရွေးပြီးပါပြီ"
    )

    try:

        await callback.message.edit_text(
            "👦 <b>Htun</b>\n\n"
            "Htun Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            reply_markup=main_menu("Htun")
        )

    except Exception:
        pass


# ============================================================
# PHYO SELECT
# ============================================================

@dp.callback_query(F.data == "voice:phyo")
async def select_phyo(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    if not PHYO_VOICE.exists():

        await callback.answer(
            "❌ Phyo voice file မတွေ့ပါ",
            show_alert=True
        )

        return

    user_voice[uid] = "phyo"

    await callback.answer(
        "👦 Phyo ကို ရွေးပြီးပါပြီ"
    )

    try:

        await callback.message.edit_text(
            "👦 <b>Phyo</b>\n\n"
            "Phyo Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            reply_markup=main_menu("Phyo")
        )

    except Exception:
        pass


# ============================================================
# PYAE SELECT
# ============================================================

@dp.callback_query(F.data == "voice:pyae")
async def select_pyae(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    if not PYAE_VOICE.exists():

        await callback.answer(
            "❌ Pyae voice file မတွေ့ပါ",
            show_alert=True
        )

        return

    user_voice[uid] = "pyae"

    await callback.answer(
        "👦 Pyae ကို ရွေးပြီးပါပြီ"
    )

    try:

        await callback.message.edit_text(
            "👦 <b>Pyae</b>\n\n"
            "Pyae Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            reply_markup=main_menu("Pyae")
        )

    except Exception:
        pass


# ============================================================
# ZXEE SELECT
# ============================================================

@dp.callback_query(F.data == "voice:zxee")
async def select_zxee(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    if not ZXEE_VOICE.exists():

        await callback.answer(
            "❌ Zxee voice file မတွေ့ပါ",
            show_alert=True
        )

        return

    user_voice[uid] = "zxee"

    await callback.answer(
        "👦 Zxee ကို ရွေးပြီးပါပြီ"
    )

    try:

        await callback.message.edit_text(
            "👦 <b>Zxee</b>\n\n"
            "Zxee Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            reply_markup=main_menu("Zxee")
        )

    except Exception:
        pass


# ============================================================
# CUSTOM SELECT
# ============================================================

@dp.callback_query(F.data == "voice:custom")
async def select_custom(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    custom_path = get_custom_voice_path(uid)

    if not custom_path.exists():

        user_voice[uid] = "custom"

        await callback.answer(
            "🎤 Custom Voice upload လုပ်ပါ",
            show_alert=True
        )

        try:

            await callback.message.edit_text(
                "🎤 <b>Custom Voice</b>\n\n"
                "ကိုယ်ပိုင် Voice file တစ်ခုကို ပို့ပါ။\n\n"
                "အသုံးပြုနိုင်သော format:\n"
                "• Voice\n"
                "• Audio\n"
                "• WAV / MP3 / M4A\n\n"
                "Voice file ပို့ပြီးရင် "
                "Custom Voice အဖြစ် အသုံးပြုနိုင်ပါပြီ။",
                reply_markup=main_menu("Custom Voice")
            )

        except Exception:
            pass

        return

    user_voice[uid] = "custom"

    await callback.answer(
        "🎤 Custom Voice ကို ရွေးပြီးပါပြီ"
    )

    try:

        await callback.message.edit_text(
            "🎤 <b>Custom Voice</b>\n\n"
            "သင့် Custom Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            reply_markup=main_menu("Custom Voice")
        )

    except Exception:
        pass


# ============================================================
# HELP
# ============================================================

@dp.callback_query(F.data == "menu:help")
async def help_menu(callback: CallbackQuery):

    uid = callback.from_user.id

    if not is_approved(uid):

        await callback.answer(
            "🔐 Access မရသေးပါ",
            show_alert=True
        )

        return

    await callback.answer()

    curr = user_voice.get(uid, "anna")
    curr_name = "Anna"
    if curr == "fangyung":
        curr_name = "Fangyung"
    elif curr == "htun":
        curr_name = "Htun"
    elif curr == "phyo":
        curr_name = "Phyo"
    elif curr == "pyae":
        curr_name = "Pyae"
    elif curr == "zxee":
        curr_name = "Zxee"
    elif curr == "custom":
        curr_name = "Custom Voice"

    try:

        await callback.message.edit_text(
            "ℹ️ <b>How to Use</b>\n\n"
            "👩 <b>Anna</b> — Original Anna Voice\n"
            "🎙️ <b>Fangyung</b> — Natural Boy Voice\n"
            "👦 <b>Htun</b> — Htun Voice\n"
            "👦 <b>Phyo</b> — Phyo Voice\n"
            "👦 <b>Pyae</b> — Pyae Voice\n"
            "👦 <b>Zxee</b> — Zxee Voice\n"
            "🎤 <b>Custom Voice</b> — ကိုယ်ပိုင် Voice file upload လုပ်ပြီး အသုံးပြုနိုင်ပါတယ်။\n\n"
            "📝 Voice ရွေးပြီးနောက် မြန်မာစာပို့ပါ။",
            reply_markup=main_menu(curr_name)
        )

    except Exception:
        pass


# ============================================================
# VOICE UPLOAD
# ============================================================

async def download_telegram_file(
    message: Message,
    file_id: str,
    output_path: Path
):

    tg_file = await bot.get_file(file_id)

    await bot.download_file(
        tg_file.file_path,
        destination=output_path
    )


async def convert_to_wav(
    input_path: Path,
    output_path: Path
):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "48000",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:

        print(
            stderr.decode(
                errors="ignore"
            )
        )

        raise RuntimeError(
            "FFmpeg conversion failed"
        )


# ============================================================
# TELEGRAM VOICE
# ============================================================

@dp.message(F.voice)
async def receive_voice(message: Message):

    uid = message.from_user.id

    if not is_approved(uid):

        await message.answer(
            "🔐 Access မရသေးပါ။"
        )

        return

    user_voice[uid] = "custom"

    raw_path = (
        BASE_DIR
        / f"temp_{uid}_voice.ogg"
    )

    output_path = get_custom_voice_path(uid)

    try:

        await message.answer(
            "⏳ <b>Custom Voice သိမ်းနေပါတယ်...</b>"
        )

        await download_telegram_file(
            message,
            message.voice.file_id,
            raw_path
        )

        await convert_to_wav(
            raw_path,
            output_path
        )

        await message.answer(
            "✅ <b>Custom Voice သိမ်းပြီးပါပြီ</b>\n\n"
            "🎤 Custom Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့ပါ။",
            reply_markup=main_menu("Custom Voice")
        )

    except Exception as e:

        print("Voice upload error:", e)

        await message.answer(
            "❌ Voice file ပြောင်းလဲရာမှာ Error ဖြစ်ပါတယ်။"
        )

    finally:

        if raw_path.exists():
            raw_path.unlink()


# ============================================================
# TELEGRAM AUDIO
# ============================================================

@dp.message(F.audio)
async def receive_audio(message: Message):

    uid = message.from_user.id

    if not is_approved(uid):

        await message.answer(
            "🔐 Access မရသေးပါ။"
        )

        return

    user_voice[uid] = "custom"

    extension = ".audio"

    if message.audio.file_name:
        extension = (
            Path(
                message.audio.file_name
            ).suffix
            or ".audio"
        )

    raw_path = (
        BASE_DIR
        / f"temp_{uid}_audio{extension}"
    )

    output_path = get_custom_voice_path(uid)

    try:

        await message.answer(
            "⏳ <b>Custom Voice သိမ်းနေပါတယ်...</b>"
        )

        await download_telegram_file(
            message,
            message.audio.file_id,
            raw_path
        )

        await convert_to_wav(
            raw_path,
            output_path
        )

        await message.answer(
            "✅ <b>Custom Voice သိမ်းပြီးပါပြီ</b>\n\n"
            "🎤 Custom Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့ပါ။",
            reply_markup=main_menu("Custom Voice")
        )

    except Exception as e:

        print("Audio upload error:", e)

        await message.answer(
            "❌ Audio file ပြောင်းလဲရာမှာ Error ဖြစ်ပါတယ်။"
        )

    finally:

        if raw_path.exists():
            raw_path.unlink()


# ============================================================
# TELEGRAM DOCUMENT
# ============================================================

@dp.message(F.document)
async def receive_document(message: Message):

    uid = message.from_user.id

    if not is_approved(uid):

        await message.answer(
            "🔐 Access မရသေးပါ။"
        )

        return

    user_voice[uid] = "custom"

    filename = (
        message.document.file_name
        or "voice"
    )

    extension = (
        Path(filename).suffix
        or ".audio"
    )

    raw_path = (
        BASE_DIR
        / f"temp_{uid}_document{extension}"
    )

    output_path = get_custom_voice_path(uid)

    try:

        await message.answer(
            "⏳ <b>Custom Voice သိမ်းနေပါတယ်...</b>"
        )

        await download_telegram_file(
            message,
            message.document.file_id,
            raw_path
        )

        await convert_to_wav(
            raw_path,
            output_path
        )

        await message.answer(
            "✅ <b>Custom Voice သိမ်းပြီးပါပြီ</b>\n\n"
            "🎤 Custom Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့ပါ။",
            reply_markup=main_menu("Custom Voice")
        )

    except Exception as e:

        print("Document upload error:", e)

        await message.answer(
            "❌ Voice file ပြောင်းလဲရာမှာ Error ဖြစ်ပါတယ်။"
        )

    finally:

        if raw_path.exists():
            raw_path.unlink()


# ============================================================
# GET USER VOICE
# ============================================================

def get_user_voice(user_id: int):

    selected = user_voice.get(
        user_id,
        "anna"
    )

    # --------------------------------------------------------
    # Anna
    # --------------------------------------------------------

    if selected == "anna":

        return (
            ANNA_VOICE,
            ANNA_NAME
        )

    # --------------------------------------------------------
    # Fangyung
    # --------------------------------------------------------

    if selected == "fangyung":

        if FANGYUNG_VOICE.exists():

            return (
                FANGYUNG_VOICE,
                FANGYUNG_NAME
            )

        user_voice[user_id] = "anna"

        return (
            ANNA_VOICE,
            ANNA_NAME
        )

    # --------------------------------------------------------
    # Htun
    # --------------------------------------------------------

    if selected == "htun":

        if HTUN_VOICE.exists():

            return (
                HTUN_VOICE,
                HTUN_NAME
            )

        user_voice[user_id] = "anna"

        return (
            ANNA_VOICE,
            ANNA_NAME
        )

    # --------------------------------------------------------
    # Phyo
    # --------------------------------------------------------

    if selected == "phyo":

        if PHYO_VOICE.exists():

            return (
                PHYO_VOICE,
                PHYO_NAME
            )

        user_voice[user_id] = "anna"

        return (
            ANNA_VOICE,
            ANNA_NAME
        )

    # --------------------------------------------------------
    # Pyae
    # --------------------------------------------------------

    if selected == "pyae":

        if PYAE_VOICE.exists():

            return (
                PYAE_VOICE,
                PYAE_NAME
            )

        user_voice[user_id] = "anna"

        return (
            ANNA_VOICE,
            ANNA_NAME
        )

    # --------------------------------------------------------
    # Zxee
    # --------------------------------------------------------

    if selected == "zxee":

        if ZXEE_VOICE.exists():

            return (
                ZXEE_VOICE,
                ZXEE_NAME
            )

        user_voice[user_id] = "anna"

        return (
            ANNA_VOICE,
            ANNA_NAME
        )

    # --------------------------------------------------------
    # Custom
    # --------------------------------------------------------

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
            ANNA_NAME
        )

    # --------------------------------------------------------
    # Unknown selection
    # --------------------------------------------------------

    user_voice[user_id] = "anna"

    return (
        ANNA_VOICE,
            ANNA_NAME
    )


# ============================================================
# SAMPLE RATE
# ============================================================

def get_sample_rate():

    if hasattr(model, "tts_model"):

        if hasattr(
            model.tts_model,
            "sample_rate"
        ):

            return model.tts_model.sample_rate

    return 24000


# ============================================================
# GENERATE VOICE (Auto Chunking)
# ============================================================

async def generate_voice(
    text,
    reference_wav,
    output_path,
    voice_name
):

    async with tts_lock:

        print()
        print("=" * 60)
        print("🎙️ Generating Voice (Auto Chunking)")
        print("Voice:", voice_name)
        print("Text Length:", len(text))
        print("=" * 60)

        if voice_name == "Anna":
            inference_timesteps = 30
        else:
            inference_timesteps = 30

        # မြန်မာစာ ပုဒ်မ (။) သို့မဟုတ် Enter ခေါက်ထားသော နေရာများမှ စာကြောင်းခွဲခြင်း
        text = text.replace('\n', '။')
        sentences = [s.strip() + "။" for s in text.split('။') if s.strip()]
        
        # အကယ်၍ ခွဲစရာမရှိလျှင် မူလစာသားအတိုင်းထားရန်
        if not sentences:
            sentences = [text]

        combined_wav = []

        with torch.inference_mode():
            for idx, chunk in enumerate(sentences):
                print(f"⏳ Generating part {idx+1}/{len(sentences)}...")
                
                # အပိုင်းတစ်ပိုင်းချင်းစီအတွက် အသံထုတ်ခြင်း
                wav_chunk = await asyncio.to_thread(
                    model.generate,
                    text=chunk,
                    reference_wav_path=str(reference_wav),
                    cfg_value=2.0,
                    inference_timesteps=inference_timesteps,
                    retry_badcase=False,
                    max_len=2000,
                )
                combined_wav.append(wav_chunk)
                
                # စာကြောင်းတစ်ကြောင်းနဲ့ တစ်ကြောင်းကြား အသံတိတ် (Silence) ၀.၅ စက္ကန့် ထည့်ခြင်း
                silence = np.zeros(int(get_sample_rate() * 0.5))
                combined_wav.append(silence)

        # ထွက်လာသမျှ အသံဖိုင်အပိုင်းများကို တစ်ခုတည်းဖြစ်အောင် ပြန်ဆက်ခြင်း
        final_wav = np.concatenate(combined_wav)

        # အသံဖိုင်အဖြစ် သိမ်းဆည်းခြင်း
        sf.write(
            str(output_path),
            final_wav,
            get_sample_rate()
        )

        print(
            "✅ Generated Combined Audio:",
            output_path
        )



# ============================================================
# TEXT TO SPEECH
# ============================================================

@dp.message(F.text)
async def text_to_speech(message: Message):

    uid = message.from_user.id

    if not is_approved(uid):

        await message.answer(
            "🔐 Access မရသေးပါ။\n\n"
            "/start ကိုနှိပ်ပြီး Access Request ပို့ပါ။"
        )

        return

    text = message.text.strip()

    if not text:

        return

    if len(text) > 3000:

        await message.answer(
            "❌ စာသားအရှည်ဆုံး 3000 characters အထိသာ "
            "အသုံးပြုနိုင်ပါတယ်။"
        )

        return

    reference_wav, voice_name = get_user_voice(
        uid
    )

    if not reference_wav.exists():

        await message.answer(
            f"❌ <b>{voice_name}</b> reference voice "
            "file မတွေ့ပါ။"
        )

        return

    status = await message.answer(
        f"🎙️ <b>{voice_name}</b>\n\n"
        "⏳ အသံထုတ်နေပါတယ်..."
    )

    output_path = (
        BASE_DIR
        / f"tts_{uid}.wav"
    )

    try:

        await generate_voice(
            text=text,
            reference_wav=reference_wav,
            output_path=output_path,
            voice_name=voice_name
        )

        try:
            await status.delete()
        except Exception:
            pass

        voice_file = FSInputFile(output_path)

        await message.answer_voice(
            voice=voice_file,
            caption=(
                f"🎙️ <b>{voice_name}</b>"
            ),
            reply_markup=main_menu(
                voice_name
            )
        )

    except Exception as e:

        print()
        print("❌ TTS ERROR")
        print(e)

        try:

            await status.edit_text(
                "❌ <b>Voice generation failed</b>\n\n"
                f"<code>{str(e)[:1000]}</code>"
            )

        except Exception:
            await message.answer(
                "❌ Voice generation failed."
            )

    finally:

        if output_path.exists():

            try:
                output_path.unlink()
            except Exception:
                pass


# ============================================================
# MODEL DOWNLOAD
# ============================================================

async def download_model():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_files = list(
        MODEL_DIR.rglob("*")
    )

    if existing_files:

        print(
            "✅ VoxCPM2 model already exists."
        )

        return

    print()
    print("=" * 60)
    print("📥 Downloading VoxCPM2 Model")
    print("=" * 60)

    await asyncio.to_thread(
        snapshot_download,
        repo_id=MODEL_REPO,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    print(
        "✅ Model download complete."
    )


# ============================================================
# LOAD MODEL
# ============================================================

async def load_model():

    global model

    print()
    print("=" * 60)
    print("🧠 Loading VoxCPM2")
    print("=" * 60)

    model = await asyncio.to_thread(
        VoxCPM.from_pretrained,
        str(MODEL_DIR),
        load_denoiser=False,
        local_files_only=True,
        optimize=False,
        device="cpu",
    )

    print()
    print("✅ VoxCPM2 Loaded Successfully")
    print("=" * 60)


# ============================================================
# STARTUP
# ============================================================

async def main():

    print()
    print("=" * 60)
    print("🚀 VoxCPM2 Telegram Bot")
    print("=" * 60)

    # --------------------------------------------------------
    # Check Anna
    # --------------------------------------------------------

    if not ANNA_VOICE.exists():

        raise FileNotFoundError(
            f"Anna voice file မတွေ့ပါ:\n"
            f"{ANNA_VOICE}"
        )

    print(
        "✅ Anna Voice:",
        ANNA_VOICE
    )

    # --------------------------------------------------------
    # Check Fangyung
    # --------------------------------------------------------

    if not FANGYUNG_VOICE.exists():

        raise FileNotFoundError(
            f"Fangyung voice file မတွေ့ပါ:\n"
            f"{FANGYUNG_VOICE}"
        )

    print(
        "✅ Fangyung Voice:",
        FANGYUNG_VOICE
    )

    # --------------------------------------------------------
    # Check Htun
    # --------------------------------------------------------

    if not HTUN_VOICE.exists():

        raise FileNotFoundError(
            f"Htun voice file မတွေ့ပါ:\n"
            f"{HTUN_VOICE}"
        )

    print(
        "✅ Htun Voice:",
        HTUN_VOICE
    )

    # --------------------------------------------------------
    # Check Phyo
    # --------------------------------------------------------

    if not PHYO_VOICE.exists():

        raise FileNotFoundError(
            f"Phyo voice file မတွေ့ပါ:\n"
            f"{PHYO_VOICE}"
        )

    print(
        "✅ Phyo Voice:",
        PHYO_VOICE
    )

    # --------------------------------------------------------
    # Check Pyae
    # --------------------------------------------------------

    if not PYAE_VOICE.exists():

        raise FileNotFoundError(
            f"Pyae voice file မတွေ့ပါ:\n"
            f"{PYAE_VOICE}"
        )

    print(
        "✅ Pyae Voice:",
        PYAE_VOICE
    )

    # --------------------------------------------------------
    # Check Zxee
    # --------------------------------------------------------

    if not ZXEE_VOICE.exists():

        raise FileNotFoundError(
            f"Zxee voice file မတွေ့ပါ:\n"
            f"{ZXEE_VOICE}"
        )

    print(
        "✅ Zxee Voice:",
        ZXEE_VOICE
    )

    # --------------------------------------------------------
    # Create custom voice directory
    # --------------------------------------------------------

    CUSTOM_VOICE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download model
    # --------------------------------------------------------

    await download_model()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    await load_model()

    # --------------------------------------------------------
    # Bot start
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("🤖 Bot Starting...")
    print("=" * 60)

    await dp.start_polling(
        bot
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n🛑 Bot stopped."
        )

    except Exception as e:

        print()
        print("❌ FATAL ERROR")
        print(e)

        raise
