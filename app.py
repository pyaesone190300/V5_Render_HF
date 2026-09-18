import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

import soundfile as sf
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, CallbackQuery, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
HF_REPO_ID = os.getenv("HF_REPO_ID", "pyaesone190300/VoxCPM2")
HF_TOKEN = os.getenv("HF_TOKEN") or None

BASE_DIR = Path("/app")
MODEL_DIR = BASE_DIR / "model"
ANNA_VOICE = BASE_DIR / "vvipvoice_v5_ref.wav"
ANNA_NAME = "Anna"
CUSTOM_VOICE_DIR = BASE_DIR / "custom_voices"
ACCESS_FILE = BASE_DIR / "access.json"

CUSTOM_VOICE_DIR.mkdir(parents=True, exist_ok=True)

# Kept for Anna's original reference setup/documentation.
PROMPT_TEXT = """ဒေါ်ခင်ခင်ညိုက နူးငယ်ရဲ့ အဒေါ်ပါ။လူကတော့ နည်းနည်းလေး sexyကျတယ်။ နူးငယ်ရဲ့ အဒေါ်ဆိုပေမယ့် အမေ့ညီမ အငယ်ဆုံးဆိုတော့ နူးငယ်နဲ့ကတော့ အသက်သိပ်မကွာပါဘူး။ နူးငယ်အတွက်က အဒေါ်ဆိုလည်းဟုတ်၊ သူငယ်ချင်းဆိုလည်းဟုတ်ဆိုတော့ တူမနှစ်ယောက်က ပြောမနာဆိုမနာပါ။"""

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable မတွေ့ပါ။")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
model = None
tts_lock = asyncio.Lock()
access_lock = asyncio.Lock()
user_voice = {}

def default_access():
    return {"approved": [], "pending": [], "rejected": []}

def load_access():
    if not ACCESS_FILE.exists():
        return default_access()
    try:
        with open(ACCESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("approved", [])
        data.setdefault("pending", [])
        data.setdefault("rejected", [])
        return data
    except Exception as e:
        print(f"⚠️ access.json load error: {e}")
        return default_access()

def save_access(data):
    tmp = ACCESS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(ACCESS_FILE)

def is_owner(user_id: int):
    return OWNER_ID != 0 and user_id == OWNER_ID

def is_approved(user_id: int):
    return is_owner(user_id) or str(user_id) in load_access()["approved"]

def is_pending(user_id: int):
    return str(user_id) in load_access()["pending"]

def get_custom_voice_path(user_id: int):
    return CUSTOM_VOICE_DIR / f"{user_id}.wav"

def has_custom_voice(user_id: int):
    return get_custom_voice_path(user_id).exists()

# FIX: this function must exist before main() calls it.
def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (MODEL_DIR / "model.safetensors").exists():
        print("✅ Hugging Face model already exists")
        return
    print(f"⬇️ Downloading {HF_REPO_ID} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
    )
    print("✅ Hugging Face model ready")

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
        print(f"Device: {next(model.parameters()).device}")
    except (StopIteration, AttributeError):
        print("Device: cpu")
    try:
        print(f"Sample rate: {model.tts_model.sample_rate}")
    except AttributeError:
        print("⚠️ Sample rate fallback: 16000")

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

def main_menu(current="Anna"):
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(
                text=f"👩 Anna{'  ✓' if current == 'Anna' else ''}",
                callback_data="voice:anna",
            ),
            InlineKeyboardButton(
                text=f"🎤 Custom Voice{'  ✓' if current == 'Custom Voice' else ''}",
                callback_data="voice:custom",
            ),
        ],
        [
            InlineKeyboardButton(
                text="ℹ️ How to Use",
                callback_data="menu:help",
            )
        ],
    ])

def back_menu():
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="⬅️ Back", callback_data="menu:main")
    ]])

async def request_access(message: Message):
    user = message.from_user
    uid = str(user.id)
    if is_owner(user.id) or is_approved(user.id):
        return True

    async with access_lock:
        data = load_access()
        if uid not in data["pending"]:
            data["pending"].append(uid)
        if uid in data["rejected"]:
            data["rejected"].remove(uid)
        save_access(data)

    username = f"@{user.username}" if user.username else "No username"
    name = user.first_name or "Unknown"
    kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Approve", callback_data=f"access:approve:{user.id}"),
        InlineKeyboardButton(text="❌ Reject", callback_data=f"access:reject:{user.id}"),
    ]])
    try:
        await bot.send_message(
            OWNER_ID,
            "🔐 <b>New Access Request</b>\n\n"
            f"👤 Name: <b>{name}</b>\n"
            f"🔹 Username: {username}\n"
            f"🆔 User ID: <code>{user.id}</code>\n\n"
            "ဒီ User ကို Bot အသုံးပြုခွင့်ပေးမလား?",
            parse_mode="HTML",
            reply_markup=kb,
        )
    except Exception as e:
        print(f"❌ Cannot contact owner: {e}")
    return False

@dp.message(CommandStart())
async def start_handler(message: Message):
    uid = message.from_user.id
    if not is_approved(uid):
        if not await request_access(message):
            await message.answer(
                "🔐 <b>Access Pending</b>\n\n"
                "ဒီ Bot ကို အသုံးပြုရန် Owner ရဲ့ ခွင့်ပြုချက်လိုအပ်ပါတယ်။\n\n"
                "⏳ Owner ဆီကို Access Request ပို့ပြီးပါပြီ။",
                parse_mode="HTML",
            )
        return

    user_voice[uid] = "anna"
    await message.answer(
        "🎙️ <b>VoxCPM2 Myanmar Voice Bot</b>\n\n"
        "👋 မင်္ဂလာပါ။\n\n"
        "ဒီ Bot မှာ မြန်မာစာကို Voice အဖြစ် ပြောင်းနိုင်ပါတယ်။\n\n"
        "🎙️ <b>Current Voice: Anna</b>\n\n"
        "အသုံးပြုလိုတဲ့ Voice ကို အောက်က Menu ကနေ ရွေးပါ။",
        parse_mode="HTML",
        reply_markup=main_menu("Anna"),
    )

@dp.message(Command("pending"))
async def pending_handler(message: Message):
    if not is_owner(message.from_user.id):
        return
    pending = load_access()["pending"]
    if not pending:
        await message.answer("📭 <b>Pending Request မရှိပါ။</b>", parse_mode="HTML")
        return
    await message.answer(
        "🔐 <b>Pending Access Requests</b>\n\n" +
        "".join(f"🆔 <code>{x}</code>\n" for x in pending),
        parse_mode="HTML",
    )

@dp.callback_query(F.data.startswith("access:"))
async def access_callback(callback: CallbackQuery):
    if not is_owner(callback.from_user.id):
        await callback.answer("❌ Owner only", show_alert=True)
        return
    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer("Invalid request", show_alert=True)
        return
    action = parts[1]
    try:
        target_id = int(parts[2])
    except ValueError:
        await callback.answer("Invalid User ID", show_alert=True)
        return

    async with access_lock:
        data = load_access()
        target = str(target_id)

        if action == "approve":
            if target not in data["approved"]:
                data["approved"].append(target)
            if target in data["pending"]:
                data["pending"].remove(target)
            if target in data["rejected"]:
                data["rejected"].remove(target)
            save_access(data)
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n✅ <b>APPROVED</b>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
            try:
                await bot.send_message(
                    target_id,
                    "🎉 <b>Access Approved!</b>\n\n"
                    "Owner က သင့်ကို Bot အသုံးပြုခွင့် ပေးလိုက်ပါပြီ။",
                    parse_mode="HTML",
                    reply_markup=main_menu("Anna"),
                )
            except Exception as e:
                print(f"⚠️ Cannot notify user: {e}")
            await callback.answer("✅ User approved")
            return

        if action == "reject":
            if target in data["pending"]:
                data["pending"].remove(target)
            if target not in data["rejected"]:
                data["rejected"].append(target)
            save_access(data)
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n❌ <b>REJECTED</b>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
            try:
                await bot.send_message(
                    target_id,
                    "❌ <b>Access မရသေးပါ။</b>\n\n"
                    "Owner က ဒီအချိန်မှာ Bot အသုံးပြုခွင့် မပေးထားသေးပါ။",
                    parse_mode="HTML",
                )
            except Exception as e:
                print(f"⚠️ Cannot notify user: {e}")
            await callback.answer("❌ User rejected")
            return

    await callback.answer("Unknown action")

@dp.callback_query(F.data == "menu:main")
async def menu_main(callback: CallbackQuery):
    uid = callback.from_user.id
    if not is_approved(uid):
        await callback.answer("🔐 Access မရသေးပါ", show_alert=True)
        return
    current = "Anna" if user_voice.get(uid, "anna") == "anna" else "Custom Voice"
    await callback.answer()
    await callback.message.edit_text(
        "🎙️ <b>VoxCPM2 Myanmar Voice Bot</b>\n\n"
        f"🎙️ <b>Current Voice:</b> {current}\n\n"
        "အသုံးပြုလိုတဲ့ Voice ကိုရွေးပါ။",
        parse_mode="HTML",
        reply_markup=main_menu(current),
    )

@dp.callback_query(F.data == "menu:help")
async def menu_help(callback: CallbackQuery):
    if not is_approved(callback.from_user.id):
        await callback.answer("🔐 Access မရသေးပါ", show_alert=True)
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
        "→ အသံအဖြစ် ပြန်ပေးပါမယ်။",
        parse_mode="HTML",
        reply_markup=back_menu(),
    )

@dp.callback_query(F.data == "voice:anna")
async def select_anna(callback: CallbackQuery):
    uid = callback.from_user.id
    if not is_approved(uid):
        await callback.answer("🔐 Access မရသေးပါ", show_alert=True)
        return
    user_voice[uid] = "anna"
    await callback.answer("👩 Anna ကို ရွေးပြီးပါပြီ")
    await callback.message.edit_text(
        "👩 <b>Anna</b>\n\n"
        "Original V5 Voice ကို အသုံးပြုနေပါတယ်။\n\n"
        "အခု မြန်မာစာပို့လိုက်ပါ။",
        parse_mode="HTML",
        reply_markup=main_menu("Anna"),
    )

@dp.callback_query(F.data == "voice:custom")
async def select_custom(callback: CallbackQuery):
    uid = callback.from_user.id
    if not is_approved(uid):
        await callback.answer("🔐 Access မရသေးပါ", show_alert=True)
        return

    if has_custom_voice(uid):
        user_voice[uid] = "custom"
        await callback.answer("🎤 Custom Voice ကို ရွေးပြီးပါပြီ")
        text = (
            "🎤 <b>Custom Voice</b>\n\n"
            "လက်ရှိသိမ်းထားတဲ့ Custom Voice ကို အသုံးပြုနေပါတယ်။\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။\n\n"
            "Voice အသစ်ပြောင်းချင်ရင် Audio / Voice file အသစ်ပို့ပါ။"
        )
    else:
        await callback.answer()
        text = (
            "🎤 <b>Custom Voice</b>\n\n"
            "ကိုယ်အသုံးပြုလိုတဲ့ အသံဖိုင်ကို ဒီနေရာမှာ ပို့ပေးပါ။\n\n"
            "📌 <b>Supported formats:</b>\n"
            "• Telegram Voice\n• WAV\n• MP3\n• M4A\n• OGG\n• FLAC\n\n"
            "ဖိုင်ရောက်လာတာနဲ့ Custom Voice အဖြစ် သိမ်းပေးပါမယ်။"
        )
    await callback.message.edit_text(
        text, parse_mode="HTML", reply_markup=back_menu()
    )

async def save_custom_voice(message: Message, user_id: int):
    custom_path = get_custom_voice_path(user_id)
    temp_original = None
    temp_converted = None
    try:
        if message.voice:
            file_id, suffix = message.voice.file_id, ".ogg"
        elif message.audio:
            file_id = message.audio.file_id
            suffix = Path(message.audio.file_name or ".mp3").suffix or ".mp3"
        elif message.document:
            file_id = message.document.file_id
            suffix = Path(message.document.file_name or ".wav").suffix or ".wav"
        else:
            return False

        tg_file = await bot.get_file(file_id)
        temp_original = Path(tempfile.gettempdir()) / f"voice_{user_id}{suffix}"
        await bot.download_file(tg_file.file_path, destination=str(temp_original))

        temp_converted = Path(tempfile.gettempdir()) / f"voice_{user_id}_converted.wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(temp_original),
            "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
            str(temp_converted),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()

        if rc == 0 and temp_converted.exists():
            shutil.copy2(temp_converted, custom_path)
        else:
            shutil.copy2(temp_original, custom_path)

        sf.read(str(custom_path), always_2d=False)
        user_voice[user_id] = "custom"
        print(f"✅ Custom Voice saved: user={user_id}")
        return True
    except Exception as e:
        print(f"❌ Custom Voice Error: {e}")
        if custom_path.exists():
            try:
                custom_path.unlink()
            except Exception:
                pass
        return False
    finally:
        for p in (temp_original, temp_converted):
            if p and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

@dp.message(F.voice | F.audio | F.document)
async def voice_upload_handler(message: Message):
    uid = message.from_user.id
    if not is_approved(uid):
        await message.answer(
            "🔐 <b>Access မရသေးပါ။</b>\n\n"
            "Owner ခွင့်ပြုချက်ရပြီးမှ Bot ကို အသုံးပြုနိုင်ပါတယ်။",
            parse_mode="HTML",
        )
        return

    if message.document:
        filename = (message.document.file_name or "").lower()
        if not filename.endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac")):
            await message.answer("❌ Audio file မဟုတ်ပါဘူး။ WAV / MP3 / M4A / OGG / FLAC တစ်ခုခု ပို့ပါ။")
            return

    status = await message.answer(
        "🎤 <b>Custom Voice</b>\n\n⏳ Voice ကို သိမ်းနေပါတယ်...",
        parse_mode="HTML",
    )
    ok = await save_custom_voice(message, uid)
    if ok:
        await status.edit_text(
            "✅ <b>Custom Voice သိမ်းပြီးပါပြီ။</b>\n\n"
            "🎤 <b>Current Voice:</b> Custom Voice\n\n"
            "အခု မြန်မာစာပို့လိုက်ပါ။",
            parse_mode="HTML",
            reply_markup=main_menu("Custom Voice"),
        )
    else:
        await status.edit_text(
            "❌ <b>Custom Voice သိမ်းမရပါ။</b>\n\n"
            "WAV / MP3 / M4A / Telegram Voice နဲ့ ပြန်ပို့ကြည့်ပါ။",
            parse_mode="HTML",
            reply_markup=main_menu("Anna"),
        )

def get_user_voice(user_id: int):
    selected = user_voice.get(user_id, "anna")
    if selected == "custom":
        p = get_custom_voice_path(user_id)
        if p.exists():
            return p, "Custom Voice"
        user_voice[user_id] = "anna"
    return ANNA_VOICE, "Anna"

async def generate_voice(text: str, reference_wav: Path, out: Path, voice_name: str):
    async with tts_lock:
        print(f"🎙️ Generating: {voice_name}")
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
        sf.write(str(out), wav, get_sample_rate())
        print(f"✅ Generated sample_rate={get_sample_rate()}")

@dp.message(F.text)
async def text_handler(message: Message):
    uid = message.from_user.id
    if not is_approved(uid):
        await message.answer(
            "🔐 <b>Access မရသေးပါ။</b>\n\n"
            "Owner ခွင့်ပြုချက်ရပြီးမှ Bot ကို အသုံးပြုနိုင်ပါတယ်။",
            parse_mode="HTML",
        )
        return

    text = message.text.strip()
    if not text:
        return
    if len(text) > 1000:
        await message.answer("❌ စာသားက 1000 characters ထက်မကျော်ရပါ။")
        return

    ref, voice_name = get_user_voice(uid)
    if not ref.exists():
        await message.answer("❌ Voice file မတွေ့ပါဘူး။ Anna voice file ကို စစ်ပေးပါ။")
        return

    status = await message.answer(
        f"🎙️ <b>{voice_name}</b>\n\n⏳ မြန်မာအသံထုတ်နေပါတယ်...",
        parse_mode="HTML",
    )
    output = Path(tempfile.gettempdir()) / f"tts_{uid}_{message.message_id}.wav"
    try:
        await generate_voice(text, ref, output, voice_name)
        await status.edit_text(
            f"🎙️ <b>{voice_name}</b>\n\n✅ အသံထွက်ပြီးပါပြီ။",
            parse_mode="HTML",
        )
        await message.answer_audio(
            audio=FSInputFile(str(output)),
            title=f"{voice_name} Voice",
        )
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        await status.edit_text(
            "❌ TTS Error ဖြစ်သွားပါတယ်။\n\n"
            f"<code>{str(e)[:500]}</code>",
            parse_mode="HTML",
        )
    finally:
        if output.exists():
            try:
                output.unlink()
            except Exception:
                pass

async def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN မတွေ့ပါ။")
    if OWNER_ID == 0:
        raise RuntimeError("OWNER_ID မထည့်ရသေးပါ။")
    if not ANNA_VOICE.exists():
        raise FileNotFoundError(f"Anna voice file မတွေ့ပါ: {ANNA_VOICE}")

    print("🚀 Starting VoxCPM2 Telegram Bot...")
    print(f"👑 Owner ID: {OWNER_ID}")

    await asyncio.to_thread(download_model)
    await asyncio.to_thread(load_model)

    print("✅ VoxCPM2 Telegram Bot READY")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
