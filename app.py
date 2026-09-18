import asyncio, os, tempfile, time
from pathlib import Path
import soundfile as sf
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, Message
from huggingface_hub import snapshot_download
from voxcpm import VoxCPM

BOT_TOKEN=os.getenv("BOT_TOKEN")
HF_REPO_ID=os.getenv("HF_REPO_ID","pyaesone190300/VoxCPM2")
HF_TOKEN=os.getenv("HF_TOKEN") or None
BASE_DIR=Path("/app")
MODEL_DIR=BASE_DIR/"model"
REFERENCE_WAV=BASE_DIR/"vvipvoice_v5_ref.wav"

PROMPT_TEXT="""ဒေါ်ခင်ခင်ညိုက နူးငယ်ရဲ့ အဒေါ်ပါ။လူကတော့ နည်းနည်းလေး sexyကျတယ်။ နူးငယ်ရဲ့ အဒေါ်ဆိုပေမယ့် အမေ့ညီမ အငယ်ဆုံးဆိုတော့ နူးငယ်နဲ့ကတော့ အသက်သိပ်မကွာပါဘူး။ နူးငယ်အတွက်က အဒေါ်ဆိုလည်းဟုတ်၊ သူငယ်ချင်းဆိုလည်းဟုတ်ဆိုတော့ တူမနှစ်ယောက်က ပြောမနာဆိုမနာပါ။"""

bot=Bot(token=BOT_TOKEN)
dp=Dispatcher()
tts_lock=asyncio.Lock()
model=None

def download_model():
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    print(f"⬇️ Downloading {HF_REPO_ID} ...")
    snapshot_download(repo_id=HF_REPO_ID,local_dir=str(MODEL_DIR),token=HF_TOKEN)
    print("✅ Hugging Face model ready")

def load_model():
    global model
    model=VoxCPM.from_pretrained(str(MODEL_DIR),load_denoiser=False,
        local_files_only=True,optimize=False,device="cpu")
    print("✅ VoxCPM2 loaded")
    print("Device:",model.device)
    print("Sample rate:",model.tts_model.sample_rate)

async def generate_v5(text,out):
    async with tts_lock:
        wav=await asyncio.to_thread(model.generate,text=text,
            prompt_wav_path=str(REFERENCE_WAV),prompt_text=PROMPT_TEXT,
            reference_wav_path=str(REFERENCE_WAV),
            inference_timesteps=10,cfg_value=2.0,
            retry_badcase=False,max_len=1000)
        sf.write(out,wav,model.tts_model.sample_rate)

@dp.message(CommandStart())
async def start(message:Message):
    await message.answer("🎙️ <b>V5 Myanmar Voice Bot</b>\n\nမြန်မာစာပို့ပါ။\nV5 Voice နဲ့ အသံထုတ်ပေးပါမယ်။\n\n⏳ CPU server ဖြစ်လို့ အချိန်ကြာနိုင်ပါတယ်။",parse_mode="HTML")

@dp.message(F.text)
async def text_handler(message:Message):
    text=message.text.strip()
    if not text:return
    if len(text)>1000:
        await message.answer("❌ စာသား 1000 characters ထက် မကျော်ပါနဲ့။")
        return
    status=await message.answer("⏳ <b>V5 အသံထုတ်နေပါတယ်...</b>",parse_mode="HTML")
    out=None
    try:
        fd,out=tempfile.mkstemp(suffix=".wav",prefix="v5_"); os.close(fd)
        t=time.time()
        await generate_v5(text,out)
        elapsed=time.time()-t
        await message.answer_audio(FSInputFile(out),caption=f"🎙️ V5 Myanmar Voice\n⏱️ {elapsed:.1f}s")
        await status.delete()
    except Exception as e:
        await status.edit_text("❌ Error ဖြစ်ပါတယ်။\n\n<code>"+str(e)[:1500]+"</code>",parse_mode="HTML")
    finally:
        if out:
            try: os.remove(out)
            except OSError: pass

async def main():
    if not BOT_TOKEN: raise RuntimeError("BOT_TOKEN မတွေ့ပါ။")
    if not REFERENCE_WAV.exists() or REFERENCE_WAV.stat().st_size==0:
        raise FileNotFoundError(f"V5 reference file မတွေ့ပါ: {REFERENCE_WAV}")
    download_model()
    load_model()
    print("🤖 Telegram V5 CPU Bot starting...")
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
