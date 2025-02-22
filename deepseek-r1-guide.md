# คู่มือการติดตั้งและรัน Deepseek R1 บน PC

Deepseek R1 เป็นโมเดลภาษาขนาดใหญ่ (LLM) ที่พัฒนาโดย Deepseek ซึ่งสามารถนำมารันบนเครื่องคอมพิวเตอร์ส่วนบุคคลได้ ต่อไปนี้เป็นขั้นตอนการติดตั้งและใช้งาน

## ความต้องการของระบบ

- **CPU**: คำแนะนำขั้นต่ำคือ CPU รุ่นใหม่ที่สนับสนุน AVX2
- **RAM**: อย่างน้อย 16GB (แนะนำ 32GB หรือมากกว่า)
- **GPU**: NVIDIA GPU ที่มี VRAM อย่างน้อย 8GB (แนะนำ 16GB หรือมากกว่าสำหรับโมเดลขนาดใหญ่)
- **พื้นที่ดิสก์**: อย่างน้อย 20GB สำหรับโมเดลและข้อมูลที่เกี่ยวข้อง
- **ระบบปฏิบัติการ**: Windows 10/11, Linux, หรือ macOS

## วิธีการติดตั้ง

### วิธีที่ 1: ใช้ LM Studio (ง่ายที่สุดสำหรับผู้เริ่มต้น)

1. **ดาวน์โหลด LM Studio**
   - ไปที่ [lmstudio.ai](https://lmstudio.ai/) และดาวน์โหลดเวอร์ชันล่าสุด
   - ติดตั้งแอปพลิเคชันตามคำแนะนำ

2. **ดาวน์โหลดโมเดล Deepseek R1**
   - เปิด LM Studio
   - ไปที่แท็บ "Models"
   - ค้นหา "Deepseek R1"
   - เลือกขนาดโมเดลที่เหมาะกับฮาร์ดแวร์ของคุณ (7B สำหรับ GPU ขนาดเล็ก, 34B สำหรับ GPU ที่มี VRAM มากกว่า)
   - คลิกดาวน์โหลด

3. **รันโมเดล**
   - หลังจากดาวน์โหลดเสร็จ คลิกที่โมเดลเพื่อโหลดเข้า LM Studio
   - ปรับการตั้งค่าตามความต้องการ (temperature, top_p, เป็นต้น)
   - เริ่มแชทกับโมเดลได้เลย

### วิธีที่ 2: ใช้ Command Line ผ่าน llama.cpp

1. **ติดตั้ง git และ C++ compiler**
   - สำหรับ Windows ติดตั้ง [Git for Windows](https://gitforwindows.org/) และ [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
   - สำหรับ Linux: `sudo apt install git build-essential cmake`

2. **โคลน llama.cpp repository**
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   ```

3. **คอมไพล์ llama.cpp**
   - Windows:
   ```bash
   cmake -B build
   cmake --build build --config Release
   ```
   - Linux/macOS:
   ```bash
   make
   ```

4. **ดาวน์โหลดโมเดล Deepseek R1**
   - ไปที่ [Hugging Face - Deepseek R1](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
   - ดาวน์โหลดไฟล์โมเดลในรูปแบบ GGUF (เช่น deepseek-llm-7b-chat.Q4_K_M.gguf)
   - บันทึกไฟล์ไว้ในโฟลเดอร์ llama.cpp/models/

5. **รันโมเดล**
   ```bash
   ./build/bin/main -m models/deepseek-llm-7b-chat.Q4_K_M.gguf -n 2048 -i -p "เริ่มต้นบทสนทนา:"
   ```

### วิธีที่ 3: ผ่าน Python API (สำหรับนักพัฒนา)

1. **ติดตั้ง Python และ pip**
   - ดาวน์โหลดและติดตั้ง [Python 3.10+](https://www.python.org/downloads/)

2. **สร้างสภาพแวดล้อมเสมือน (แนะนำ)**
   ```bash
   python -m venv deepseek-env
   # Windows
   deepseek-env\Scripts\activate
   # Linux/macOS
   source deepseek-env/bin/activate
   ```

3. **ติดตั้ง library ที่จำเป็น**
   ```bash
   pip install torch transformers huggingface_hub
   ```

4. **ดาวน์โหลดและรันโมเดล**
   - สร้างไฟล์ Python ใหม่ (เช่น run_deepseek.py) ด้วยโค้ดนี้:

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch

   # โหลดโมเดลและ tokenizer
   model_name = "deepseek-ai/deepseek-llm-7b-chat"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16,
       device_map="auto",
   )

   # ฟังก์ชันสำหรับการแชท
   def chat_with_model(prompt):
       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
       with torch.no_grad():
           outputs = model.generate(
               inputs["input_ids"],
               max_new_tokens=512,
               temperature=0.7,
               top_p=0.9,
           )
       response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
       return response

   # ตัวอย่างการใช้งาน
   prompt = "คุณช่วยอธิบายว่า Deepseek R1 คืออะไร?"
   print(chat_with_model(prompt))

   # โหมด Interactive chat
   while True:
       user_input = input("\nคุณ: ")
       if user_input.lower() in ["exit", "quit", "bye"]:
           break
       print("\nDeepseek R1:", chat_with_model(user_input))
   ```

5. **รันสคริปต์**
   ```bash
   python run_deepseek.py
   ```

## การแก้ไขปัญหาทั่วไป

1. **CUDA out of memory**
   - ลองใช้โมเดลที่มีขนาดเล็กลง (เช่น 7B แทน 34B)
   - ลดค่า batch size หรือ context length
   - ใช้การ quantize โมเดล (เช่น 4-bit quantization)

2. **โมเดลทำงานช้าเกินไป**
   - เพิ่มการ quantize (ลดความแม่นยำลงเพื่อเพิ่มความเร็ว)
   - ใช้ GPU ที่มีประสิทธิภาพมากขึ้น
   - ลดค่า max_new_tokens เพื่อให้การตอบกลับสั้นลง

3. **โมเดลไม่โหลด**
   - ตรวจสอบว่าคุณมีพื้นที่ดิสก์และ RAM เพียงพอ
   - ตรวจสอบว่า CUDA/GPU driver เป็นเวอร์ชันล่าสุด
   - ตรวจสอบว่าคุณใช้ไฟล์โมเดลที่ถูกต้อง

## เครื่องมือที่น่าสนใจอื่นๆ สำหรับรัน Deepseek R1

- **Ollama**: [ollama.ai](https://ollama.ai/) - แพลตฟอร์มง่ายๆ สำหรับการรัน LLM แบบโอเพนซอร์ส
- **Text Generation WebUI**: [github.com/oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) - UI แบบเว็บที่เต็มไปด้วยคุณสมบัติ
- **koboldcpp**: [github.com/LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) - เวอร์ชั่นที่ปรับแต่งของ llama.cpp เพื่อจุดประสงค์เฉพาะ
