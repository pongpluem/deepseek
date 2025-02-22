import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. โหลดโมเดลที่ fine-tune แล้ว
model_path = "./my-finetuned-deepseek"  # ตำแหน่งที่เก็บโมเดลที่ fine-tune แล้ว
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # ใช้ความแม่นยำครึ่งเพื่อประหยัดหน่วยความจำ
    device_map="auto",          # จัดสรรโมเดลไปยังอุปกรณ์ที่มีอยู่โดยอัตโนมัติ
)

# 2. สร้างฟังก์ชันสำหรับการสร้างข้อความ
def generate_text(prompt, max_new_tokens=512, temperature=0.7):
    # รูปแบบ prompt สำหรับ DeepSeek
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # สร้าง input tokens
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # ทำการสร้างข้อความ
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # แปลง tokens กลับเป็นข้อความ
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # ดึงเฉพาะส่วนคำตอบของ assistant ออกมา
    assistant_response = full_output.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")
    
    return assistant_response.strip()

# 3. ทดลองใช้งาน
response = generate_text("สวัสดี ช่วยอธิบายเรื่องการเรียนรู้ของเครื่องให้หน่อย")
print(response)