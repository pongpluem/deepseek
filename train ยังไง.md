# การ Fine-tune โมเดล DeepSeek LLM

หากคุณต้องการ fine-tune โมเดล DeepSeek-LLM-7B-Chat ด้วยข้อมูลของคุณเอง นี่คือขั้นตอนที่คุณต้องทำ:

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# 1. โหลดโมเดลและ tokenizer
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. เตรียมข้อมูลในรูปแบบที่เหมาะสม
def create_chat_prompt(example):
    # แปลงข้อมูลเป็นรูปแบบบทสนทนาของ DeepSeek
    prompt = f"<|im_start|>user\n{example['user_input']}<|im_end|>\n<|im_start|>assistant\n{example['assistant_response']}<|im_end|>"
    return {"text": prompt}

# ตัวอย่างข้อมูล (คุณต้องเตรียมข้อมูลของคุณเอง)
data = [
    {"user_input": "สวัสดี", "assistant_response": "สวัสดีครับ มีอะไรให้ช่วยไหมครับ?"},
    {"user_input": "อธิบาย AI คืออะไร", "assistant_response": "AI หรือปัญญาประดิษฐ์ คือระบบคอมพิวเตอร์ที่สามารถเรียนรู้และปรับตัวได้..."}
    # เพิ่มข้อมูลอื่นๆ
]

# สร้าง Dataset
dataset = Dataset.from_list(data)
processed_dataset = dataset.map(create_chat_prompt)

# 3. เตรียม tokenize ข้อมูล
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = processed_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text", "user_input", "assistant_response"]
)

# 4. กำหนดพารามิเตอร์สำหรับการเทรน
training_args = TrainingArguments(
    output_dir="./deepseek-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=True,  # ใช้ mixed precision training
    report_to="tensorboard",
    logging_steps=100,
)

# 5. กำหนด Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # ไม่ใช้ masked language modeling สำหรับ causal LM
)

# 6. เริ่มการเทรน
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# 7. บันทึกโมเดลที่เทรนเสร็จแล้ว
model.save_pretrained("./my-finetuned-deepseek")
tokenizer.save_pretrained("./my-finetuned-deepseek")
```

## เทคนิคเพิ่มเติมสำหรับการ Fine-tune

1. **ใช้ Parameter-Efficient Fine-Tuning (PEFT)**: เพื่อประหยัดทรัพยากร
   - เทคนิค LoRA (Low-Rank Adaptation) เป็นที่นิยมและมีประสิทธิภาพ
   - ลดการใช้ VRAM ลงมาก ช่วยให้สามารถ fine-tune บนฮาร์ดแวร์ที่มีข้อจำกัดได้

```python
from peft import LoraConfig, get_peft_model

# กำหนดค่า LoRA
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

# แปลงโมเดลเป็น PEFT model
model = get_peft_model(model, lora_config)
```

2. **ใช้ QLoRA**: หากคุณมีข้อจำกัดด้าน VRAM
   - ใช้ 4-bit quantization ร่วมกับ LoRA
   - ต้องใช้ bitsandbytes และ transformers เวอร์ชันล่าสุด

3. **การจัดเตรียมข้อมูล**:
   - คุณภาพข้อมูลสำคัญมากกว่าปริมาณ
   - ตรวจสอบให้แน่ใจว่าข้อมูลอยู่ในรูปแบบที่ถูกต้อง 
   - ใช้เทคนิค weight decay และ learning rate scheduler เพื่อป้องกัน overfitting

คุณต้องมี GPU ที่มี VRAM เพียงพอ หรือใช้บริการคลาวด์เช่น Google Colab Pro, AWS, หรือ Azure สำหรับการ fine-tune โมเดลขนาดใหญ่