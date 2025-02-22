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

