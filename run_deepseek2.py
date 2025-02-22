from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import datetime

# โหลดโมเดลและ tokenizer
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
""" 
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
 """

# First create a prompt with the proper format for DeepSeek Chat models
def create_prompt(user_message):
    prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

# Function for text generation
def generate_response(user_message, max_length=512, temperature=0.7, top_p=0.9):
    prompt = create_prompt(user_message)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response and remove the prompt part
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_response = full_response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")
    
    return assistant_response.strip()

# Example usage
user_message = "Explain quantum computing in simple terms."
print(datetime.datetime.now())
response = generate_response(user_message)
print(response)
print(datetime.datetime.now())

