from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, system_prompt, user_prompt, max_length=200):
    chat_template = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>human\n{human_message}<|im_end|>\n<|im_start|>assistant"
    
    prompt = chat_template.format(
        system_message=system_prompt,
        human_message=user_prompt
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    assistant_response = response.split("<|im_start|>assistant")[-1].strip()
    
    return assistant_response

def main():
    model_name = "NousResearch/Hermes-3-Llama-3.1-8B"  # You can change this to other models like "NousResearch/Hermes-3-Llama-3.1-8B"
    model, tokenizer = load_model_and_tokenizer(model_name)

    print("Welcome to the Persona-based Question Answering System!")
    persona = input("Please enter a persona description: ")

    system_prompt = f"You are an AI assistant roleplaying as the following persona: {persona}. Answer questions as this persona would."

    print(f"\nGreat! I'll now answer questions as if I were: {persona}")

    while True:
        question = input("\nAsk a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        response = generate_response(model, tokenizer, system_prompt, question)

        print(f"Answer: {response}")

if __name__ == "__main__":
    main()