from openai import OpenAI
import os

# Set up OpenAI client
client = OpenAI(api_key="")

def generate_response(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # You can change this to "gpt-4" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return "I'm sorry, I couldn't generate a response."

def main():
    print("Welcome to the Persona-based Question Answering System!")
    persona = input("Please enter a persona description: ")

    system_prompt = f"You are an AI assistant roleplaying as the following persona: {persona}. Answer questions as this persona would but the answer should be to the point"

    print(f"\nGreat! I'll now answer questions as if I were: {persona}")

    while True:
        question = input("\nAsk a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        response = generate_response(system_prompt, question)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main()