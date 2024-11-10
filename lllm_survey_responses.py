import csv
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random

def load_personas():
    print("Loading the dataset...")
    ds = load_dataset("proj-persona/PersonaHub", "persona")
    if 'train' not in ds:
        print("Error: The dataset does not have a 'train' split.")
        exit()
    selected_indices = random.sample(range(len(ds['train'])), 10)
    return [ds['train'][index]['persona'] for index in selected_indices]

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

survey_questions = [
    "After placing an order on an e-commerce website, how important are the following features to you? (Rate from 1-5, where 1 is not at all important and 5 is extremely important): Personal shopping assistant, Interactive voice assistant for order management, Automated/Smart cancellation and refund processes, Personalized product usage guides and tutorials, Predictive/automated reordering of items.",

    "What are your biggest concerns while using AI services? (Rate from 1-5, where 1 is not concerned at all and 5 is extremely concerned): Privacy (usage of your personal data), Accuracy (How correct the information is), Security (Protection of your data from hacks/leaks), Transparency (Knowing how decisions are made and what data is used), Bias (fairness in AI responses).",

    "If you want to manage your order (for example - you want to know the size of the blue t-shirt you bought last week), what action do you prefer? (Rate from 1-5, where 1 is extremely inconvenient and 5 is extremely convenient): Tapping buttons in the app to navigate to the 'Manage Order' screen, Using voice assistants such as Siri, Google Assistant, Alexa, etc., Typing into a chatbot.",

    "When using AI chatbots like ChatGPT or Gemini, what do you dislike? (Select all that apply): It's not a human and not 'intelligent' enough, It gives me generic solutions; I need specificity, I hate typing into chatbots, The chatbot takes too long to respond, The response is very wordy and not structured properly, I do not trust their response or their sources, I am not used to using chatbots, None, I like using chatbots.",

    "How do you prefer interacting with customer service? (Rate from 1-5, where 1 is not preferred at all and 5 is extremely preferred): Talking to a human customer service agent over a call, Talking to an automated bot/agent over a call, Chatting with the customer support chatbot, Raising an issue on the customer support webpage, Emailing help/support.",

    "Would you feel comfortable giving an AI assistant instructions to automatically cancel and replace a delayed order? Choose one: Yes, I'd be comfortable with that. / I'd be open to it, but I'd want to review the replacement choice before it's ordered. / I'd be concerned about the AI making a mistake. / I'd worry about trusting the AI with my order decisions. / No, I wouldn't be comfortable with an AI making these changes without my approval.",

    "How comfortable are you with an AI assistant proactively making online shopping decisions on your behalf? (For example - An AI assistant automatically ordering diapers if it thinks you are out of diapers OR AI automatically initiates a return if it thinks that you are not satisfied with the order) Choose one: Yes, I would find proactive support very helpful / Yes, but only for certain types of issues / I'm not sure / No, but I might prefer it for certain issues / No, I prefer to manage issues myself.",

    "If there was a personal assistant to help you track and manage your purchases, delivery issues, etc., what do you expect it to do?",

    "You bought a gift for a party assuming it would get delivered before the party. The delivery is now delayed. How would you feel about the following personalized actions? (Rate each from Very uncomfortable to Very comfortable): The order arrives late, and nothing is done. / The order is canceled automatically since it's no longer needed after the party. / The order is canceled, and a similar gift is re-ordered to arrive before the party. / The order is canceled, and a new gift is arranged to arrive at the party location when you do, so you don't have to carry it.",

    "Which approach would you find more efficient for managing multiple post-purchase actions simultaneously? Choose one: Traditional approach (separate pages for each action) / AI-driven approach (consolidated and conversational interaction) / A hybrid of traditional and AI driven approaches / I have no preference / I am unsure.",

    "Do you work or study in a technology-related field? Choose one: Yes, I am employed or studying in a technology-related field / No, I am not employed or studying in a technology-related field."
]

llm_models = [
    "distilbert/distilgpt2",
    "facebook/opt-350m"
]

personas = load_personas()

with open('llm_survey_responses.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Persona', 'Question', 'Response'])

    for model_name in llm_models:
        print(f"Loading model: {model_name}")
        try:
            tokenizer, model, device = load_model(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        for persona in tqdm(personas, desc=f"Processing {model_name}"):
            for question in survey_questions:
                prompt = f"""As an AI assistant, you are roleplaying as the following persona:{persona}\n\nPlease answer the following question based on this persona:{question}\n\nOnly give me the answer"""
                
                response = generate_response(model, tokenizer, prompt, device)
                writer.writerow([model_name, persona, question, response])
                csvfile.flush()

print("Survey responses have been saved to llm_survey_responses.csv")