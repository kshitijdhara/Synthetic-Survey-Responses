from datasets import load_dataset
import random
from openai import OpenAI
import csv

# Set up OpenAI client
client = OpenAI(api_key="")

def load_personas(num_personas=10):
    print("Loading the dataset...")
    ds = load_dataset("proj-persona/PersonaHub", "persona")
    if 'train' not in ds:
        print("Error: The dataset does not have a 'train' split.")
        exit()
    total_personas = len(ds['train'])
    selected_indices = random.sample(range(total_personas), num_personas)
    return [ds['train'][index]['persona'] for index in selected_indices]

def generate_response(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    personas = load_personas(3)
    survey_questions = [
        "After placing an order on an e-commerce website, how important are the following features to you? (Not at all important (1) , slightly important (2), moderately important (3), very important (4), extremely important (5)) Personal shopping assistant",
        "After placing an order on an e-commerce website, how important are the following features to you? (Not at all important (1) , slightly important (2), moderately important (3), very important (4), extremely important (5)) Interactive voice assistant for order management",
        "After placing an order on an e-commerce website, how important are the following features to you? (Not at all important (1) , slightly important (2), moderately important (3), very important (4), extremely important (5)) Automated/Smart cancellation and refund processes",
        "After placing an order on an e-commerce website, how important are the following features to you? (Not at all important (1) , slightly important (2), moderately important (3), very important (4), extremely important (5)) Personalized product usage guides and tutorials",
        "After placing an order on an e-commerce website, how important are the following features to you? (Not at all important (1) , slightly important (2), moderately important (3), very important (4), extremely important (5)) Predictive/automated reordering of items",
        "What are your biggest concerns while using AI services? (Not concerned at all (1), Somewhat concerned (2), Moderately concerned (3), Very concerned (4), Extremely concerned (5)) Privacy (usage of your personal data)",
        "What are your biggest concerns while using AI services? (Not concerned at all (1), Somewhat concerned (2), Moderately concerned (3), Very concerned (4), Extremely concerned (5)) Accuracy (How correct the information is)",
        "What are your biggest concerns while using AI services? (Not concerned at all (1), Somewhat concerned (2), Moderately concerned (3), Very concerned (4), Extremely concerned (5)) Security (Protection of your data from hacks/leaks)",
        "What are your biggest concerns while using AI services? (Not concerned at all (1), Somewhat concerned (2), Moderately concerned (3), Very concerned (4), Extremely concerned (5)) Transparency (Knowing how decisions are made and what data is used)",
        "What are your biggest concerns while using AI services? (Not concerned at all (1), Somewhat concerned (2), Moderately concerned (3), Very concerned (4), Extremely concerned (5)) Bias (fairness in AI responses)",
        "If you want to manage your order (for example - you want to know the size of the blue t-shirt you bought last week), what action do you prefer? (Extremely inconvenient (1), somewhat inconvenient (2), neither convenient or inconvenient (3), somewhat convenient (4), extremely convenient (5))Tapping buttons in the app to navigate to the “Manage Order” screen ",
        "If you want to manage your order (for example - you want to know the size of the blue t-shirt you bought last week), what action do you prefer? (Extremely inconvenient (1), somewhat inconvenient (2), neither convenient or inconvenient (3), somewhat convenient (4), extremely convenient (5))Using voice assistants such as Siri, Google Assistant, Alexa, etc. (“Hey Siri, what was the size of the blue shirt I got last week from Amazon?”)",
        "If you want to manage your order (for example - you want to know the size of the blue t-shirt you bought last week), what action do you prefer? (Extremely inconvenient (1), somewhat inconvenient (2), neither convenient or inconvenient (3), somewhat convenient (4), extremely convenient (5))Typing into a chatbot (Type - “I want to know the size of the blue shirt I got last week”)",
        "How do you prefer interacting with customer service? (Not preferred at all (1), Somewhat preferred (2), Moderately preferred (3), Very preferred (4), Extremely preferred (5)) Talking to a human customer service agent over a call",
        "How do you prefer interacting with customer service? (Not preferred at all (1), Somewhat preferred (2), Moderately preferred (3), Very preferred (4), Extremely preferred (5)) Talking to an automated bot /agent over a call",
        "How do you prefer interacting with customer service? (Not preferred at all (1), Somewhat preferred (2), Moderately preferred (3), Very preferred (4), Extremely preferred (5)) Chatting with the customer support chatbot",
        "How do you prefer interacting with customer service? (Not preferred at all (1), Somewhat preferred (2), Moderately preferred (3), Very preferred (4), Extremely preferred (5)) Raising an issue on the customer support webpage",
        "How do you prefer interacting with customer service? (Not preferred at all (1), Somewhat preferred (2), Moderately preferred (3), Very preferred (4), Extremely preferred (5)) Emailing help/support",
        "If there was a personal assistant to help you track and manage your purchases, delivery issues, etc, what do you expect it to do? Scenario: You bought a gift for a party assuming it would get delivered before the party. The delivery is now delayed. How would you feel about the following personalized actions? (Very uncomfortable, Uncomfortable, Neutral, Comfortable, Very comfortable) The order arrives late, and nothing is done.",
        "If there was a personal assistant to help you track and manage your purchases, delivery issues, etc, what do you expect it to do? Scenario: You bought a gift for a party assuming it would get delivered before the party. The delivery is now delayed. How would you feel about the following personalized actions? (Very uncomfortable, Uncomfortable, Neutral, Comfortable, Very comfortable) The order is canceled automatically since it’s no longer needed after the party.",
        "If there was a personal assistant to help you track and manage your purchases, delivery issues, etc, what do you expect it to do? Scenario: You bought a gift for a party assuming it would get delivered before the party. The delivery is now delayed. How would you feel about the following personalized actions? (Very uncomfortable, Uncomfortable, Neutral, Comfortable, Very comfortable) The order is canceled, and a similar gift is re-ordered to arrive before the party.",
        "If there was a personal assistant to help you track and manage your purchases, delivery issues, etc, what do you expect it to do? Scenario: You bought a gift for a party assuming it would get delivered before the party. The delivery is now delayed. How would you feel about the following personalized actions? (Very uncomfortable, Uncomfortable, Neutral, Comfortable, Very comfortable) The order is canceled, and a new gift is arranged to arrive at the party location when you do, so you don’t have to carry it.",
        "Would you feel comfortable giving an AI assistant instructions to automatically cancel and replace a delayed order? 1) Yes, I’d be comfortable with that. 2) I’d be open to it, but I’d want to review the replacement choice before it’s ordered. 3) I’d be concerned about the AI making a mistake (e.g., ordering the wrong item or something I won’t like). 4) I’d worry about trusting the AI with my order decisions (e.g., it might not understand my preferences). 5) No, I wouldn’t be comfortable with an AI making these changes without my approval.",
        "How comfortable are you with an AI assistant proactively making online shopping decisions on your behalf? (For example - An AI assistant automatically ordering diapers if it thinks you are out of diapers OR AI automatically initiates a return if it thinks that you are not satisfied with the order) 1) Yes, I would find proactive support very helpful 2) Yes, but only for certain types of issues 3) I'm not sure 4) No, but I might prefer it for certain issues 5) No, I prefer to manage issues myself.",
        "Which approach would you find more efficient for managing multiple post-purchase actions simultaneously? Scenario A (Traditional Approach): 'You want to check order statuses for multiple packages and request refunds for some. You have to visit the 'Order Status' page for each package and the 'Refunds' page for each refund request separately.' Scenario B (AI-Driven Approach): 'You want to check order statuses for multiple packages and request refunds for some. You tell the AI assistant, 'Show me the status of all my orders and process refunds for the ones I want to return.' The AI provides a consolidated summary and processes your refund requests in one interaction.' 1) Traditional approach (separate pages for each action) 2) AI-driven approach (consolidated and conversational interaction) 3) A hybrid of traditional and AI driven approaches 4) I have no preference 5) I am unsure",
        "Do you work or study in a technology-related field? 1) Yes, I am employed or studying in a technology-related field 2) No, I am not employed or studying in a technology-related field",
        "When using AI chatbots like ChatGPT or Gemini, what do you dislike? (Select all that apply) 1) It’s not a human and not “intelligent” enough 2) It gives me generic solutions; I need specificity 3) I hate typing into chatbots 4) The chatbot takes too long to respond 5) The response is very wordy and not structured properly 6) I do not trust their response or their sources 7) I am not used to using chatbots 8) None, I like using chatbots"
    ]

    with open('survey_responses.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['response_id', 'persona'] + [f'question_{i+1}' for i in range(len(survey_questions))]
        writer.writerow(header)

        for idx, persona in enumerate(personas, start=1):
            print(f"Processing persona {idx}: {persona[:50]}...")  # Print first 50 characters of persona
            
            responses = [idx, persona]  # Start with response_id and persona
            
            system_prompt = f"You are an AI assistant roleplaying as the following persona: {persona}. Answer questions as this persona would, only give me the rating number or answer option, nothing else."

            for question in survey_questions:
                response = generate_response(system_prompt, question)
                responses.append(response)

            writer.writerow(responses)

    print("Survey responses have been saved to survey_responses.csv")

if __name__ == "__main__":
    main()