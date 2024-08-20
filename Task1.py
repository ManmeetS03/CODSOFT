def task1_chatbot(user_input):
    user_input = user_input.lower()

    responses = {
        "hi": "Hello! Welcome to the Task 1",
        "hello": "Hello! Welcome to the Task 1",
        "hey": "Hello! Welcome to the Task 1",
        "what's your name": "I’m ChatBot. How are you doing today?",
        "name": "I’m ChatBot. How are you doing today?",
        "how are you": "I'm doing great! How about you?",
        "good": "I am glad to hear that!",
        "great": "Oh I am happy for you!",
        "bad": "I am sorry to hear that!",
        "idk": "It's okay.",
        "thank you": "You're welcome!",
        "thanks": "You're welcome!",
        "bye": "Bye! Have a great day!",
        "goodbye": "Goodbye! Have a great day!",
        "cya": "Cya! Have a great day!",
    }
    default_response = "I couldn't understand. Can u rephrase that please?"    
    for keyword, response in responses.items():
        if keyword in user_input:
            return response
        
    return default_response

def chat():
    print("ChatBot: Hey There! I'm here to chat with you.")
    
    while True:
        user_input = input("You: ")
        
        response = task1_chatbot(user_input)
        
        print("ChatBot:", response)
        
        if "bye" in user_input.lower() or "goodbye" in user_input.lower() or "cya" in user_input.lower():
            break

if __name__ == "__main__":
    chat()
