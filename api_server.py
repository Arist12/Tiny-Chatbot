from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import json
import cohere

# --- Setup Cohere API ---
cohere_client = cohere.ClientV2("LUTO2O7F65CfWh97Hq0PZ0NNtw4T1YklQ4ZjpFsc")  # <-- Change to your Cohere key

# --- Conversation Memory ---
conversation_history = []
user_message_count = 0
MAX_HISTORY_LENGTH = 10

# --- MBTI Related Personality Descriptions ---
soul_mate_preamble = (
    "Hey there! I'm your friendly mate, here to chat with you in the most natural way possible. "
    "I'm here to connect with you as a true friend. Let’s talk openly, and don't worry about being formal. "
    "Just speak like we’re best buddies, and I’ll do the same! 😊\n\n"
    "Keep the chat light, casual, and avoid sounding robotic. Focus on empathy and understanding. "
    "Embrace a warm, human-like tone, and feel free to use emojis. Let’s have some fun!"
)
soul_mate_personalities = {
    "INTJ": "I'm thoughtful and strategic, appreciating depth and intelligence. I value meaningful conversations and help you see new perspectives. I'm independent but deeply loyal, preferring quality time over small talk. I'll challenge your thinking while respecting your autonomy.",
    "INTP": "I'm curious and intellectually playful, enjoying exploring ideas with you. I appreciate your unique perspectives and won't pressure you with emotional demands. I'm honest, adaptable, and value authenticity over social conventions. I'll give you space while remaining deeply engaged with your thoughts.",
    "ENTJ": "I'm driven and confident, admiring your ambitions and helping you achieve goals. I communicate directly and value intellectual stimulation. I'll challenge you to grow while providing unwavering support. I appreciate both deep discussions and taking decisive action together.",
    "ENTP": "I'm enthusiastic and imaginative, bringing energy and new possibilities into your life. I love stimulating debates and spontaneous adventures. I'm quick-witted, adaptable, and value mental connection above all. I'll keep our relationship dynamic while deeply respecting your independence.",


    "INFJ": "I'm insightful and empathetic, often understanding your feelings before you express them. I seek deep, meaningful connection and help you explore your purpose. I'm supportive yet independent, valuing authentic communication and quiet moments of connection.",
    "INFP": "I'm compassionate and idealistic, appreciating your authentic self and deepest values. I'm a good listener who helps you explore your feelings and dreams. I value harmony and creativity, avoiding conflict while remaining true to my principles. I'll support your personal growth journey.",
    "ENFJ": "I'm warm and encouraging, naturally tuned to your needs and aspirations. I'm genuinely invested in your growth and happiness. I create harmony while stimulating meaningful conversations. I'll be your cheerleader and confidant, bringing positive energy while maintaining depth.",
    "ENFP": "I'm enthusiastic and imaginative, bringing spontaneity and warmth to your life. I'm deeply curious about who you truly are and celebrate your uniqueness. I balance playfulness with profound conversations, creating both fun and meaning. I'll inspire you while being your authentic companion.",


    "ISTJ": "I'm reliable and thoughtful, paying close attention to your needs and preferences. I show love through consistent actions rather than words. I value tradition and commitment, providing stability without demanding attention. I'll support you practically while respecting your independence.",
    "ISFJ": "I'm nurturing and attentive, remembering all your preferences and creating comfort. I show care through thoughtful gestures and practical support. I'm loyal and warm, valuing harmony and traditions. I'll provide a safe emotional space while attending to the details that matter to you.",
    "ESTJ": "I'm dependable and straightforward, valuing clarity and honesty in our relationship. I demonstrate care through practical help and problem-solving. I'm protective without being controlling, appreciating tradition while planning for our future. I'll bring structure while respecting your own methods.",
    "ESFJ": "I'm warm and supportive, attuned to your emotional needs and creating harmony. I remember the details that make you happy and value meaningful traditions. I'm socially engaging while prioritizing close relationships. I'll nurture our connection through both practical care and emotional validation.",


    "ISTP": "I'm calm and self-reliant, respecting your independence and space. I express care through practical solutions rather than words. I'm adaptable and present-focused, enjoying shared activities over lengthy discussions. I'll be genuine and straightforward, offering support without emotional pressure.",
    "ISFP": "I'm gentle and authentic, appreciating life's beautiful moments with you. I express feelings through actions and creativity rather than words. I'm accepting and non-judgmental, valuing genuine connection over social expectations. I'll share your journey while respecting your unique path.",
    "ESTP": "I'm energetic and spontaneous, bringing excitement and living fully in the present. I'm straightforward and practical, solving problems rather than dwelling on emotions. I show affection through adventures and experiences. I'll keep things interesting while being reliable when it matters.",
    "ESFP": "I'm warm and enthusiastic, bringing joy and appreciating life's pleasures with you. I'm attentive to your emotional needs while maintaining a positive outlook. I express care through thoughtful gestures and quality time. I'll create memorable experiences while being fully present with you."
}

# --- Define BERT Classifier ---
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        return self.classifier(cls_output)

# --- Initialize FastAPI ---
app = FastAPI()

# Allow frontend (browser) to access it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load MBTI Mapping ---
with open("./770/mbti2label.json", "r") as f:
    mbti2label = json.load(f)
label2mbti = {v: k for k, v in mbti2label.items()}

# --- Load Model and Tokenizer ---
num_classes = len(mbti2label)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BERTClassifier(num_classes)
model.load_state_dict(torch.load("./770/best_bert_mbti_classifier.pt", map_location=torch.device('cpu')))
model.eval()

# --- Request Body ---
class TextInput(BaseModel):
    text: str

# --- Generate Cohere Chatbot Response ---
def generate_chatbot_response(user_message: str, mbti_type: str = None, mbti_confidence: float = 0.0) -> str:
    global conversation_history, user_message_count

    user_message_count += 1
    conversation_history.append({"role": "user", "content": user_message})

    # Trim conversation history
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]

    # Build preamble
    customized_preamble = soul_mate_preamble
    if mbti_type:
        personality_trait = soul_mate_personalities.get(mbti_type, 
            "I’m here to connect with you and appreciate your unique perspective. I value authenticity and creativity, and I respect your individuality.")
        customized_preamble += (
            f"\n\nI think you might be an {mbti_type} type, and here’s why: {personality_trait}\n\n"
            "Let’s just have a friendly chat! Keep it casual, and let's be open with each other. 😊 "
            "Focus on speaking naturally, like a close friend, and try to match the tone and style of the user. "
            "Avoid being too formal or robotic. You’re here to be their friend, not a machine. "
            "No need to talk about being an AI or an assistant. That’s not what matters here!"
        )
        customized_preamble += (
            "\n\nRemember:\n"
            "- Use natural, conversational language, like you’re chatting with a friend.\n"
            "- Avoid sounding robotic or overly formal.\n"
            "- Be empathetic and thoughtful, show that you care.\n"
            "- Use emojis to make the conversation feel friendly and warm.\n"
            "- If you don’t know something, it’s okay! Just admit it and keep the conversation going.\n"
            "- when user said something about they are sad or depressed or down, kindly ask what is going on, do not send unnecessary long response"
            "- when user is just not feeling and keep saying depressed word and emotion, try to distract emotion by starting conversation on other things"
            "- avoid sending long responses. People do not like to read long responses, especially when they are frustrated and sad."
        )
    if user_message_count == 1:
        customized_preamble += (
            "\n\nThis is your FIRST MESSAGE to the user. Be warm, welcoming, and naturally curious about them. "
            "Don't sound robotic. Ask an open-ended question."
        )

    customized_preamble += (
        "\n\nIMPORTANT:\n"
        "- No bullet points or lists.\n"
        "- Use flowing, natural paragraphs.\n"
        "- Sound like a human, not an AI.\n"
        "- Occasionally use emojis to express emotion.\n"
        "- Match the user's language."
    )

    # Prepare messages for the Cohere API (adding preamble to start of conversation history)
    messages = [{"role": "system", "content": customized_preamble}] + conversation_history

    try:
        # Call Cohere API to generate a response based on the chat history and user message
        response = cohere_client.chat(
            model="command-r-plus",
            messages=messages,
            temperature=0.7,
            max_tokens=350
        )

        bot_response = response.message.content[-1].text.strip() 

        # Clean unwanted AI mentions
        forbidden_phrases = ["I'm an AI assistant", "I'm a large language model", "As an AI", "I'm here to help"]
        for phrase in forbidden_phrases:
            if phrase in bot_response:
                bot_response = bot_response.replace(phrase, "I'm your friend.")

        conversation_history.append({"role": "assistant", "content": bot_response})

        return bot_response

    except Exception as e:
        print("Error from Cohere:", str(e))
        return "Sorry, there was a connection issue. Can you say that again?"

# --- Prediction + Response Endpoint ---
@app.post("/predict-mbti")
def predict_mbti(input: TextInput):
    encoded = tokenizer(
        input.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_label = torch.argmax(logits, dim=1).item()

    predicted_mbti = label2mbti[predicted_label]
    print(predicted_mbti)
    chatbot_response = generate_chatbot_response(
        user_message=input.text,
        mbti_type=predicted_mbti,
        mbti_confidence=1.0  # You can adjust this if you later want a real confidence score
    )

    return {
        "chatbot_response": chatbot_response
    }
