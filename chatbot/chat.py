# ============================================================
#   BANK CHATBOT - ML Based (scikit-learn + NLTK)
#   For beginners | Runs in the terminal
# ============================================================

import numpy as np
import random
import string
import warnings
warnings.filterwarnings("ignore")

# --- NLTK Setup ---
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only needed once)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Scikit-learn for ML ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

lemmatizer = WordNetLemmatizer()


# ============================================================
# STEP 1: TRAINING DATA
# This is the "brain" of the chatbot.
# Each intent has:
#   - a tag (the category name)
#   - patterns (example sentences users might type)
#   - responses (what the bot replies)
# ============================================================

intents = [
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "good morning", "good evening",
            "howdy", "hi there", "greetings", "what's up", "good afternoon"
        ],
        "responses": [
            "Hello! Welcome to ABC Bank. How can I help you today?",
            "Hi there! I'm your bank assistant. What can I do for you?",
            "Good day! How may I assist you with your banking needs?"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "bye", "goodbye", "see you", "exit", "quit", "take care",
            "see you later", "I'm done", "that's all", "close"
        ],
        "responses": [
            "Thank you for using ABC Bank. Goodbye!",
            "Have a great day! Stay safe.",
            "Goodbye! Don't hesitate to return if you need anything."
        ]
    },
      {
        "tag": "john",
        "patterns": [
               "my name is john",
        "I am john",
        "call me john",
        "this is john",
        "john here",
        "hey it's jon",
        "jan speaking",
        "jen calling"
        ],
        "responses": [
            "Welcome sir",
            "Hey There",
            "Hello sir",
            "nice to meet you sir",
            "happy to meet",
            "happy to answer",
            "good morning"
        ]
    },
    {
        "tag": "check_balance",
        "patterns": [
            "what is my balance", "check my balance", "how much money do I have",
            "show my account balance", "current balance", "account balance",
            "tell me my balance", "what's in my account", "my funds",
            "how much is in my account"
        ],
        "responses": [
            "Your current account balance is ₹45,230.00. (This is a demo balance)",
            "As of today, your balance stands at ₹45,230.00. For real balance, please log in to your account.",
            "Demo balance: ₹45,230.00. To check your real balance, use the mobile app or visit a branch."
        ]
    },
    {
        "tag": "transfer_money",
        "patterns": [
            "transfer money", "send money", "I want to transfer funds",
            "how to transfer", "move money", "pay someone", "fund transfer",
            "send funds to another account", "wire transfer", "NEFT", "IMPS", "RTGS"
        ],
        "responses": [
            "To transfer money: 1) Log in to net banking 2) Go to 'Fund Transfer' 3) Add beneficiary 4) Enter amount and confirm. Need help with anything else?",
            "You can transfer via NEFT, IMPS, or RTGS. IMPS is instant 24/7. Want me to explain the difference?",
            "For transfers, visit our app or net banking portal. Make sure you have the recipient's account number and IFSC code ready."
        ]
    },
    {
        "tag": "loan",
        "patterns": [
            "I need a loan", "apply for loan", "home loan", "personal loan",
            "car loan", "loan interest rate", "loan eligibility", "how to get a loan",
            "EMI calculator", "loan application", "education loan", "business loan"
        ],
        "responses": [
            "We offer Home Loans (7.5% p.a.), Personal Loans (10.5% p.a.), and Car Loans (8.5% p.a.). Which one interests you?",
            "To apply for a loan, visit our nearest branch or apply online at our website. You'll need ID proof, income proof, and address proof.",
            "Our loans start from ₹50,000 with flexible EMI options. Shall I explain the eligibility criteria?"
        ]
    },
    {
        "tag": "lost_card",
        "patterns": [
            "I lost my card", "lost my debit card", "lost my credit card",
            "card stolen", "block my card", "my card is missing",
            "card lost", "stolen card", "deactivate my card", "freeze card"
        ],
        "responses": [
            "🚨 URGENT: Call our 24/7 helpline 1800-XXX-XXXX immediately to block your card. You can also block it instantly via our mobile app under 'Card Settings'.",
            "Don't worry! To block your card right away: open our app → Cards → Block Card. Or call 1800-XXX-XXXX anytime.",
            "Your card can be blocked instantly! Go to our app → My Cards → Temporarily Block. Report to us immediately to prevent misuse."
        ]
    },
    {
        "tag": "atm",
        "patterns": [
            "nearest ATM", "find ATM", "ATM location", "where is ATM",
            "cash withdrawal", "ATM near me", "where can I withdraw cash",
            "ATM not working", "ATM limit"
        ],
        "responses": [
            "You can find the nearest ATM on our mobile app under 'ATM Locator', or visit our website. Daily ATM withdrawal limit is ₹20,000.",
            "Use our ATM Locator in the app to find the nearest ATM. Having trouble with an ATM? Call 1800-XXX-XXXX.",
            "Daily cash withdrawal limit at ATMs is ₹20,000. For higher amounts, visit our branch."
        ]
    },
    {
        "tag": "open_account",
        "patterns": [
            "open an account", "create account", "new account", "savings account",
            "current account", "how to open account", "account opening",
            "start a bank account", "I want to open account"
        ],
        "responses": [
            "To open an account, you need: Aadhaar card, PAN card, passport photo, and a minimum deposit. Visit our branch or apply online!",
            "Account opening is simple! Apply online on our website with your KYC documents, or walk into any branch.",
            "We offer Savings, Current, and Fixed Deposit accounts. Online account opening takes just 10 minutes with your Aadhaar and PAN."
        ]
    },
    {
        "tag": "fixed_deposit",
        "patterns": [
            "fixed deposit", "FD", "create FD", "FD interest rate",
            "how to make FD", "invest in FD", "FD calculator", "break FD",
            "premature withdrawal FD", "FD maturity"
        ],
        "responses": [
            "Our FD interest rates range from 6.5% to 7.2% p.a. depending on tenure. You can book an FD instantly via our app!",
            "Fixed Deposits start from ₹1,000 with tenures from 7 days to 10 years. Senior citizens get an extra 0.5% interest.",
            "To create an FD: App → Investments → Fixed Deposit → Enter amount and tenure → Confirm. Simple and safe!"
        ]
    },
    {
        "tag": "customer_support",
        "patterns": [
            "talk to agent", "human support", "customer care", "contact support",
            "speak to representative", "helpline", "complaint", "I have a complaint",
            "real person", "customer service number"
        ],
        "responses": [
            "You can reach our customer care at 1800-XXX-XXXX (toll-free, 24/7), or email support@abcbank.com.",
            "To file a complaint or speak to an agent, call 1800-XXX-XXXX. You can also visit any branch.",
            "Our support team is available 24/7. Call 1800-XXX-XXXX or use the 'Raise Complaint' option in our app."
        ]
    },
    {
        "tag": "thanks",
        "patterns": [
            "thanks", "thank you", "that's helpful", "great", "awesome",
            "perfect", "wonderful", "nice", "good", "ok thanks"
        ],
        "responses": [
            "You're welcome! Is there anything else I can help you with?",
            "Happy to help! Let me know if you have more questions.",
            "Glad I could assist! Anything else you need?"
        ]
    }
]


# ============================================================
# STEP 2: PREPARE TRAINING DATA
# Convert patterns into format the ML model can learn from
# ============================================================

def preprocess(text):
    """Clean and lemmatize text"""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in string.punctuation]
    return " ".join(tokens)

# Build training lists
train_sentences = []
train_labels = []

for intent in intents:
    for pattern in intent["patterns"]:
        train_sentences.append(preprocess(pattern))
        train_labels.append(intent["tag"])


# ============================================================
# STEP 3: TRAIN THE ML MODEL
# TF-IDF converts text to numbers
# Logistic Regression learns which numbers → which intent
# ============================================================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_sentences)

model = LogisticRegression(max_iter=200)
model.fit(X, train_labels)

print("✅ Model trained successfully on", len(train_sentences), "examples!\n")


# ============================================================
# STEP 4: PREDICTION FUNCTION
# Given a user message, predict which intent it belongs to
# ============================================================

def predict_intent(user_input):
    """Predict the intent of user input"""
    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    predicted_tag = model.predict(vectorized)[0]
    confidence = max(model.predict_proba(vectorized)[0])
    return predicted_tag, confidence

def get_response(tag):
    """Get a random response for the predicted intent"""
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I didn't understand that. Can you rephrase?"


# ============================================================
# STEP 5: CHAT LOOP
# This is where the user interacts with the bot in the terminal
# ============================================================

def chat():
    print("=" * 55)
    print("       🏦  Welcome to ABC Bank Chatbot  🏦")
    print("=" * 55)
    print("  I can help you with:")
    print("  • Account balance       • Fund transfers")
    print("  • Loans & EMI           • Lost/stolen card")
    print("  • Fixed deposits        • ATM locations")
    print("  • Account opening       • Customer support")
    print("-" * 55)
    print("  Type 'bye' or 'exit' to end the conversation.")
    print("=" * 55)
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Predict intent
        tag, confidence = predict_intent(user_input)

        # If confidence is too low, give a fallback response
        if confidence < 0.25:
            print("Bot: I'm not sure I understood that. Could you rephrase? You can ask about balance, loans, transfers, or lost cards.")
        else:
            response = get_response(tag)
            print(f"Bot: {response}")

        # Exit condition
        if tag == "goodbye":
            break

        print()

# ============================================================
# RUN THE CHATBOT
# ============================================================

if __name__ == "__main__":
    chat()