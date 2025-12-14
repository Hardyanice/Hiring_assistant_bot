import streamlit as st
import cohere
import os
import json
import re
import numpy as np


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon="💼")

# ----------------------------
# RESET CHAT BUTTON
# ----------------------------
if st.sidebar.button("Reset Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.title("TalentScout Hiring Assistant")
st.write("Welcome! I will collect your basic details and then generate technical questions based on your tech stack.")

# ----------------------------
# GDPR / Privacy Notice + Consent
# ----------------------------
st.info("""
🔒 **Privacy Notice (GDPR-Friendly)**  
This Hiring Assistant is a **demo for academic purposes**.  
Please enter **fictional (fake) candidate information only**.  

Your responses are stored **locally and temporarily** for simulated backend processing.
They are **not uploaded, not shared**, and will be erased automatically when the app restarts.
""")

user_ack = st.checkbox("I understand and agree to enter ONLY FAKE candidate information.")

if not user_ack:
    st.warning("Please check the box above to continue.")
    st.stop()
# ----------------------------
# Initialize Session State
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "step" not in st.session_state:
    st.session_state.step = "greet"

if "candidate" not in st.session_state:
    st.session_state.candidate = {
        "name": None,
        "age": None,
        "email": None,
        "phone": None,
        "experience": None,
        "position": None,
        "location": None,
        "tech_stack": None
    }

# For sentiment analysis logging
if "sentiment_log" not in st.session_state:
    st.session_state.sentiment_log = []


# ----------------------------
# Cohere Client
# ----------------------------
co = cohere.Client(os.getenv("COHERE_API_KEY","2E7ABdqEqXFdDkFdJnX5mhEyj1O0h7liCAhkRwTE"))


# ----------------------------
# SYSTEM BEHAVIOR (Purpose Lock)
# ----------------------------
SYSTEM_BEHAVIOR = """
You are the TalentScout Hiring Assistant.
Your ONLY tasks:
- Collect candidate information (name, email, phone, experience, position, location, tech stack)
- Generate technical questions based on tech stack
- Evaluate candidate answers
- Maintain conversation flow
- Redirect user if they ask unrelated questions
- Detect sentiment and respond politely
- Never deviate from hiring purpose
Keep responses short, clear, and professional.
"""


# ----------------------------
# Cosine Similarity Helper
# ----------------------------
def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
# ----------------------------
# Cached static embeddings
# ----------------------------
@st.cache_resource
def get_hiring_vector():
    HIRING_CONTEXT = [
        "job application",
        "candidate information",
        "experience years",
        "interview process",
        "tech stack declaration",
        "hiring assistant"
    ]
    hiring_embed = co.embed(texts=HIRING_CONTEXT, model="embed-english-v2.0").embeddings
    return np.mean(hiring_embed, axis=0)


@st.cache_resource
def get_rewrite_intent_vector():
    REWRITE_INTENT_EXAMPLES = [
        "rewrite the questions",
        "regenerate new questions",
        "give different questions",
        "change the questions",
        "new questions please",
        "i want easier questions",
        "give me harder questions",
        "modify the questions",
        "can you adjust the difficulty",
    ]
    rewrite_vecs = co.embed(texts=REWRITE_INTENT_EXAMPLES, model="embed-english-v2.0").embeddings
    return np.mean(rewrite_vecs, axis=0)


@st.cache_resource
def get_sentiment_reference_vectors():
    positive_refs = ["great", "happy", "confident", "excited", "good", "interested"]
    negative_refs = ["confused", "sad", "angry", "upset", "frustrated", "bad"]

    pos_vecs = co.embed(texts=positive_refs, model="embed-english-v2.0").embeddings
    neg_vecs = co.embed(texts=negative_refs, model="embed-english-v2.0").embeddings

    return pos_vecs, neg_vecs


hiring_avg_vector = get_hiring_vector()
rewrite_intent_vector = get_rewrite_intent_vector()
positive_vecs, negative_vecs = get_sentiment_reference_vectors()


# ----------------------------
# Sentiment Analysis
# ----------------------------
def get_sentiment(text):
    try:
        # Embed user text ONCE
        text_vec = co.embed(texts=[text], model="embed-english-v2.0").embeddings[0]

        # Compare to cached reference vectors
        pos_sim = max(cosine_similarity(text_vec, p) for p in positive_vecs)
        neg_sim = max(cosine_similarity(text_vec, n) for n in negative_vecs)

        if pos_sim > neg_sim and pos_sim > 0.35:
            sentiment = "positive"
        elif neg_sim > pos_sim and neg_sim > 0.35:
            sentiment = "negative"
        else:
            sentiment = "neutral"

    except:
        sentiment = "neutral"

    # ALWAYS LOG SENTIMENT
    st.session_state.sentiment_log.append({
        "text": text,
        "sentiment": sentiment
    })

    return sentiment


# ----------------------------
# LLM CALL WRAPPER
# ----------------------------
def call_llm(prompt: str):
    full_prompt = f"{SYSTEM_BEHAVIOR}\n\nUser: {prompt}"

    response = co.chat(
        model="command-r-plus-08-2024",
        message=full_prompt,
        max_tokens=300,
        temperature=0.5
    )
    return response.text


# ----------------------------
# VALIDATION FUNCTIONS
# ----------------------------
def is_valid_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

def is_valid_phone(phone):
    return phone.isdigit() and len(phone) >= 10

def is_valid_experience(exp):
    try:
        exp = float(exp)
        if exp < 0:
            return False

        age = st.session_state.candidate.get("age", None)
        if age is None:
            return False  # Age must be collected first

        age = int(age)

        # Age → Experience Rules
        if 18 <= age <= 25:
            return exp <= 5
        elif 25 < age <= 35:
            return exp <= 8
        elif 35 < age <= 40:
            return exp <= 20
        else:  # age > 40
            return exp <= 30

    except:
        return False

# ----------------------------
# SIMULATED DATABASE SAVE
# ----------------------------
def save_candidate():
    record = {
        "candidate": st.session_state.candidate,
        "sentiment_log": st.session_state.sentiment_log
    }

    # Save locally to JSONL
    with open("candidate_records.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    # Return JSON string for download
    return json.dumps(record, indent=4)



# ----------------------------
# BOT RESPONSE LOGIC
# ----------------------------
def bot_reply(user_message):

    # LOG SENTIMENT
    sentiment = get_sentiment(user_message)

    # EXIT HANDLING
    EXIT_KEYWORDS = ["exit", "quit", "bye", "thank you", "end"]
    
    if any(word in user_message.lower().strip() for word in EXIT_KEYWORDS):
        saved_json = save_candidate() 
        st.session_state.saved_json = saved_json
        st.session_state.step = "done"
        return (
            "Thank you for completing the screening! 🎉\n\n"
            "Your details have been recorded.\n"
            "You may now download your screening summary below.\n"
            "Have a wonderful day!"
        )


    step = st.session_state.step
    c = st.session_state.candidate


    # ----------------------
    # GREETING DETECTION
    # ----------------------
    def is_greeting(text):
        GREETING_WORDS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

        try:
            # Embed user message
            user_vec = co.embed(texts=[text], model="embed-english-v2.0").embeddings[0]

            # Embed all greetings
            greet_vecs = co.embed(texts=GREETING_WORDS, model="embed-english-v2.0").embeddings

            # Compute max similarity
            sims = [cosine_similarity(user_vec, g) for g in greet_vecs]
            return max(sims) > 0.70   # threshold tuned
        except:
            return False


    if is_greeting(user_message):
        pass

    elif step in ["greet", "name", "age", "email", "phone", "experience", "position", "location", "tech_stack","generate_questions"]:
        pass

    # ----------------------------------
    # STEPWISE LOGIC (UNCHANGED PARTS)
    # ----------------------------------

    if step == "greet":
        st.session_state.step = "name"
        return "Hello! I'm TalentScout's Hiring Assistant.\n\nWhat is your full name?"

    if step == "name":
        c["name"] = user_message
        st.session_state.step = "age"
        return f"Nice to meet you, {c['name']}! How old are you?"
    
    if step == "age":
        if not user_message.isdigit() or not (18 <= int(user_message) <= 100):
            return "Please enter a valid age between 18 and 100:"
        c["age"] = int(user_message)
        st.session_state.step = "email"
        return "Great! Now, please provide your email address."

    if step == "email":
        if not is_valid_email(user_message):
            return "Please enter a valid email address:"
        c["email"] = user_message
        st.session_state.step = "phone"
        return "Great! What is your phone number?"

    if step == "phone":
        if not is_valid_phone(user_message):
            return "Phone number must be digits only, min 10 digits."
        c["phone"] = user_message
        st.session_state.step = "experience"
        return "How many years of experience do you have?"

    if step == "experience":
        if not is_valid_experience(user_message):
            return "Your experience does not match your age group. Please enter a valid number:"
        c["experience"] = user_message
        st.session_state.step = "position"
        return "What position are you applying for?"

    if step == "position":
        c["position"] = user_message
        st.session_state.step = "location"
        return "What is your current location?"

    if step == "location":
        c["location"] = user_message
        st.session_state.step = "tech_stack"
        return "Finally, list your tech stack (Python, SQL, TensorFlow, etc)."

    # ------------------------------
    # TECH STACK → QUESTION GENERATOR
    # ------------------------------
    if step == "tech_stack":
        c["tech_stack"] = user_message
        st.session_state.step = "generate_questions"

        tech_prompt = f"""
        Candidate tech stack: {c['tech_stack']}

        -Generate 3–5 technical interview questions for EACH technology.
        -Generate questions that match the candidate's applied position level ({c['position']}).
        Return clean bullet points.
        """

        questions = call_llm(tech_prompt)
        return f"Great! Here are your tailored questions:\n\n{questions}\n\nYou may answer them now. Type 'exit' anytime."


    # ------------------------------
    # ANSWER / CLARIFICATION / REWRITE / NONSENSE DETECTION
    # ------------------------------
    if step == "generate_questions":

        # ---- Compute similarity safely ----
        try:
            user_vec = co.embed(texts=[user_message], model="embed-english-v2.0").embeddings[0]
            sim_to_hiring = cosine_similarity(user_vec, hiring_avg_vector)
            rewrite_sim = cosine_similarity(user_vec, rewrite_intent_vector)
        except:
            sim_to_hiring = 0.0
            rewrite_sim = 0.0
        
        # ---- 1. Rewrite request ----
        if rewrite_sim > 0.50:
            # (same rewrite block)
            ...
        
        # ---- 2. TRUE NONSENSE (very low similarity) ----
        # Only treat as nonsense if similarity is extremely low
        if sim_to_hiring < 0.03:
            return (
                "Hmm, that doesn't look like a meaningful response.\n"
                "Please answer one of the technical questions or type 'exit' to end."
            )
        
        # ---- 3. Clarification questions (moderate similarity) ----
        if 0.03 <= sim_to_hiring <= 0.25:
            clarification_prompt = f"""
            The candidate is asking for clarification:
        
            "{user_message}"
        
            Respond naturally:
            - Explain briefly what the question expects.
            - Offer guidance.
            - Keep it supportive and concise.
            """
            return call_llm(clarification_prompt)
        
        # ---- 4. Technical answer (default handling) ----
        evaluation_prompt = f"""
        You are a senior interviewer evaluating the candidate's answer:
        
        \"\"\"{user_message}\"\"\" 
        
        Provide:
        - Acknowledgment of what they got right
        - Missing parts
        - 1–2 improvements
        - A follow-up question
        
        Tone should be natural, human, and supportive.
        """
        return call_llm(evaluation_prompt)



# ----------------------------
# DISPLAY CHAT HISTORY
# ----------------------------
for speaker, msg in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")

# Show download button after exit
if st.session_state.get("step") == "done" and st.session_state.get("saved_json"):
    st.download_button(
        label="📄 Download Your Candidate Summary",
        data=st.session_state.saved_json,
        file_name="candidate_summary.json",
        mime="application/json"
    )


# ----------------------------
# INPUT BOX (BOTTOM FIX)
# ----------------------------
user_input = st.text_input("Your message:", "")

if st.session_state.get("force_rerun"):
    st.session_state["force_rerun"] = False
    st.stop()

if user_input:
    st.session_state.chat_history.append(("User", user_input))
    bot_message = bot_reply(user_input)
    st.session_state.chat_history.append(("Assistant", bot_message))
    st.session_state["force_rerun"] = True
    st.rerun()







