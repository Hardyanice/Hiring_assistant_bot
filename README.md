# TalentScout Hiring Assistant — README

## Project Overview  
TalentScout Hiring Assistant is an intelligent LLM-powered chatbot designed to conduct **initial candidate screening** for a fictional tech recruitment agency, *TalentScout*.  
The assistant collects essential candidate details, analyzes sentiment, generates personalized technical interview questions based on the candidate’s declared tech stack, and evaluates their answers conversationally—simulating a realistic interviewer experience.

This project demonstrates **prompt engineering**, **context-aware LLM behavior**, **validation pipelines**, **GDPR-friendly data handling**, and a complete **Streamlit conversational UI**.

---

## Key Objectives

The chatbot is engineered to:

### Collect structured candidate information  
- Full Name  
- Age  
- Email  
- Phone  
- Experience (validated using age bracket rules)  
- Position applied for  
- Location  
- Tech stack  

### Generate tailored technical questions  
- 3–5 custom questions **per technology**  
- Tailored to the candidate’s **role and experience level**  
- Ability to regenerate or rewrite questions using semantic similarity (not keywords)

### Maintain context and flow  
- Greeting detection using embeddings  
- State-based conversation progression  
- Different responses for:  
  - Nonsense  
  - Clarification  
  - Rewrite requests  
  - Technical evaluation  

### Evaluate candidate’s technical answers  
The assistant provides a conversational evaluation:  
- What the candidate did correctly  
- Gaps or missing details  
- Gentle improvement suggestions  
- Follow-up question  

### Handle GDPR-safe data  
- Explicit consent screen  
- Only **fake data** allowed  
- Data is stored **locally**, never uploaded  
- Users can optionally **download** their own session summary

---

## Architecture & Workflow

### Streamlit Front-End  
- Clean chat UI  
- Reset handling  
- Download summary feature  

### Session State Machine  
Conversation moves through a defined sequence:

greet → name → age → email → phone → experience → position → location → tech_stack → generate_questions → evaluation

### Cohere LLM + Embeddings  
Used for:
- Question generation  
- Answer evaluation  
- Detecting rewrite requests  
- Nonsense detection using cosine similarity  
- Sentiment analysis based on embedding similarity  

### Local Simulated Database  
Each finished interview generates:  
- Candidate details  
- Sentiment history  
- Stored in `candidate_records.jsonl`  

Users can download their personal summary as `candidate_summary.json`.

---

## Prompt Engineering Strategy

### System Prompt — Purpose Lock  
Ensures:  
- The assistant stays in hiring mode  
- Prevents deviation from defined purpose  
- Produces professional & concise outputs  

### Stepwise Prompts  
Each stage has a unique prompt for extraction or generation.

### Semantic Intent Detection  
Without relying on keywords — instead uses embeddings to classify:  
- Rewrite requests  
- Clarification questions  
- Real technical answers  
- Nonsense or unrelated text  

This fulfills the assignment requirements for *contextual LLM behavior*.

---

## Data Privacy & GDPR Compliance

This demo:  
✔ Requires a **consent checkbox**  
✔ Only accepts **fake candidate information**  
✔ Stores data *locally* in JSONL  
✔ Ensures data is never uploaded or shared  
✔ Provides optional download of candidate’s **own data**  

The project fully meets the data protection guidelines.

---

## Installation

### Clone the repository  

git clone https://github.com/<your-username>/hiring_assistant_bot.git
cd hiring_assistant_bot

### Install dependencies
pip install -r requirements.txt

### Set Cohere API Key
export COHERE_API_KEY="your_api_key_here"

### Run the app
streamlit run final_bot.py

## Usage Guide

- Start the app.

- Accept GDPR notice.

- Chatbot begins collecting candidate information.

- Provide tech stack → chatbot generates personalized technical questions.

- Write your answers — chatbot evaluates them.

- Ask for clarification or request fresh questions anytime.

- Type "exit" to conclude the interview.

- Download your candidate summary JSON.

## Challenges & Solutions

### Challenge 1 — Preventing irrelevant input

#### Solution:
Cosine similarity with hiring-domain embeddings to detect nonsense and classify input intent.

### Challenge 2 — Personalized technical questions

#### Solution:
Dynamic prompt generation tailored to tech stack + position.

### Challenge 3 — Reliable conversation flow

#### Solution:
State machine approach with st.session_state.step.

### Challenge 4 — Rewrite question requests

#### Solution:
Embedding-based semantic intent detection—no hardcoded keywords.

### Challenge 5 — Privacy considerations

#### Solution:
Consent screen, fake data enforcement, local-only storage.

## Technologies Used

- Python 3

- Streamlit

- Cohere Chat API

- Cohere Embeddings API

- NumPy

- JSONL backend simulation

## Demo (Optional)

 - Demo Video: Included in github repo 

- Live App: https://hiringassistantbot-fdw3rshguxfgkdjwdqfvgv.streamlit.app
