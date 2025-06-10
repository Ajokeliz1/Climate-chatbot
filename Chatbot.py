import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time
from collections import defaultdict
import json

def preprocess(text):
    """Enhanced text preprocessing with spell checking and stemming"""
    text = text.lower()
    
    
    translator = str.maketrans('', '', string.punctuation.replace("'", "").replace("-", ""))
    text = text.translate(translator)
    
    
    contractions = {
        "what's": "what is",
        "it's": "it is",
        "i'm": "i am",
        "don't": "do not",
        "can't": "cannot",
        "wont": "will not",
        "climatechange": "climate change",
        "globalwarming": "global warming"
    }
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    
    return text.strip()


class KnowledgeBase:
    def __init__(self, file_path="Climate_FAQ.json"):
        self.file_path = file_path
        self.qa_pairs = defaultdict(list)
        self.load_data()
        
    def load_data(self):
        """Load knowledge base from JSON file"""
        try:
            if not os.path.exists(self.file_path):
                self.initialize_sample_data()
            else:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.qa_pairs = defaultdict(list, json.load(f))
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
            self.initialize_sample_data()
    
    def initialize_sample_data(self):
        """Initialize with sample climate data"""
        self.qa_pairs = defaultdict(list, {
            "general": [
                {"question": "what is climate change", "answer": "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities."},
                {"question": "what causes climate change", "answer": "The main causes are burning fossil fuels, deforestation, and industrial activities that increase greenhouse gases."},
                {"question": "how can i help", "answer": "Reduce energy use, eat less meat, use public transport, and support environmental policies."}
            ],
            "science": [
                {"question": "what is the greenhouse effect", "answer": "The greenhouse effect is when gases in Earth's atmosphere trap heat, similar to how glass traps heat in a greenhouse."}
            ]
        })
        self.save_data()
    
    def save_data(self):
        """Save knowledge base to file"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.qa_pairs), f, indent=2)
    
    def add_question(self, category, question, answer):
        """Add new Q&A pair to knowledge base"""
        self.qa_pairs[category].append({
            "question": preprocess(question),
            "answer": answer
        })
        self.save_data()
    
    def get_all_questions(self):
        """Get all questions for similarity matching"""
        return [q["question"] for cat in self.qa_pairs.values() for q in cat]

    def get_answer(self, question):
        """Get answer for a specific question"""
        processed_q = preprocess(question)
        for cat in self.qa_pairs.values():
            for qa in cat:
                if qa["question"] == processed_q:
                    return qa["answer"]
        return None

# ================== ENHANCED RESPONSE GENERATION ==================
class ChatbotEngine:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer()
        self._train_vectorizer()
        self.conversation_history = []
    
    def _train_vectorizer(self):
        """Train TF-IDF vectorizer on all questions"""
        questions = self.kb.get_all_questions()
        if questions:
            self.vectorizer.fit(questions)
    
    def get_response(self, query):
        """Generate response with context awareness"""
        start_time = time.time()
        
        # Check for greetings/special commands first
        response = self._check_special_queries(query)
        if response:
            return response
        
        # Find best matching question
        processed_query = preprocess(query)
        questions = self.kb.get_all_questions()
        
        if not questions:
            return "I'm not properly initialized with knowledge."
        
        # Vectorize query and questions
        query_vec = self.vectorizer.transform([processed_query])
        question_vecs = self.vectorizer.transform(questions)
        
        # Calculate similarities and get best match
        similarities = cosine_similarity(query_vec, question_vecs)
        best_match_idx = np.argmax(similarities)
        best_score = similarities[0, best_match_idx]
        
        # Get all possible answers
        all_answers = []
        for cat in self.kb.qa_pairs.values():
            for qa in cat:
                if qa["question"] == questions[best_match_idx]:
                    all_answers.append(qa["answer"])
        
        # Select response based on confidence
        if best_score > 0.7:  # High confidence threshold
            response = np.random.choice(all_answers) if len(all_answers) > 1 else all_answers[0]
        elif best_score > 0.4:  # Medium confidence
            response = f"I think you're asking about: {all_answers[0]}\nIs this what you meant?"
        else:  # Low confidence
            response = "I'm not sure I understand. Could you rephrase your question about climate change?"
        
        # Log interaction
        self.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "processing_time": time.time() - start_time
        })
        
        return response
    
    def _check_special_queries(self, query):
        """Handle greetings, thanks, and special commands"""
        query_lower = query.lower()
        
        greetings = {'hi', 'hello', 'hey', 'greetings'}
        if any(g in query_lower for g in greetings):
            return "Hello! I'm your Climate Change Assistant. Ask me anything about environmental science!"
        
        goodbyes = {'bye', 'goodbye', 'exit', 'quit'}
        if any(g in query_lower for g in goodbyes):
            return "Goodbye! Remember to think green in your daily choices!"
        
        thanks = {'thank', 'thanks', 'appreciate'}
        if any(t in query_lower for t in thanks):
            return "You're welcome! I'm happy to help with any climate-related questions."
        
        if "your name" in query_lower:
            return "I'm EcoBot, your climate change information assistant!"
        
        return None

# ================== STREAMLIT INTERFACE ==================
def main():
    st.set_page_config(page_title="EcoBot - Climate Change Assistant", layout="wide")
    
    # Initialize knowledge base and chatbot
    if 'kb' not in st.session_state:
        st.session_state.kb = KnowledgeBase()
    if 'bot' not in st.session_state:
        st.session_state.bot = ChatbotEngine(st.session_state.kb)
    
    # Sidebar for admin functions
    with st.sidebar:
        st.header("Knowledge Base Management")
        st.subheader("Add New Q&A Pair")
        
        category = st.text_input("Category (e.g., 'science', 'policies')", key="new_cat")
        question = st.text_input("Question", key="new_q")
        answer = st.text_area("Answer", key="new_a")
        
        if st.button("Add to Knowledge Base"):
            if question and answer:
                st.session_state.kb.add_question(
                    category if category else "general",
                    question,
                    answer
                )
                st.session_state.bot._train_vectorizer()  # Retrain with new data
                st.success("Question added successfully!")
            else:
                st.warning("Please enter both question and answer")
        
        st.subheader("Statistics")
        total_questions = sum(len(cat) for cat in st.session_state.kb.qa_pairs.values())
        st.write(f"Total Q&A Pairs: {total_questions}")
        st.write(f"Categories: {', '.join(st.session_state.kb.qa_pairs.keys())}")
    
    # Main chat interface
    st.title("🌱 EcoBot - Climate Change Assistant")
    st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: #f0f8f0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    for i, exchange in enumerate(st.session_state.conversation):
        with st.chat_message("user" if exchange['user'] else "assistant"):
            st.write(exchange['text'])
    
    # User input
    user_query = st.chat_input("Ask me anything about climate change...")
    
    if user_query:
        # Add user message to conversation
        st.session_state.conversation.append({'user': True, 'text': user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Get and display bot response
        with st.spinner("Thinking..."):
            response = st.session_state.bot.get_response(user_query)
        
        st.session_state.conversation.append({'user': False, 'text': response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()