import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalRAGEngine:
    def __init__(self):
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.initialize_corpus()

    def initialize_corpus(self):
        # 1. Load proposal text
        proposal_path = r"C:\Users\PRAVEEN PRABAKARN\Desktop\new fold\AGRI_AI_BACKEND\agri_pdf_text.txt"
        text = ""
        if os.path.exists(proposal_path):
            try:
                with open(proposal_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"RAG Engine: Failed to read {proposal_path}: {e}")
                
        if not text:
            # Fallback proposal proposal content
            text = """
AGRISMART – SMART FARMING SYSTEM WITH AI  
Overview:  
This project aims to revolutionize traditional agriculture by introducing a Smart Farming System (AgriSmart) that leverages Artificial Intelligence (AI) and Internet of Things (IoT) technologies. The system will provide farmers with real-time data on soil health, weather conditions, and crop growth to help them make informed decisions and optimize their crop yields.  
Features:  
1. Soil Analysis and Fertilizer Recommendation: The system analyzes soil parameters like Nitrogen (N), Phosphorus (P), Potassium (K), pH, moisture, and temperature to recommend suitable crops and fertilizers.
2. Crop Disease Detection: Uses an image analysis model to analyze crop images, identify diseases, and provide treatment solutions.
3. Market Trends Analysis: Real-time response from the internet regarding current crop prices, demand, and market trends.
4. AI Assistant: Conversational assistant providing intelligence to farmers.
Development Team: Designed by PRAVEEN PRABAKARAN, Microsoft Print To PDF Producer. Project Proposal created in June 2026.
Implementation details: Built with a Node.js API Gateway, Express Server, SQLite database, and Python FastAPI AI analysis module.
            """

        # Split text into chunks (e.g. by double newlines or paragraphs)
        raw_chunks = re.split(r'\n\n+|\.\s*\n', text)
        self.chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 30]
        
        # Add basic farming knowledge chunks to make the assistant smarter offline
        self.chunks.extend([
            "Soil pH levels below 5.5 are acidic and require agricultural lime (calcium carbonate) to neutralize. pH levels above 7.8 are alkaline and require gypsum to reduce alkalinity.",
            "Nitrogen (N) deficiency leads to stunted growth and yellowing of older leaves (chlorosis). Apply urea (46% N) to quickly correct nitrogen deficits.",
            "Phosphorus (P) is vital for root development, flowering, and seed formation. Deficiencies cause dark green or purple leaves. Correct with SSP (Single Superphosphate) or DAP.",
            "Potassium (K) regulates water balance, enzyme activation, and disease resistance. Deficiencies show as scorching/burning on leaf margins. Correct with MOP (Muriate of Potash).",
            "Crop rotation (alternating cereals like rice or maize with leguminous crops like groundnut or pulses) naturally restores soil nitrogen levels and breaks pest life cycles.",
            "Integrated Pest Management (IPM) incorporates biological controls (beneficial insects), mechanical traps (yellow sticky boards), and cultural practices before deploying chemical pesticides."
        ])

        # Vectorize corpus
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve_context(self, query: str, top_n: int = 2) -> str:
        """Find the top_n most relevant text chunks matching the query using TF-IDF cosine similarity."""
        if not self.chunks or self.tfidf_matrix is None:
            return "No local document context available."
            
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[::-1][:top_n]
            
            # Filter matches with a minimum similarity score to avoid irrelevant text
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    results.append(self.chunks[idx])
                    
            if not results:
                return "No high-confidence document chunks found."
                
            return "\n\n".join(results)
        except Exception as e:
            print(f"RAG Search Error: {e}")
            return "RAG Search execution failed."
            
    def offline_qa(self, query: str, context: str) -> str:
        """Rule-based QA answering if offline without LLM keys."""
        query_lower = query.lower()
        if "author" in query_lower or "designed" in query_lower or "creator" in query_lower:
            return "According to the project proposal document, AgriSmart was designed and authored by **PRAVEEN PRABAKARAN**."
        if "overview" in query_lower or "about" in query_lower or "what is" in query_lower:
            return "The AgriSmart Smart Farming System aims to revolutionize traditional agriculture by integrating Artificial Intelligence (AI) and IoT. It monitors soil health, predicts disease, fetches real-time market trends, and supports farm decision making."
        
        return f"""Based on AgriSmart local documentation:
{context}

(Note: To receive a fully custom, conversational response to complex questions, please connect an API key in your .env file.)"""

rag_engine = LocalRAGEngine()
