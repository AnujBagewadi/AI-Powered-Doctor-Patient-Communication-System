# llm_service.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import os
import torch

class LLMService:
    """Service for handling LLM-based operations like summarization, QA, and translation"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.load_models()
        
    def load_models(self):
        """Load all required models"""
        # Load summarization model
        self.tokenizers['summarization'] = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.models['summarization'] = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(self.device)
        
        # Load QA model for medical chat
        self.tokenizers['qa'] = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.models['qa'] = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(self.device)
        
        # Load translation model
        self.tokenizers['translation'] = AutoTokenizer.from_pretrained("t5-small")
        self.models['translation'] = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
        
        # Create pipelines
        self.summarizer = pipeline("summarization", model=self.models['summarization'], tokenizer=self.tokenizers['summarization'], device=0 if self.device == "cuda" else -1)
        self.qa_pipeline = pipeline("question-answering", model=self.models['qa'], tokenizer=self.tokenizers['qa'], device=0 if self.device == "cuda" else -1)
        self.translator = pipeline("translation_en_to_fr", model=self.models['translation'], tokenizer=self.tokenizers['translation'], device=0 if self.device == "cuda" else -1)
    
    def summarize_text(self, text, max_length=150, min_length=50):
        """Generate a summary of the given text"""
        if not text:
            return ""
        
        # Split text into chunks if it's too long
        max_tokens = self.tokenizers['summarization'].model_max_length - 100  # Buffer for generation
        tokenized = self.tokenizers['summarization'].encode(text)
        
        if len(tokenized) <= max_tokens:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            return summary
        
        # If text is too long, summarize in chunks and then summarize the combined summaries
        chunks = self._split_text(text)
        chunk_summaries = []
        
        for chunk in chunks:
            chunk_summary = self.summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            chunk_summaries.append(chunk_summary)
        
        combined_summary = " ".join(chunk_summaries)
        
        # If the combined summary is still too long, summarize it again
        if len(self.tokenizers['summarization'].encode(combined_summary)) > max_tokens:
            combined_summary = self.summarize_text(combined_summary, max_length=max_length, min_length=min_length)
        
        return combined_summary
    
    def _split_text(self, text, max_chunk_size=1000):
        """Split text into chunks of approximately equal size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for the space
            
            if current_size >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def answer_question(self, question, context=None):
        """Answer a medical question using the QA model"""
        if not context:
            # Default medical context if none provided
            context = """
            Medical knowledge encompasses diagnosis, treatment, and prevention of disease, 
            illness, injury, and other physical and mental impairments in humans. Medicine 
            encompasses a variety of health care practices evolved to maintain and restore health 
            by the prevention and treatment of illness. Contemporary medicine applies biomedical 
            sciences, biomedical research, genetics, and medical technology to diagnose, treat, 
            and prevent injury and disease, typically through pharmaceuticals or surgery, but 
            also through therapies as diverse as psychotherapy, external splints and traction, 
            medical devices, biologics, and ionizing radiation, amongst others. Medicine has 
            been practiced since prehistoric times, during most of which it was an art (an area 
            of skill and knowledge) frequently having connections to the religious and 
            philosophical beliefs of local culture.
            """
        
        # Use the QA pipeline to get an answer
        result = self.qa_pipeline(question=question, context=context)
        
        # If confidence is low, provide a fallback response
        if result['score'] < 0.1:
            return "I don't have enough information to answer that question accurately. Please consult with a medical professional for specific medical advice."
        
        return result['answer']
    
    def translate_text(self, text, source_lang="en", target_lang="fr"):
        """Translate text from source language to target language"""
        # Currently using a simple T5 model that only supports English to French
        # In a production system, you would use a more comprehensive model or an API
        
        if source_lang == "en" and target_lang == "fr":
            result = self.translator(text)[0]['translation_text']
            return result
        
        # Fallback for unsupported language pairs
        return "Translation for this language pair is not supported yet."

# Singleton instance
llm_service = LLMService()