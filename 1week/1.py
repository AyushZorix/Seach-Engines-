import requests
import json
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv
import time

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextAnalyzer:
    def __init__(self, api_key, model="gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
    def get_llm_response(self, prompt, max_tokens=1024):
        """Get response from Google Gemini API"""
        headers = {
            "Content-Type": "application/json",
        }
        
        # Gemini has a different payload structure
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7
            }
        }
        
        # The API key is passed as a query parameter
        params = {"key": self.api_key}
        
        retries = 3
        for i in range(retries):
            try:
                response = requests.post(self.base_url, headers=headers, params=params, json=data, timeout=45)
                response.raise_for_status()
                
                # Check for a valid response
                response_json = response.json()
                if "candidates" in response_json and response_json["candidates"]:
                    return response_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    print(f"API Warning: Received an unexpected response on attempt {i+1}/{retries}: {response_json}")
                    
            except requests.exceptions.RequestException as e:
                print(f"API Error on attempt {i+1}/{retries}: {e}")
            except KeyError:
                print(f"Unexpected response format from API on attempt {i+1}/{retries}")

            # Wait before retrying
            if i < retries - 1:
                time.sleep(2 * (i + 1))  # Exponential backoff: 2s, 4s

        print("API calls failed after multiple retries.")
        return None
    
    def extract_significant_words(self, text, top_n=10):
        """Extract significant words using frequency analysis"""
        # Tokenize and clean text
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words('english'))
        filtered_words = [
            word for word in words 
            if word.isalpha() and word not in stop_words and len(word) > 2
        ]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        return word_freq.most_common(top_n)
    
    def analyze_text(self, text):
        """Main function to analyze text and get questions + significant words"""
        if not text.strip():
            return {"error": "Please provide non-empty text"}
        
        # Prompt for question identification
        question_prompt = f"""
        Analyze the following text and identify 10 specific questions that can be answered based on the information provided. 
        Format your response as a numbered list of questions.
        
        Text: {text}
        
        Questions that can be answered:
        """
        
        print("Analyzing text with LLM...")
        questions_response = self.get_llm_response(question_prompt)
        
        print("Extracting significant words...")
        significant_words = self.extract_significant_words(text)
        
        return {
            "answerable_questions": questions_response,
            "significant_words": significant_words,
            "word_count": len(text.split())
        }
    
    def display_results(self, analysis_result):
        """Display the analysis results in a formatted way"""
        print("\n" + "="*60)
        print("TEXT ANALYSIS RESULTS")
        print("="*60)
        
        if "error" in analysis_result:
            print(f"Error: {analysis_result['error']}")
            return
        
        print(f"\n📊 Word Count: {analysis_result['word_count']}")
        
        print(f"\n❓ ANSWERABLE QUESTIONS:")
        print("-" * 40)
        if analysis_result['answerable_questions']:
            print(analysis_result['answerable_questions'])
        else:
            print("No questions could be generated.")
        
        print(f"\n🔑 MOST SIGNIFICANT WORDS:")
        print("-" * 40)
        for word, count in analysis_result['significant_words']:
            print(f"{word}: {count} occurrences")
        
        print("\n" + "="*60)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("API key not found. Make sure you have a .env file with GEMINI_API_KEY set.")
        return
    
    # Initialize analyzer
    analyzer = TextAnalyzer(api_key)
    
    # Read text from corpus.txt
    try:
        with open("../corpus.txt", "r", encoding="utf-8") as f:
            user_text = f.read()
    except FileNotFoundError:
        print("Error: corpus.txt not found in the parent directory.")
        return

    if not user_text.strip():
        print("corpus.txt is empty. Please provide some text to analyze.")
        return
    
    print("Analyzing text from corpus.txt...")
    # Analyze the text
    result = analyzer.analyze_text(user_text)
    analyzer.display_results(result)

if __name__ == "__main__":
    main()