import requests
import json
import os
from dotenv import load_dotenv
import time

class RobustTextAnalyzer:
    def __init__(self, api_key, model="gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
    
    def get_llm_response(self, prompt, max_tokens=800):
        """Robust LLM call with better error handling"""
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.4
            }
        }
        
        params = {"key": self.api_key}
        retries = 3
        
        for i in range(retries):
            try:
                print(f"  📡 Calling API (attempt {i+1}/{retries})...")
                response = requests.post(self.base_url, headers=headers, params=params, json=data, timeout=60)
                response.raise_for_status()
                
                response_json = response.json()
                if "candidates" in response_json and response_json["candidates"]:
                    candidate = response_json["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                        result = candidate["content"]["parts"][0]["text"]
                        print(f"  ✅ Success: Got {len(result)} characters")
                        return result.strip()
                    elif "finishReason" in candidate:
                        print(f"  ⚠️ API finished with reason: {candidate['finishReason']}")
                        if candidate['finishReason'] == "MAX_TOKENS":
                            # Try with more tokens
                            data["generationConfig"]["maxOutputTokens"] = max_tokens * 2
                            continue
                
                print(f"  ❌ Unexpected response structure on attempt {i+1}")
                print("Response:", json.dumps(response_json, indent=2)[:500] + "...")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ❌ API Error: {e}")
            except (KeyError, IndexError) as e:
                print(f"  ❌ Response parsing error: {e}")

            if i < retries - 1:
                wait_time = 3 * (i + 1)
                print(f"  ⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        return None

    def generate_questions(self, text):
        """Agent 1: Generate initial questions from the text."""
        print("\n" + "🤖" * 20)
        print("🤖 AGENT 1: QUESTION GENERATOR")
        print("🤖" * 20)
        
        # Use a shorter version of text if it's too long
        if len(text) > 3000:
            text_chunk = text[:3000] + "... [text truncated for efficiency]"
        else:
            text_chunk = text
            
        prompt = f"""
        TASK: Generate 5 clear, specific questions that can be directly answered from the provided text.
        
        TEXT:
        {text_chunk}
        
        REQUIREMENTS:
        - Create 5 distinct questions
        - Questions must be answerable using ONLY information from the text
        - Cover different aspects of the content
        - Make questions specific and concrete
        
        FORMAT: Return only a numbered list:
        1. First question?
        2. Second question?
        3. Third question?
        4. Fourth question?
        5. Fifth question?
        """
        
        result = self.get_llm_response(prompt, max_tokens=800)
        if result:
            print("🎯 AGENT 1 COMPLETED: Questions generated successfully!")
        else:
            print("❌ AGENT 1 FAILED: Could not generate questions")
        return result

    def critique_questions(self, text, questions):
        """Agent 2: Critique whether the questions are answerable from the text."""
        print("\n" + "🧐" * 20)
        print("🧐 AGENT 2: CRITIQUE AGENT")
        print("🧐" * 20)
        
        if not questions:
            print("❌ No questions to critique - skipping Agent 2")
            return None

        # Use a shorter version of text if it's too long
        if len(text) > 2000:
            text_chunk = text[:2000] + "... [text truncated]"
        else:
            text_chunk = text
            
        prompt = f"""
        TASK: Review each question and determine if it can be answered DIRECTLY from the text.
        Be strict - only mark as "Yes" if the text contains explicit information to answer it.
        
        TEXT:
        {text_chunk}
        
        QUESTIONS TO CRITIQUE:
        {questions}
        
        FORMAT: For each question, provide:
        [Question Number]. [Yes/No] - [Brief reason explaining why]
        
        EXAMPLE:
        1. Yes - The text explicitly mentions the main character's age.
        2. No - The text doesn't provide information about the location.
        
        Now analyze the questions above:
        """
        
        result = self.get_llm_response(prompt, max_tokens=600)
        if result:
            print("🎯 AGENT 2 COMPLETED: Critique provided!")
        else:
            print("❌ AGENT 2 FAILED: Could not generate critique")
        return result

    def generate_validated_questions(self, text, questions, critique):
        """Agent 3: Generate a final list of verified, answerable questions."""
        print("\n" + "✅" * 20)
        print("✅ AGENT 3: VALIDATION AGENT")
        print("✅" * 20)
        
        if not questions or not critique:
            print("❌ Missing input - skipping Agent 3")
            return "Cannot generate validated questions due to missing input from previous agents."

        prompt = f"""
        TASK: Create a final list of validated questions that passed the critique.
        Only include questions that were marked as "Yes" (answerable) in the critique.
        
        ORIGINAL QUESTIONS:
        {questions}
        
        CRITIQUE ANALYSIS:
        {critique}
        
        INSTRUCTIONS:
        1. Extract ONLY the questions that received "Yes" in the critique
        2. Keep them in their original numbered order
        3. If all questions were rejected, state: "No questions were validated as answerable from the text."
        4. If some questions passed, present them as a clean numbered list
        
        FINAL VALIDATED QUESTIONS:
        """
        
        result = self.get_llm_response(prompt, max_tokens=500)
        if result:
            print("🎯 AGENT 3 COMPLETED: Final questions validated!")
        else:
            print("❌ AGENT 3 FAILED: Could not generate final questions")
        return result

    def analyze_text(self, text):
        """Main analysis function with the 3-agent workflow."""
        if not text.strip():
            return {"error": "No text provided"}

        print("🚀 STARTING 3-AGENT ANALYSIS WORKFLOW")
        print("=" * 50)

        # Agent 1: Question Generator
        initial_questions = self.generate_questions(text)
        time.sleep(2)

        # Agent 2: Critique Agent
        critique = self.critique_questions(text, initial_questions)
        time.sleep(2)

        # Agent 3: Validator Agent
        validated_questions = self.generate_validated_questions(text, initial_questions, critique)
        time.sleep(1)

        return {
            "agent1_initial_questions": initial_questions,
            "agent2_critique": critique,
            "agent3_validated_questions": validated_questions,
        }

    def display_results(self, result):
        """Display all agent outputs in a clean format"""
        print("\n" + "📊" * 20)
        print("📊 FINAL RESULTS")
        print("📊" * 20)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print("\n" + "🔹" * 50)
        print("🤖 AGENT 1 - INITIAL QUESTIONS")
        print("🔹" * 50)
        if result.get('agent1_initial_questions'):
            print(result['agent1_initial_questions'])
        else:
            print("❌ No questions generated")

        print("\n" + "🔸" * 50)
        print("🧐 AGENT 2 - CRITIQUE ANALYSIS")
        print("🔸" * 50)
        if result.get('agent2_critique'):
            print(result['agent2_critique'])
        else:
            print("❌ No critique generated")

        print("\n" + "✅" * 50)
        print("✅ AGENT 3 - VALIDATED QUESTIONS")
        print("✅" * 50)
        if result.get('agent3_validated_questions'):
            print(result['agent3_validated_questions'])
        else:
            print("❌ No validated questions")

        print("\n" + "🎉" * 20)
        print("🎉 WORKFLOW COMPLETE")
        print("🎉" * 20)

def main():
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file")
        print("Please make sure your .env file contains: GEMINI_API_KEY=your_actual_key_here")
        return
    
    print("🔧 Initializing Text Analysis System...")
    analyzer = RobustTextAnalyzer(api_key)
    
    # Read from corpus.txt
    try:
        with open("../corpus.txt", "r", encoding="utf-8") as f:
            text = f.read().strip()
        print(f"📖 Loaded corpus.txt ({len(text)} characters)")
    except FileNotFoundError:
        print("❌ Error: corpus.txt not found in parent directory")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    if not text:
        print("❌ Error: corpus.txt is empty")
        return
    
    print("\n🚀 Starting 3-Agent Analysis Workflow...")
    
    # Analyze
    result = analyzer.analyze_text(text)
    analyzer.display_results(result)
    
    # Save to file
    try:
        timestamp = time.strftime("%H%M%S")
        filename = f"analysis_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("3-AGENT TEXT ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("AGENT 1 - INITIAL QUESTIONS:\n")
            f.write("-" * 30 + "\n")
            f.write(result.get('agent1_initial_questions', 'N/A') + "\n\n")
            
            f.write("AGENT 2 - CRITIQUE:\n")
            f.write("-" * 30 + "\n")
            f.write(result.get('agent2_critique', 'N/A') + "\n\n")
            
            f.write("AGENT 3 - VALIDATED QUESTIONS:\n")
            f.write("-" * 30 + "\n")
            f.write(result.get('agent3_validated_questions', 'N/A') + "\n")
        
        print(f"💾 Results saved to {filename}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

if __name__ == "__main__":
    main()