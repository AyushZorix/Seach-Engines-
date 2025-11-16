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

    def generate_questions(self, text, iteration=1):
        """Agent 1: Generate initial questions from the text."""
        print(f"\n" + "🤖" * 25)
        print(f"🤖 AGENT 1: QUESTION GENERATOR (Iteration {iteration})")
        print("🤖" * 25)
        
        # Use a shorter version of text if it's too long
        if len(text) > 4000:
            text_chunk = text[:4000] + "... [text truncated for efficiency]"
        else:
            text_chunk = text
            
        prompt = f"""
        TASK: Generate 10 clear, specific questions that can be directly answered from the provided text.
        
        TEXT:
        {text_chunk}
        
        REQUIREMENTS:
        - Create exactly 10 distinct questions
        - Questions must be answerable using ONLY information from the text
        - Cover different aspects of the content
        - Make questions specific and concrete
        - Avoid questions that require external knowledge
        - Focus on factual information present in the text
        
        FORMAT: Return only a numbered list:
        1. First question?
        2. Second question?
        3. Third question?
        ...
        10. Tenth question?
        """
        
        result = self.get_llm_response(prompt, max_tokens=1000)
        if result:
            print(f"🎯 AGENT 1 COMPLETED: 10 questions generated successfully!")
        else:
            print("❌ AGENT 1 FAILED: Could not generate questions")
        return result

    def critique_questions(self, text, questions, iteration=1):
        """Agent 2: Critique whether the questions are answerable from the text."""
        print(f"\n" + "🧐" * 25)
        print(f"🧐 AGENT 2: CRITIQUE AGENT (Iteration {iteration})")
        print("🧐" * 25)
        
        if not questions:
            print("❌ No questions to critique - skipping Agent 2")
            return None, 0

        # Use a shorter version of text if it's too long
        if len(text) > 3000:
            text_chunk = text[:3000] + "... [text truncated]"
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
        
        After analyzing all questions, provide a summary at the end:
        SUMMARY: [X]/10 questions are answerable
        
        EXAMPLE:
        1. Yes - The text explicitly mentions the main character's age.
        2. No - The text doesn't provide information about the location.
        ...
        SUMMARY: 3/10 questions are answerable
        
        Now analyze the 10 questions above:
        """
        
        result = self.get_llm_response(prompt, max_tokens=800)
        
        # Count how many questions are answerable
        answerable_count = 0
        if result:
            # Simple heuristic to count "Yes" responses
            yes_count = result.lower().count('yes -')
            no_count = result.lower().count('no -')
            answerable_count = yes_count
            
            print(f"🎯 AGENT 2 COMPLETED: {answerable_count}/10 questions are answerable")
            
            # Check if we have the summary in the response
            if "SUMMARY:" in result:
                summary_line = [line for line in result.split('\n') if "SUMMARY:" in line]
                if summary_line:
                    print(f"📋 {summary_line[0]}")
        else:
            print("❌ AGENT 2 FAILED: Could not generate critique")
            
        return result, answerable_count

    def refine_questions_based_on_critique(self, text, previous_questions, critique, iteration=1):
        """Agent 1 (again): Refine questions based on critique feedback."""
        print(f"\n" + "🔄" * 25)
        print(f"🔄 AGENT 1: REFINING QUESTIONS (Iteration {iteration})")
        print("🔄" * 25)
        
        if not critique:
            print("❌ No critique available - cannot refine questions")
            return None

        # Use a shorter version of text if it's too long
        if len(text) > 4000:
            text_chunk = text[:4000] + "... [text truncated]"
        else:
            text_chunk = text
            
        prompt = f"""
        TASK: Based on the previous critique, generate IMPROVED questions that address the issues found.
        
        ORIGINAL TEXT:
        {text_chunk}
        
        PREVIOUS QUESTIONS AND CRITIQUE:
        {critique}
        
        REQUIREMENTS:
        - Generate exactly 10 NEW questions
        - Focus on creating questions that WILL be answerable from the text
        - Learn from the previous critique about what types of questions work
        - Ensure questions are specific and directly reference information in the text
        - Cover different aspects of the content
        
        FORMAT: Return only a numbered list of 10 improved questions:
        1. First improved question?
        2. Second improved question?
        ...
        10. Tenth improved question?
        """
        
        result = self.get_llm_response(prompt, max_tokens=1000)
        if result:
            print("🎯 AGENT 1 REFINED: New set of 10 questions generated!")
        else:
            print("❌ AGENT 1 FAILED: Could not refine questions")
        return result

    def analyze_text(self, text, max_iterations=11):
        """Main analysis function with 2-agent iterative workflow."""
        if not text.strip():
            return {"error": "No text provided"}

        print("- STARTING 2-AGENT ITERATIVE ANALYSIS WORKFLOW")
        print("=" * 60)
        print(f"- Target: Generate 10 validated questions")
        print(f"- Maximum iterations: {max_iterations}")
        print("=" * 60)

        all_results = []
        current_questions = None
        current_critique = None
        answerable_count = 0
        iteration = 1

        while iteration <= max_iterations:
            print(f"\n" + "🔄" * 20)
            print(f"🔄 ITERATION {iteration}")
            print("🔄" * 20)
            
            # Agent 1: Generate or refine questions
            if iteration == 1:
                current_questions = self.generate_questions(text, iteration)
            else:
                current_questions = self.refine_questions_based_on_critique(
                    text, current_questions, current_critique, iteration
                )
            
            if not current_questions:
                print("❌ Failed to generate questions - stopping iteration")
                break
                
            time.sleep(2)

            # Agent 2: Critique questions
            current_critique, current_answerable_count = self.critique_questions(text, current_questions, iteration)
            time.sleep(2)
            
            # Store results for this iteration
            iteration_result = {
                "iteration": iteration,
                "questions": current_questions,
                "critique": current_critique,
                "answerable_count": current_answerable_count
            }
            all_results.append(iteration_result)
            
            print(f"\n📊 Iteration {iteration} Results: {current_answerable_count}/10 questions answerable")
            
            # Check if we've achieved our goal
            if current_answerable_count >= 10:
                print("🎉 SUCCESS: All 10 questions are answerable!")
                break
            elif iteration == max_iterations:
                print("⚠️ Maximum iterations reached with {current_answerable_count}/10 answerable questions")
            else:
                print(f"🔄 Continuing to iteration {iteration + 1} to improve questions...")
            
            iteration += 1
            time.sleep(1)

        # Find the best iteration
        best_iteration = max(all_results, key=lambda x: x["answerable_count"]) if all_results else None
        
        return {
            "all_iterations": all_results,
            "best_iteration": best_iteration,
            "final_questions": best_iteration["questions"] if best_iteration else None,
            "final_critique": best_iteration["critique"] if best_iteration else None,
            "final_answerable_count": best_iteration["answerable_count"] if best_iteration else 0,
            "total_iterations": len(all_results)
        }

    def display_results(self, result):
        """Display all agent outputs in a clean format"""
        print("\n" + "📊" * 25)
        print("📊 FINAL RESULTS SUMMARY")
        print("📊" * 25)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print(f"\n🔄 Total Iterations: {result['total_iterations']}")
        print(f"✅ Final Answerable Questions: {result['final_answerable_count']}/10")
        
        # Show iteration summary
        print(f"\n" + "📈" * 20)
        print("📈 ITERATION PROGRESS")
        print("📈" * 20)
        for iter_data in result['all_iterations']:
            print(f"Iteration {iter_data['iteration']}: {iter_data['answerable_count']}/10 answerable")

        # Show best iteration details
        if result['best_iteration']:
            best = result['best_iteration']
            print(f"\n" + "🏆" * 20)
            print(f"🏆 BEST ITERATION: {best['iteration']}")
            print("🏆" * 20)
            
            print("\n" + "🔹" * 50)
            print("🤖 FINAL QUESTIONS")
            print("🔹" * 50)
            print(best['questions'])
            
            print("\n" + "🔸" * 50)
            print("🧐 FINAL CRITIQUE")
            print("🔸" * 50)
            print(best['critique'])
        else:
            print("\n❌ No successful iterations completed")

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
    
    print("🔧 Initializing Iterative Text Analysis System...")
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
    
    print("\n🚀 Starting 2-Agent Iterative Analysis Workflow...")
    
    # Analyze with maximum 3 iterations
    result = analyzer.analyze_text(text, max_iterations=11)
    analyzer.display_results(result)
    
    # Save to file
    try:
        timestamp = time.strftime("%H%M%S")
        filename = f"iterative_analysis_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("2-AGENT ITERATIVE TEXT ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Iterations: {result['total_iterations']}\n")
            f.write(f"Final Answerable Questions: {result['final_answerable_count']}/10\n\n")
            
            # Write all iterations
            for i, iter_data in enumerate(result['all_iterations']):
                f.write(f"ITERATION {iter_data['iteration']} ({iter_data['answerable_count']}/10 answerable)\n")
                f.write("-" * 40 + "\n")
                f.write("QUESTIONS:\n")
                f.write(iter_data.get('questions', 'N/A') + "\n\n")
                f.write("CRITIQUE:\n")
                f.write(iter_data.get('critique', 'N/A') + "\n")
                f.write("\n" + "="*50 + "\n\n")
            
            # Write best iteration separately
            if result['best_iteration']:
                f.write("BEST ITERATION:\n")
                f.write("=" * 30 + "\n")
                f.write("FINAL QUESTIONS:\n")
                f.write(result['best_iteration'].get('questions', 'N/A') + "\n\n")
                f.write("FINAL CRITIQUE:\n")
                f.write(result['best_iteration'].get('critique', 'N/A') + "\n")
        
        print(f"💾 Results saved to {filename}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

if __name__ == "__main__":
    main()