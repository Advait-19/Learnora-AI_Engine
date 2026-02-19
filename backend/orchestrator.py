import os
import json
from typing import Dict, List, Optional
import google.generativeai as genai
import requests
from ai_orchestration import AIOrchestration


def generate_learning_path(query: str, user_profile: Dict, search_results: List[Dict]) -> Dict:
    """
    Generate a learning path from FAISS search results.
    
    Args:
        query: The user's learning query
        user_profile: Dict containing user profile information
        search_results: List of dicts from FAISS search
    
    Returns:
        Structured JSON learning path
    """
    # Call Gemini to sequence the resources
    sequenced_path = call_gemini_sequence(query, user_profile, search_results)
    
    # Check if missing prerequisites
    if 'missingPrerequisites' in sequenced_path and sequenced_path['missingPrerequisites']:
        # Use AIOrchestration to fill prerequisite gaps
        ai_orchestrator = AIOrchestration()
        additional_resources = ai_orchestrator.fill_prerequisite_gaps(query, sequenced_path['missingPrerequisites'])
        # Merge into the path
        sequenced_path['additionalResources'] = additional_resources or []
    
    return sequenced_path


def call_gemini_sequence(query: str, user_profile: Dict, resources: List[Dict]) -> Dict:
    """
    Call Gemini API to sequence resources into learning phases.
    """
    # Use the same approach as ai_orchestration.py
    import google.generativeai as genai
    
    api_key = os.environ['GEMINI_API_KEY']
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # Prepare resources as JSON string
    resources_json = json.dumps(resources, indent=2)
    
    prompt = f"""You are an expert curriculum designer.
Learner profile: {json.dumps(user_profile)}
Query: {query}
Resources: {resources_json}

Organize into: Remedial, Beginner, Intermediate, Advanced.
Include 'missingPrerequisites' if required.
Output only valid JSON."""
    
    response = model.generate_content(prompt)
    
    # Parse the JSON response
    result_text = response.text.strip()
    
    # Extract JSON if wrapped in code blocks or extra text
    if result_text.startswith('```json'):
        result_text = result_text[7:]
    if result_text.endswith('```'):
        result_text = result_text[:-3]
    
    parsed_json = json.loads(result_text)
    return parsed_json


def call_sonar_api(query: str, missing_prerequisites: List[str]) -> List[Dict]:
    """
    Call Sonar API (Perplexity) to find resources for missing prerequisites.
    """
    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key:
        return []
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    prompt = f"""Find high-quality learning resources for these missing prerequisites: {', '.join(missing_prerequisites)}
Query context: {query}

For each prerequisite, provide:
- Resource title
- URL/link
- Brief description
- Content type (video, article, course, etc.)
- Difficulty level
- Source/platform

Return as a list of dictionaries in JSON format."""
    
    data = {
        'model': 'sonar-pro',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 1000
    }
    
    try:
        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Assume the response is JSON
            return json.loads(content)
        else:
            print(f"Sonar API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error calling Sonar API: {e}")
        return []


def generate_learning_path_stream(query: str, user_profile: Dict, search_results: List[Dict]):
    """
    Streaming version of generate_learning_path.
    Yields status messages and final result.
    """
    yield {"message": "Phase-wise learning path generating...", "type": "status"}
    
    path = generate_learning_path(query, user_profile, search_results)
    
    yield {"type": "result", "data": path}