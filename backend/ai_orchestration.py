import os
import json
import networkx as nx
from typing import Dict, List, Optional, Tuple
import requests
from google import genai
from dataclasses import dataclass
import sys

# Prefer absolute package import; fallback to sys.path tweak for direct runs
try:
    from gptscripts.key_manager import APIKeyManager
    KEY_MANAGER_AVAILABLE = True
except ImportError:
    # Add the gptscripts folder to the path for non-package contexts
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gptscripts'))
    try:
        from gptscripts.key_manager import APIKeyManager
        KEY_MANAGER_AVAILABLE = True
    except ImportError:
        KEY_MANAGER_AVAILABLE = False
        print("Warning: Key manager not available, using environment variables directly")

@dataclass
class UserProfile:
    background: str
    goals: str
    content_type: str
    experience_years: str
    preferred_format: str

@dataclass
class LearningResource:
    title: str
    summary: str
    link: str
    labels: List[str]
    content_type: str
    source: str
    difficulty_level: str
    prerequisites: List[str]
    credibility_score: int

class AIOrchestration:
    def __init__(self):
        # Initialize key managers if available
        self.gemini_manager = None
        self.perplexity_manager = None
        
        if KEY_MANAGER_AVAILABLE:
            try:
                self.gemini_manager = APIKeyManager("GEMINI_API_KEYS")
                print("✅ Gemini API key manager initialized successfully")
            except ValueError as e:
                print(f"⚠️ Gemini key manager failed: {e}")
                self.gemini_manager = None
                
            try:
                self.perplexity_manager = APIKeyManager("PERPLEXITY_API_KEYS")
                print("✅ Perplexity API key manager initialized successfully")
            except ValueError as e:
                print(f"⚠️ Perplexity key manager failed: {e}")
                self.perplexity_manager = None
        else:
            # Fallback to direct environment variables
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
            if self.gemini_api_key:
                print("✅ Using Gemini API key from environment variable")
            if self.perplexity_api_key:
                print("✅ Using Perplexity API key from environment variable")
        
        # Gemini SDK configuration (prefer SDK over raw HTTP)
        self.gemini_model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.perplexity_base_url = "https://api.perplexity.ai/chat/completions"
        
    def _get_gemini_key(self) -> Optional[str]:
        """Get Gemini API key with rotation support"""
        if self.gemini_manager:
            return self.gemini_manager.get_key()
        return self.gemini_api_key if hasattr(self, 'gemini_api_key') else None
        
    def _get_perplexity_key(self) -> Optional[str]:
        """Get Perplexity API key with rotation support"""
        if self.perplexity_manager:
            return self.perplexity_manager.get_key()
        return self.perplexity_api_key if hasattr(self, 'perplexity_api_key') else None
        
    def _rotate_gemini_key(self) -> Optional[str]:
        """Rotate to next Gemini API key"""
        if self.gemini_manager:
            return self.gemini_manager.rotate_key()
        return None
        
    def _rotate_perplexity_key(self) -> Optional[str]:
        """Rotate to next Perplexity API key"""
        if self.perplexity_manager:
            return self.perplexity_manager.rotate_key()
        return None

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
        # Call Sonar API to fill gaps
        sonar_results = call_sonar_api(query, sequenced_path['missingPrerequisites'])
        # Integrate sonar results into the path
        sequenced_path['additionalResources'] = sonar_results
    
    return sequenced_path

def call_gemini_sequence(query: str, user_profile: Dict, resources: List[Dict]) -> Dict:
    """
    Call Gemini API to sequence resources into learning phases.
    """
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    
    # Prepare resources as JSON string
    resources_json = json.dumps(resources, indent=2)
    
    prompt = f"""You are an expert curriculum designer.
Learner profile: {json.dumps(user_profile)}
Query: {query}
Resources: {resources_json}

Organize into: Remedial, Beginner, Intermediate, Advanced.
Include 'missingPrerequisites' if required.
Output only valid JSON."""
    
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt
    )
    
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
           
    
    def _process_search_results(self, search_results: List[Dict]) -> List[LearningResource]:
        """Convert search results to structured resources"""
        resources = []
        for result in search_results:
            resource = LearningResource(
                title=result.get('title', ''),
                summary=result.get('summary', ''),
                link=result.get('link', ''),
                labels=result.get('labels', []),
                content_type=result.get('content_type', 'video'),
                source=result.get('source', 'Unknown'),
                difficulty_level=result.get('difficulty_level', 'Beginner'),
                prerequisites=result.get('prerequisites', []),
                credibility_score=result.get('credibility_score', 50)
            )
            resources.append(resource)
        return resources
    
    def _topological_sort_resources(self, resources: List[LearningResource]) -> List[LearningResource]:
        """Use NetworkX to create dependency graph and sort topologically"""
        G = nx.DiGraph()
        
        # Add nodes
        for resource in resources:
            G.add_node(resource.title, resource=resource)
        
        # Add edges based on prerequisites
        for resource in resources:
            for prereq in resource.prerequisites:
                # Find resources that match prerequisites
                for other_resource in resources:
                    if (other_resource.title != resource.title and 
                        prereq.lower() in other_resource.title.lower()):
                        G.add_edge(other_resource.title, resource.title)
        
        try:
            # Perform topological sort
            sorted_nodes = list(nx.topological_sort(G))
            ordered_resources = []
            
            for node in sorted_nodes:
                resource = G.nodes[node]['resource']
                ordered_resources.append(resource)
            
            # Add any remaining resources that weren't in the sort
            titles_in_order = set(n for n in sorted_nodes)
            for resource in resources:
                if resource.title not in titles_in_order:
                    ordered_resources.append(resource)
                    
            return ordered_resources
            
        except Exception as exc:
            # Handle cycles (NetworkXUnfeasible) and runtime iteration errors
            print(f"Warning: Topological sort failed ({exc}); using original order")
            return resources
    
    def _adjust_for_user_profile(self, resources: List[LearningResource], 
                                user_profile: UserProfile) -> List[LearningResource]:
        """Adjust resource ordering and selection based on user profile"""
        adjusted_resources = []
        
        # Filter by content type preference
        if user_profile.content_type != 'mixed':
            preferred_resources = [
                r for r in resources 
                if r.content_type.lower() == user_profile.content_type.lower()
            ]
            if preferred_resources:
                adjusted_resources.extend(preferred_resources)
        
        # Add remaining resources
        for resource in resources:
            if resource not in adjusted_resources:
                adjusted_resources.append(resource)
        
        # Adjust difficulty based on background
        if user_profile.background == 'beginner':
            # Prioritize beginner resources
            adjusted_resources.sort(key=lambda x: self._difficulty_score(x.difficulty_level))
        elif user_profile.background == 'advanced':
            # Prioritize advanced resources
            adjusted_resources.sort(key=lambda x: -self._difficulty_score(x.difficulty_level))
        
        return adjusted_resources
    
    def _difficulty_score(self, difficulty: str) -> int:
        """Convert difficulty to numeric score for sorting"""
        difficulty_map = {
            'Remedial': 0,
            'Beginner': 1,
            'Intermediate': 2,
            'Advanced': 3,
            'Expert': 4
        }
        return difficulty_map.get(difficulty, 1)
    
    def _call_gemini_pro(self, query: str, resources: List[LearningResource], 
                         user_profile: UserProfile) -> Dict:
        """Call Gemini using official SDK (model loaded by name)"""
        gemini_key = self._get_gemini_key()
        if not gemini_key:
            raise Exception("Gemini API key not available")

        # Lazy import to avoid hard dependency if not used
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise Exception(f"Gemini SDK not installed: {exc}")

        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(self.gemini_model_name)

        prompt = self._create_gemini_prompt(query, resources, user_profile)
        response = model.generate_content(prompt)

        # The SDK returns an object with .text and/or candidates
        result: Dict = {
            'candidates': [
                {
                    'content': {
                        'parts': [{'text': getattr(response, 'text', '') or ''}]
                    }
                }
            ]
        }
        return self._parse_gemini_response(result, resources)
    
    def _create_gemini_prompt(self, query: str, resources: List[LearningResource], 
                             user_profile: UserProfile) -> str:
        """Create structured prompt for Gemini Pro"""
        prompt = f"""
You are an expert learning path designer. Create a structured learning path for the query: "{query}"

User Profile:
- Background: {user_profile.background}
- Goals: {user_profile.goals}
- Content Preference: {user_profile.content_type}
- Experience: {user_profile.experience_years} years

Available Resources ({len(resources)} total):
"""
        
        for i, resource in enumerate(resources[:20]):  # Limit to first 20 for prompt
            prompt += f"""
{i+1}. {resource.title}
   - Difficulty: {resource.difficulty_level}
   - Type: {resource.content_type}
   - Prerequisites: {', '.join(resource.prerequisites) if resource.prerequisites else 'None'}
   - Summary: {resource.summary[:100]}...
"""
        
        prompt += """
Please organize these resources into a 4-phase learning path:

Phase 1 (Remedial): Foundation concepts and prerequisites
Phase 2 (Beginner): Core concepts and fundamental understanding  
Phase 3 (Intermediate): Advanced concepts and practical applications
Phase 4 (Advanced): Expert-level topics and cutting-edge applications

For each phase, select 3-8 most appropriate resources and provide:
- Phase name and description
- Selected resources with brief reasoning
- Estimated duration for that phase

Return the response as a JSON object with this structure:
{
  "phases": [
    {
      "phase": "Remedial",
      "description": "...",
      "resources": [resource_indices],
      "phase_number": 1,
      "estimated_duration": "..."
    }
  ],
  "total_resources": total_count,
  "estimated_duration": "overall_duration",
  "prerequisites_met": true/false,
  "generation_method": "gemini_pro"
}

IMPORTANT: Return ONLY valid JSON, no additional text or explanations.
"""
        return prompt
    
    def _parse_gemini_response(self, gemini_result: Dict, resources: List[LearningResource]) -> Dict:
        """Parse Gemini Pro response and map to resources"""
        try:
            # Extract text from Gemini response
            if 'candidates' in gemini_result and len(gemini_result['candidates']) > 0:
                content = gemini_result['candidates'][0]['content']
                if 'parts' in content and len(content['parts']) > 0:
                    text = content['parts'][0]['text']
                    
                    # Try to extract JSON from the response
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_str = text[json_start:json_end]
                        parsed = json.loads(json_str)
                        
                        # Map resource indices to actual resources
                        for phase in parsed.get('phases', []):
                            resource_indices = phase.get('resources', [])
                            phase_resources = []
                            
                            for idx in resource_indices:
                                if 0 <= idx - 1 < len(resources):
                                    phase_resources.append(resources[idx - 1])
                            
                            phase['resources'] = phase_resources
                        
                        return parsed
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
        
        # Fallback if parsing fails
        return self._fallback_learning_path_generation("", resources, None)
    
    def _call_perplexity_pro(self, query: str, missing_prerequisites: List[str]) -> List[Dict]:
        """Call Perplexity Pro to fetch external resources for missing prerequisites"""
        perplexity_key = self._get_perplexity_key()
        if not perplexity_key:
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {perplexity_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
Find high-quality learning resources for these missing prerequisites: {', '.join(missing_prerequisites)}
Query context: {query}

For each prerequisite, provide:
- Resource title
- URL/link
- Brief description
- Content type (video, article, course, etc.)
- Difficulty level
- Source/platform
"""
            
            data = {
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.perplexity_base_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_perplexity_response(result)
            else:
                print(f"Perplexity API failed: {response.status_code}")
                # Try rotating the key and retry once
                rotated_key = self._rotate_perplexity_key()
                if rotated_key and rotated_key != perplexity_key:
                    try:
                        headers["Authorization"] = f"Bearer {rotated_key}"
                        response = requests.post(
                            self.perplexity_base_url,
                            headers=headers,
                            json=data
                        )
                        if response.status_code == 200:
                            result = response.json()
                            print("✅ Successfully used rotated Perplexity API key")
                            return self._parse_perplexity_response(result)
                    except Exception as retry_e:
                        print(f"Retry with rotated key also failed: {retry_e}")
                return []
                
        except Exception as e:
            print(f"Error calling Perplexity Pro: {e}")
            return []
    
    def _parse_perplexity_response(self, perplexity_result: Dict) -> List[Dict]:
        """Parse Perplexity Pro response for external resources"""
        try:
            if 'choices' in perplexity_result and len(perplexity_result['choices']) > 0:
                content = perplexity_result['choices'][0]['message']['content']
                
                # Simple parsing - in production, you'd want more sophisticated parsing
                resources = []
                lines = content.split('\n')
                
                current_resource = {}
                for line in lines:
                    if line.strip():
                        if 'title:' in line.lower():
                            if current_resource:
                                resources.append(current_resource)
                            current_resource = {'title': line.split(':', 1)[1].strip()}
                        elif 'url:' in line.lower() or 'link:' in line.lower():
                            current_resource['link'] = line.split(':', 1)[1].strip()
                        elif 'description:' in line.lower():
                            current_resource['summary'] = line.split(':', 1)[1].strip()
                        elif 'type:' in line.lower():
                            current_resource['content_type'] = line.split(':', 1)[1].strip()
                        elif 'difficulty:' in line.lower():
                            current_resource['difficulty_level'] = line.split(':', 1)[1].strip()
                        elif 'source:' in line.lower():
                            current_resource['source'] = line.split(':', 1)[1].strip()
                
                if current_resource:
                    resources.append(current_resource)
                
                return resources
                
        except Exception as e:
            print(f"Error parsing Perplexity response: {e}")
        
        return []
    
    def _fallback_learning_path_generation(self, query: str, resources: List[LearningResource], 
                                         user_profile: UserProfile) -> Dict:
        """Fallback to rule-based learning path generation"""
        # Group resources by difficulty
        difficulty_groups = {
            'Remedial': [],
            'Beginner': [],
            'Intermediate': [],
            'Advanced': []
        }
        
        for resource in resources:
            difficulty = resource.difficulty_level
            if difficulty in difficulty_groups:
                difficulty_groups[difficulty].append(resource)
            else:
                # Map unknown difficulties
                if 'basic' in difficulty.lower() or 'intro' in difficulty.lower():
                    difficulty_groups['Beginner'].append(resource)
                elif 'advanced' in difficulty.lower() or 'expert' in difficulty.lower():
                    difficulty_groups['Advanced'].append(resource)
                else:
                    difficulty_groups['Intermediate'].append(resource)
        
        # Build phases first
        phases = [
            {
                'phase': 'Remedial',
                'description': 'Foundation concepts and prerequisites',
                'resources': difficulty_groups['Remedial'][:5],
                'phase_number': 1,
                'estimated_duration': '2-3 weeks'
            },
            {
                'phase': 'Beginner',
                'description': 'Core concepts and fundamental understanding',
                'resources': difficulty_groups['Beginner'][:8],
                'phase_number': 2,
                'estimated_duration': '3-4 weeks'
            },
            {
                'phase': 'Intermediate',
                'description': 'Advanced concepts and practical applications',
                'resources': difficulty_groups['Intermediate'][:6],
                'phase_number': 3,
                'estimated_duration': '3-4 weeks'
            },
            {
                'phase': 'Advanced',
                'description': 'Expert-level topics and cutting-edge applications',
                'resources': difficulty_groups['Advanced'][:4],
                'phase_number': 4,
                'estimated_duration': '2-3 weeks'
            }
        ]
        
        # Build complete learning path
        learning_path = {
            'phases': phases,
            'total_resources': sum(len(phase['resources']) for phase in phases),
            'estimated_duration': '10-14 weeks',
            'prerequisites_met': len(difficulty_groups['Remedial']) > 0,
            'generation_method': 'fallback_topological_sort'
        }
        
        return learning_path
    
    def _emergency_fallback(self, query: str, search_results: List[Dict], 
                           user_profile: UserProfile) -> Dict:
        """Emergency fallback when all else fails"""
        return {
            'phases': [
                {
                    'phase': 'General',
                    'description': 'Learning resources for your query',
                    'resources': search_results[:10],
                    'phase_number': 1,
                    'estimated_duration': 'Varies'
                }
            ],
            'total_resources': min(10, len(search_results)),
            'estimated_duration': 'Varies',
            'prerequisites_met': False,
            'generation_method': 'emergency_fallback'
        }
    
    def fill_prerequisite_gaps(self, query: str, missing_prerequisites: List[str]) -> List[Dict]:
        """Fill gaps in prerequisites using Perplexity Pro"""
        if not missing_prerequisites:
            return []
        
        external_resources = self._call_perplexity_pro(query, missing_prerequisites)
        
        # Convert to standard format
        formatted_resources = []
        for resource in external_resources:
            formatted_resource = {
                'title': resource.get('title', ''),
                'summary': resource.get('summary', ''),
                'link': resource.get('link', ''),
                'labels': ['External Resource'],
                'content_type': resource.get('content_type', 'article'),
                'source': resource.get('source', 'External'),
                'difficulty_level': resource.get('difficulty_level', 'Beginner'),
                'prerequisites': [],
                'credibility_score': 70,
                'is_external': True
            }
            formatted_resources.append(formatted_resource)
        
        return formatted_resources
    
    def get_key_status(self) -> Dict:
        """Get the status of API keys and key managers"""
        status = {
            'key_manager_available': KEY_MANAGER_AVAILABLE,
            'gemini': {
                'manager_initialized': self.gemini_manager is not None,
                'key_available': self._get_gemini_key() is not None
            },
            'perplexity': {
                'manager_initialized': self.perplexity_manager is not None,
                'key_available': self._get_perplexity_key() is not None
            }
        }
        
        if self.gemini_manager:
            status['gemini']['total_keys'] = len(self.gemini_manager.keys)
            status['gemini']['current_key_index'] = list(self.gemini_manager.keys).index(self.gemini_manager.current_key) + 1
            
        if self.perplexity_manager:
            status['perplexity']['total_keys'] = len(self.perplexity_manager.keys)
            status['perplexity']['current_key_index'] = list(self.perplexity_manager.keys).index(self.perplexity_manager.current_key) + 1
            
        return status
