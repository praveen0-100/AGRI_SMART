import os
import base64
import requests
import json
from PIL import Image
import io
from duckduckgo_search import DDGS

# Import Google GenAI SDK if available
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

class AIEngine:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Check if keys exist in environment
        if not self.gemini_key and os.getenv("GOOGLE_API_KEY"):
            self.gemini_key = os.getenv("GOOGLE_API_KEY")
            
        print(f"AI Engine loaded. Keys found - Gemini: {bool(self.gemini_key)}, OpenAI: {bool(self.openai_key)}, Anthropic: {bool(self.anthropic_key)}")

    def search_web(self, query: str, max_results: int = 5) -> str:
        """Search the web in real-time for agricultural query."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return "No search results found."
                
                search_text = []
                for idx, r in enumerate(results):
                    search_text.append(f"[{idx+1}] Source: {r.get('href')}\nTitle: {r.get('title')}\nSnippet: {r.get('body')}\n")
                return "\n".join(search_text)
        except Exception as e:
            print(f"Web search error for '{query}': {e}")
            return f"Web search failed: {str(e)}"

    def call_gemini(self, prompt: str, image_bytes: bytes = None, image_mime: str = "image/jpeg", system_instruction: str = None) -> str:
        """Call Google Gemini API using google-genai SDK."""
        if not HAS_GEMINI_SDK:
            raise ImportError("google-genai SDK is not installed.")
        
        client = genai.Client(api_key=self.gemini_key)
        
        # Build contents
        contents = []
        if image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
        contents.append(prompt)
        
        # Build config
        config_args = {}
        if system_instruction:
            config_args["system_instruction"] = system_instruction
            
        # We can use gemini-2.5-flash as it is fast and supports vision
        config = types.GenerateContentConfig(**config_args)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=config
        )
        return response.text

    def call_openai(self, prompt: str, image_bytes: bytes = None, image_mime: str = "image/jpeg", system_instruction: str = None) -> str:
        """Call OpenAI API using direct requests."""
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        user_content = []
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime};base64,{base64_image}"
                }
            })
            
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        messages.append({"role": "user", "content": user_content})
        
        payload = {
            "model": "gpt-4o-mini", # Cost-effective and highly intelligent
            "messages": messages,
            "max_tokens": 1000
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"]

    def call_anthropic(self, prompt: str, image_bytes: bytes = None, image_mime: str = "image/jpeg", system_instruction: str = None) -> str:
        """Call Anthropic API using direct requests."""
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        content_list = []
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            content_list.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_mime,
                    "data": base64_image
                }
            })
            
        content_list.append({
            "type": "text",
            "text": prompt
        })
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": content_list}]
        }
        
        if system_instruction:
            payload["system"] = system_instruction
            
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        return res_json["content"][0]["text"]

    def call_llm(self, prompt: str, image_bytes: bytes = None, image_mime: str = "image/jpeg", system_instruction: str = None) -> str:
        """Call the available LLM in priority: Gemini -> OpenAI -> Anthropic -> Fallback Rule System."""
        # Try Gemini
        if self.gemini_key and HAS_GEMINI_SDK:
            try:
                return self.call_gemini(prompt, image_bytes, image_mime, system_instruction)
            except Exception as e:
                print(f"Gemini call failed: {e}. Trying alternative LLM...")
                
        # Try OpenAI
        if self.openai_key:
            try:
                return self.call_openai(prompt, image_bytes, image_mime, system_instruction)
            except Exception as e:
                print(f"OpenAI call failed: {e}. Trying alternative LLM...")

        # Try Anthropic
        if self.anthropic_key:
            try:
                return self.call_anthropic(prompt, image_bytes, image_mime, system_instruction)
            except Exception as e:
                print(f"Anthropic call failed: {e}. Using offline fallback...")
                
        # If no API key or all failed, return an intelligent offline fallback response
        return self._offline_fallback(prompt, image_bytes is not None, system_instruction)

    def _offline_fallback(self, prompt: str, has_image: bool, system_instruction: str = None) -> str:
        """Rule-based text generation for local offline runs without API keys."""
        prompt_lower = prompt.lower()
        
        # If it's a soil analysis request
        if "soil" in prompt_lower or "ph" in prompt_lower or "nitrogen" in prompt_lower:
            return """
### [OFFLINE INTELLIGENCE FALLBACK] Soil Analysis & Crop Recommendation Report
*Note: This report is generated offline using built-in agricultural agronomy rules. For full dynamic insights, please supply a GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.*

#### 1. Soil Suitability & Analysis
- **pH Level Analysis**:
  - Ideal range: 6.0 to 7.5 for most cereals and pulses.
  - Highly acidic soils (<5.5) require lime application (calcium carbonate).
  - Alkaline soils (>8.0) require gypsum application to balance pH.
- **NPK Ratio Analysis**:
  - Nitrogen (N) is crucial for vegetative leaf growth.
  - Phosphorus (P) supports root development and early maturity.
  - Potassium (K) increases disease resistance and fruit quality.

#### 2. Recommended Crops
- **Rice (Paddy)**: Thrives in clayey/loamy soil, pH 5.5-6.5, high moisture.
- **Groundnut**: Thrives in sandy loam, pH 6.0-7.0, moderate moisture.
- **Jowar/Bajra**: Drought-tolerant, grows well in dry soils.
- **Sugarcane**: High nutrient requirements, thrives in fertile loam.

#### 3. Recommended Fertilizers
- For Nitrogen deficiency: Apply **Urea** or **Ammonium Sulfate**.
- For Phosphorus deficiency: Apply **Single Superphosphate (SSP)** or **DAP (Diammonium Phosphate)**.
- For Potassium deficiency: Apply **Muriate of Potash (MOP)**.
- **Organic recommendation**: Apply 10-15 tons of Farm Yard Manure (FYM) per hectare.
"""

        # If it's an image disease analysis request
        if has_image:
            return """
### [OFFLINE INTELLIGENCE FALLBACK] Crop Disease Analysis
*Note: This analysis is running in offline mode. Please configure an API Key to activate live Gemini Vision disease diagnosis.*

#### 1. Visual Diagnostics & Analysis
- **Suspected Pathogen**: Fungal Leaf Spot / Blight or Rust.
- **Symptoms Found**: Brown necrotic lesions with surrounding yellow halos on foliage, causing leaf chlorosis and premature leaf drop.

#### 2. Treatment & Control Solutions
- **Immediate Action**: Remove and burn infected leaves to prevent secondary spread. Avoid overhead sprinkler irrigation; apply water directly to soil.
- **Organic Controls**: Spray Neem Oil (1-2% solution) or apply a baking soda spray (1 tbsp baking soda, 1 tbsp horticultural oil, 1 gallon water).
- **Chemical Controls**: Apply copper-based fungicides (e.g., copper oxychloride) or systemic fungicides (e.g., carbendazim) if infection exceeds 15% of crop canopy.
"""

        # General farming questions assistant
        return f"""
### [OFFLINE INTELLIGENCE FALLBACK] AgriSmart AI Assistant
*Note: Operating in local fallback mode. Connect an API Key for real-time web search and dynamic LLM reasoning.*

Based on your question: "{prompt[:100]}...", here is some relevant farming guidance:
1. **Soil & Fertilizers**: Maintain a balanced NPK application. Incorporate organic matter (compost, green manures) to improve soil structure and water retention.
2. **Crop Yield**: Always source certified seeds, practice crop rotation (cereal followed by pulses), and monitor soil moisture.
3. **Pest Management**: Practice Integrated Pest Management (IPM). Use yellow sticky traps, introduce beneficial predatory insects, and apply neem-based sprays before opting for chemical pesticides.
4. **Market & Trends**: In Tamil Nadu, prices of cash crops like turmeric, sugarcane, and bananas are highly dependent on seasonal monsoon distribution.
"""
