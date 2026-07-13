import os
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from PIL import Image

# Import custom models and services
from data_analyser import AgriDataAnalyser
from ai_engine import AIEngine
from weather_service import get_realtime_weather
from soil_model import predict_soil_crop
from disease_model import analyze_crop_image
from assistant_model import rag_engine

app = FastAPI(title="AgriSmart AI Model Backend", version="2.0.0")

# Enable CORS for communication from the Node.js backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Analyser and AI Engine
analyser = AgriDataAnalyser()
ai_engine = AIEngine()

# Pydantic Schemas
class SoilAnalysisRequest(BaseModel):
    n: float
    p: float
    k: float
    ph: float
    moisture: float
    temperature: float
    district: Optional[str] = "Tamil Nadu"

class MarketTrendsRequest(BaseModel):
    crop: Optional[str] = "Rice"
    district: Optional[str] = "Tamil Nadu"

class MeasurementRequest(BaseModel):
    district: str
    crop: str
    area: float  # In hectares
    production: float  # In tonnes

class AssistantRequest(BaseModel):
    message: str
    chat_history: Optional[List[dict]] = []
    district: Optional[str] = "Tamil Nadu"

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "AgriSmart AI Python Backend (V2.0 - Local Models Integrated)",
        "has_crop_data": analyser.crop_data is not None,
        "has_rainfall_data": analyser.rainfall_data is not None,
        "has_land_use_data": analyser.land_use_data is not None
    }

@app.post("/analyze-soil")
def analyze_soil(request: SoilAnalysisRequest):
    # 1. Run local RandomForest Classifier Crop Prediction
    ml_prediction = predict_soil_crop(
        request.n, request.p, request.k, request.ph, request.moisture, request.temperature
    )
    
    # 2. Fetch real-time weather readings for the district
    weather = get_realtime_weather(request.district)
    
    # 3. Get historical district statistics
    recommendations = analyser.get_crop_recommendations_by_district(request.district)
    recommendations_str = ""
    if recommendations:
        recommendations_str = "Historically successful crops in " + request.district + ":\n"
        for idx, rec in enumerate(recommendations):
            recommendations_str += f"- {rec['crop']} (Avg Yield: {rec['avg_yield']:.2f} t/ha)\n"

    system_instruction = "You are a professional agronomist and soil scientist advising a farmer in Tamil Nadu, India."
    
    prompt = f"""
Analyze the following soil test measurements and weather forecast to provide a structured agronomy report:
Soil Metrics:
- Nitrogen (N): {request.n} kg/ha
- Phosphorus (P): {request.p} kg/ha
- Potassium (K): {request.k} kg/ha
- Soil pH: {request.ph}
- Soil Moisture: {request.moisture}%
- Soil Temperature: {request.temperature}°C

Real-Time Weather Readings (District: {request.district}):
- Current Air Temperature: {weather['temperature']}°C
- Current Relative Humidity: {weather['humidity']}%
- Current Precipitation (Rainfall): {weather['precipitation']} mm
- Current Wind Speed: {weather['wind_speed']} km/h

Local Machine Learning Soil Model Output:
- Recommended Crop: {ml_prediction['recommended_crop']}
- Confidence Status: Nitrogen is {ml_prediction['nutrient_status']['nitrogen_status']}, Phosphorus is {ml_prediction['nutrient_status']['phosphorus_status']}, Potassium is {ml_prediction['nutrient_status']['potassium_status']}
- Recommended fertilizer correction inputs (kg/ha): 
  * Urea: {ml_prediction['fertilizer_plan_kg_per_ha']['urea_nitrogen']} kg
  * SSP: {ml_prediction['fertilizer_plan_kg_per_ha']['single_superphosphate_phosphorus']} kg
  * MOP: {ml_prediction['fertilizer_plan_kg_per_ha']['muriate_of_potash_potassium']} kg

{recommendations_str}

Please generate a detailed report with the following structure:
1. **Soil Health Assessment**: Analyze the pH level and N-P-K nutrient status.
2. **Crop Recommendations**: Explain why the crop '{ml_prediction['recommended_crop']}' is recommended by the ML model. Contrast with historical crops if necessary.
3. **Fertilizer Dosage & Nutrition Plan**: Give precise recommendations for correcting deficiencies (incorporating the ML model's calculated Urea, SSP, and MOP dosages).
4. **Weather-Guided Water Management Tips**: Formulate an irrigation schedule based on the real-time air temperature ({weather['temperature']}°C) and relative humidity ({weather['humidity']}%).

Write your response in clear, readable markdown format.
"""
    try:
        report = ai_engine.call_llm(prompt=prompt, system_instruction=system_instruction)
        return {
            "soil_parameters": request.dict(),
            "local_ml_recommendations": ml_prediction,
            "realtime_weather_readings": weather,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Soil Analysis failed: {str(e)}")

@app.post("/detect-disease")
async def detect_disease(
    file: UploadFile = File(...),
    crop_name: Optional[str] = Form(None)
):
    try:
        # Read file bytes
        contents = await file.read()
        
        # Verify it's a valid image
        try:
            image = Image.open(io.BytesIO(contents))
            mime_type = file.content_type or "image/jpeg"
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # 1. Run local image thresholding model (Color signature diagnosis)
        local_model_results = analyze_crop_image(contents)

        # 2. Run LLM Vision model (Gemini / OpenAI / Anthropic)
        system_instruction = "You are an expert crop pathologist. You analyze images of infected crops to identify diseases and recommend treatment."
        crop_context = f"This is reported to be a {crop_name} crop." if crop_name else "Please identify the crop type first."
        
        prompt = f"""
{crop_context}

Examine the uploaded image of the crop leaf.
Local Color Signature Analysis Output:
- Health Status: {local_model_results['status']}
- Suspected Disease: {local_model_results['detected_disease']}
- Canopy Damage Severity Index: {local_model_results['severity_percentage']}% ({local_model_results['severity_level']})
- Color Distribution: Green: {local_model_results['color_signature'].get('healthy_green_pct') or 0}%, Yellow (Chlorosis): {local_model_results['color_signature'].get('chlorosis_yellow_pct') or 0}%, Brown (Necrosis): {local_model_results['color_signature'].get('necrotic_brown_pct') or 0}%

Please output a detailed diagnosis report:
1. **Visual Findings**: Verify if the leaf matches the local color model's finding of {local_model_results['detected_disease']} with {local_model_results['severity_level']} severity.
2. **Crop Identification & Disease Name**: Identify the crop and the formal disease.
3. **Remediation Plan**:
   - **Organic remedies**: (e.g. neem oil, ash, cultural actions).
   - **Chemical remedies**: (specific fungicides or pesticides).
   - **Preventative measures** for next season.

Provide your response in clean markdown format.
"""
        # Call LLM with image
        analysis = ai_engine.call_llm(prompt=prompt, image_bytes=contents, image_mime=mime_type, system_instruction=system_instruction)
        
        return {
            "filename": file.filename,
            "crop_name": crop_name,
            "local_analysis": local_model_results,
            "ai_analysis_report": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop disease analysis failed: {str(e)}")

@app.post("/market-trends")
def market_trends(request: MarketTrendsRequest):
    # 1. Search the web for current market trends and prices for the crop in Tamil Nadu
    search_query = f"current market price and demand of {request.crop} in Tamil Nadu agriculture markets 2026"
    search_results = ai_engine.search_web(search_query)

    # 2. Get historical crop data averages in the district
    hist_stats = analyser.get_district_crop_stats(request.district, request.crop)
    hist_stats_str = ""
    if "error" not in hist_stats and "message" not in hist_stats:
        hist_stats_str = f"Historical District Averages for {request.crop} in {request.district}:\n- Area: {hist_stats['avg_area']:.1f} ha\n- Production: {hist_stats['avg_production']:.1f} tonnes\n- Yield: {hist_stats['avg_yield']:.2f} t/ha\n"

    # 3. Get real-time weather (affects supply forecast!)
    weather = get_realtime_weather(request.district)

    system_instruction = "You are an agricultural economist specializing in Indian crop markets and price forecasting."
    
    prompt = f"""
Analyze the current market trends, demand, and prices for '{request.crop}' in '{request.district}', Tamil Nadu.
Here are the real-time search results from the internet:
{search_results}

{hist_stats_str}

District Weather Readings:
- Temperature: {weather['temperature']}°C
- Precipitation (Rain): {weather['precipitation']} mm
- Humidity: {weather['humidity']}%

Please compile a structured market trends report including:
1. **Current Market Price**: The estimated trading price per quintal or tonne in major agricultural markets (mandis) in Tamil Nadu.
2. **Demand & Supply Outlook**: Is the demand high, low, or stable? Any supply constraints?
3. **Weather Impact**: How current temperature ({weather['temperature']}°C) and precipitation ({weather['precipitation']} mm) affect supply.
4. **Farming Advice**: Recommendations on whether the farmer should sell now, hold, or change crop varieties for the next cycle.

Provide your response in markdown format.
"""
    try:
        report = ai_engine.call_llm(prompt=prompt, system_instruction=system_instruction)
        return {
            "crop": request.crop,
            "district": request.district,
            "realtime_weather": weather,
            "web_search_context": search_results,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market trends analysis failed: {str(e)}")

@app.post("/measure")
def measure_yield(request: MeasurementRequest):
    results = analyser.compare_yield_efficiency(
        district=request.district,
        crop=request.crop,
        farmer_area=request.area,
        farmer_production=request.production
    )
    # Fetch real-time weather readings to append context
    weather = get_realtime_weather(request.district)
    return {
        "input_parameters": request.dict(),
        "realtime_weather": weather,
        "analysis": results
    }

@app.post("/assistant")
def assistant(request: AssistantRequest):
    # 1. RAG context retrieval using TF-IDF cosine similarity search over proposal & agronomic guidelines
    rag_context = rag_engine.retrieve_context(request.message, top_n=2)
    
    # 2. Add district context
    district_data = ""
    if request.district and request.district != "Tamil Nadu":
        district_stats = analyser.get_district_crop_stats(request.district)
        if "error" not in district_stats:
            district_data = f"District Stats for {request.district}:\n- Top crops: {list(district_stats['top_crops_by_production'].keys())}\n- Seasons: {district_stats['seasons']}\n"

    # 3. Add real-time weather context
    weather = get_realtime_weather(request.district)
    weather_str = f"Real-Time Weather in {request.district}: Temp: {weather['temperature']}°C, Humidity: {weather['humidity']}%, Rainfall: {weather['precipitation']}mm"

    # 4. Check if user prompt requires internet search
    query_lower = request.message.lower()
    search_context = ""
    is_realtime_query = any(k in query_lower for k in ["today", "weather", "forecast", "price of", "latest", "current", "news", "trend", "2026"])
    
    if is_realtime_query:
        search_query = f"{request.message} in Tamil Nadu"
        search_context = ai_engine.search_web(search_query, max_results=3)

    # 5. Format chat history
    chat_history_str = ""
    if request.chat_history:
        chat_history_str = "Recent Chat History:\n"
        for msg in request.chat_history[-6:]:
            role = "Farmer" if msg.get("role") == "user" else "Assistant"
            chat_history_str += f"{role}: {msg.get('content')}\n"

    # Check if we should use local offline QA
    is_offline = not (ai_engine.gemini_key or ai_engine.openai_key or ai_engine.anthropic_key)
    
    if is_offline and not is_realtime_query:
        reply = rag_engine.offline_qa(request.message, rag_context)
        return {
            "response": reply,
            "used_web_search": False,
            "web_search_query": None,
            "rag_context_used": rag_context
        }

    system_instruction = "You are the AgriSmart AI Assistant, a friendly, intelligent agricultural advisor helping farmers optimize their crops and yields."

    prompt = f"""
Here is the RAG context (AgriSmart Proposal documents and agronomic rules):
{rag_context}

{district_data}
{weather_str}

{"Real-time search context from the internet:\n" + search_context if search_context else ""}

{chat_history_str}
Farmer's new question: {request.message}

Please provide a helpful, scientifically accurate, and encouraging response to the farmer. Quote the RAG context or real-time weather readings ({weather['temperature']}°C temp, {weather['humidity']}% humidity) where appropriate. Keep it structured and easy to read.
"""
    try:
        reply = ai_engine.call_llm(prompt=prompt, system_instruction=system_instruction)
        return {
            "response": reply,
            "used_web_search": is_realtime_query,
            "web_search_query": search_query if is_realtime_query else None,
            "rag_context_used": rag_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
