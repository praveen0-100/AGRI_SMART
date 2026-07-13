# AgriSmart AI Backend Test Suite & Run Guide (V2.0)

This guide describes how to run both the Python FastAPI server and the Node.js API Gateway, explains the integrated local ML/CV models, and lists sample payloads, commands, and expected JSON outputs to test each feature.

---

## 1. Integrated AI & ML Models

The backend now contains local, on-the-fly intelligence services:
1. **Soil Crop Predictor (`soil_model.py`)**: Uses a `RandomForestClassifier` trained on startup to classify soil NPK/pH/moisture and compute Urea, SSP, and MOP fertilizer deficits (in kg/ha).
2. **Crop Disease Image Model (`disease_model.py`)**: Implements color thresholding over uploaded leaf images to calculate healthy green vs. yellow chlorosis vs. brown necrosis percentages and severity indices.
3. **Live Meteorological Weather (`weather_service.py`)**: Maps district queries to latitudes/longitudes and fetches live **air temperature**, **relative humidity**, **wind speed**, and **rainfall (precipitation)** from the Open-Meteo API.
4. **TF-IDF cosine-similarity RAG (`assistant_model.py`)**: Extracts text context matches dynamically from the project proposal PDF and agronomic rule bases.

---

## 2. Prerequisites & Startup

### Option A: Automatic Launch (Windows)
Double-click **[start_backend.bat](file:///C:/Users/PRAVEEN%20PRABAKARN/Desktop/new%20fold/AGRI_AI_BACKEND/start_backend.bat)** in the root folder. This automatically launches:
- **Python FastAPI Backend** on `http://127.0.0.1:8000` (automatically trains the soil RandomForest model and loads RAG documents)
- **Node.js Express Gateway** on `http://127.0.0.1:5000`

### Option B: Manual Launch
1. **Python Server**:
   ```bash
   cd python_backend
   py -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```
2. **Node.js Server**:
   ```bash
   cd node_backend
   npm start
   ```

---

## 3. Endpoints, Inputs, and Outputs

### 1. Soil Analysis & Fertilizer Recommendation
- **URL**: `POST http://127.0.0.1:5000/api/analyze-soil`
- **Headers**: `Content-Type: application/json`
- **JSON Input Payload**:
  ```json
  {
    "n": 240.5,
    "p": 12.8,
    "k": 180.2,
    "ph": 5.8,
    "moisture": 35.0,
    "temperature": 29.2,
    "district": "Coimbatore"
  }
  ```
- **Expected JSON Output**:
  ```json
  {
    "id": 1,
    "soil_parameters": {
      "n": 240.5, "p": 12.8, "k": 180.2, "ph": 5.8, "moisture": 35.0, "temperature": 29.2, "district": "Coimbatore"
    },
    "local_ml_recommendations": {
      "recommended_crop": "Rice",
      "alternatives": [
        { "crop": "Ragi", "probability": 0.18 },
        { "crop": "Groundnut", "probability": 0.14 }
      ],
      "nutrient_status": {
        "nitrogen_status": "deficient",
        "phosphorus_status": "deficient",
        "potassium_status": "optimal"
      },
      "fertilizer_plan_kg_per_ha": {
        "urea_nitrogen": 130.4,
        "single_superphosphate_phosphorus": 232.5,
        "muriate_of_potash_potassium": 0.0,
        "farm_yard_manure": 12000.0
      }
    },
    "realtime_weather_readings": {
      "district": "Coimbatore",
      "temperature": 28.6,
      "humidity": 68.0,
      "precipitation": 0.0,
      "wind_speed": 11.5,
      "source": "Open-Meteo API (Real-Time)"
    },
    "report": "### Soil Health Analysis... [Generated AI report]"
  }
  ```
- **cURL Command**:
  ```bash
  curl -X POST http://127.0.0.1:5000/api/analyze-soil \
       -H "Content-Type: application/json" \
       -d "{\"n\": 240.5, \"p\": 12.8, \"k\": 180.2, \"ph\": 5.8, \"moisture\": 35.0, \"temperature\": 29.2, \"district\": \"Coimbatore\"}"
  ```

---

### 2. Crop Leaf Disease Detection
- **URL**: `POST http://127.0.0.1:5000/api/detect-disease`
- **Headers**: `multipart/form-data`
- **Form Fields**:
  - `crop_name`: "Rice"
  - `file`: (File Upload)
- **Expected JSON Output**:
  ```json
  {
    "id": 1,
    "crop_name": "Rice",
    "image_path": "uploads/1720894562-leaf_test.jpg",
    "local_analysis": {
      "status": "Diseased",
      "detected_disease": "Leaf Spot / Fungal Leaf Blast",
      "severity_percentage": 24.5,
      "severity_level": "Moderate",
      "recommended_action": "Remove infected leaves. Spray copper oxychloride (2g/L)...",
      "color_signature": {
        "healthy_green_pct": 75.5,
        "chlorosis_yellow_pct": 12.0,
        "necrotic_brown_pct": 12.5,
        "other_shadow_pct": 0.0
      }
    },
    "ai_analysis_report": "### Diagnostic Report... [Generated AI visual analysis]"
  }
  ```
- **cURL Command** (Create a dummy `leaf_test.jpg` file first):
  ```bash
  curl -X POST http://127.0.0.1:5000/api/detect-disease \
       -F "crop_name=Rice" \
       -F "file=@leaf_test.jpg"
  ```

---

### 3. Real-Time Market Trends
- **URL**: `POST http://127.0.0.1:5000/api/market-trends`
- **Headers**: `Content-Type: application/json`
- **JSON Input Payload**:
  ```json
  {
    "crop": "Banana",
    "district": "Madurai"
  }
  ```
- **Expected JSON Output**:
  ```json
  {
    "id": 1,
    "crop": "Banana",
    "district": "Madurai",
    "realtime_weather": {
      "district": "Madurai", "temperature": 32.1, "humidity": 55.0, "precipitation": 0.0, "wind_speed": 14.2
    },
    "web_search_context": "[Real-time scraped news and price trends text]",
    "report": "### Market Trends Report for Banana... [Generated AI report]"
  }
  ```
- **cURL Command**:
  ```bash
  curl -X POST http://127.0.0.1:5000/api/market-trends \
       -H "Content-Type: application/json" \
       -d "{\"crop\": \"Banana\", \"district\": \"Madurai\"}"
  ```

---

### 4. Yield Efficiency Measurement
- **URL**: `POST http://127.0.0.1:5000/api/measure`
- **Headers**: `Content-Type: application/json`
- **JSON Input Payload**:
  ```json
  {
    "district": "Ariyalur",
    "crop": "Rice",
    "area": 2.0,
    "production": 6.8
  }
  ```
- **cURL Command**:
  ```bash
  curl -X POST http://127.0.0.1:5000/api/measure \
       -H "Content-Type: application/json" \
       -d "{\"district\": \"Ariyalur\", \"crop\": \"Rice\", \"area\": 2.0, \"production\": 6.8}"
  ```

---

### 5. Multi-Turn AI Assistant (RAG + Live Weather/Search)
- **URL**: `POST http://127.0.0.1:5000/api/chat-message`
- **Headers**: `Content-Type: application/json`
- **JSON Input Payload**:
  ```json
  {
    "session_id": "session_test_123",
    "message": "What is the proposal overview and what is the current weather in Coimbatore?",
    "district": "Coimbatore"
  }
  ```
- **cURL Command**:
  ```bash
  curl -X POST http://127.0.0.1:5000/api/chat-message \
       -H "Content-Type: application/json" \
       -d "{\"session_id\": \"session_test_123\", \"message\": \"What is the proposal overview and what is the current weather in Coimbatore?\", \"district\": \"Coimbatore\"}"
  ```

---

## 4. History Retrieval Endpoints
To verify that records are successfully saved in SQLite:
* **Soil History**: `GET http://127.0.0.1:5000/api/soil-records`
* **Disease History**: `GET http://127.0.0.1:5000/api/diseases`
* **Market Trends History**: `GET http://127.0.0.1:5000/api/market-trends`
* **Measurements History**: `GET http://127.0.0.1:5000/api/measurements`
* **Session Chat Log**: `GET http://127.0.0.1:5000/api/chat-history/session_test_123`
