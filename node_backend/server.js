const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../.env') });
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
const db = require('./db');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

// Set up storage for uploaded crop disease images
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const upload = multer({ storage: storage });

// 1. Root Endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'online',
    service: 'AgriSmart Node.js API Gateway',
    dbPath: path.join(__dirname, 'agri_smart.db'),
    pythonBackendUrl: PYTHON_BACKEND_URL
  });
});

// 2. Soil Analysis & Fertilizer Recommendation Endpoints
app.get('/api/soil-records', async (req, res) => {
  try {
    const rows = await db.all('SELECT * FROM soil_records ORDER BY created_at DESC');
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/analyze-soil', async (req, res) => {
  const { n, p, k, ph, moisture, temperature, district } = req.body;
  
  if (n === undefined || p === undefined || k === undefined || ph === undefined || moisture === undefined || temperature === undefined) {
    return res.status(400).json({ error: 'All soil parameters (N, P, K, pH, moisture, temperature) are required.' });
  }

  try {
    // 1. Call Python AI Backend for report
    const response = await axios.post(`${PYTHON_BACKEND_URL}/analyze-soil`, {
      n: parseFloat(n),
      p: parseFloat(p),
      k: parseFloat(k),
      ph: parseFloat(ph),
      moisture: parseFloat(moisture),
      temperature: parseFloat(temperature),
      district: district || 'Tamil Nadu'
    });

    const report = response.data.report;

    // 2. Save in SQLite Database
    const insertResult = await db.run(
      `INSERT INTO soil_records (n, p, k, ph, moisture, temperature, district, report) 
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [n, p, k, ph, moisture, temperature, district || 'Tamil Nadu', report]
    );

    res.json({
      id: insertResult.id,
      soil_parameters: response.data.soil_parameters,
      local_ml_recommendations: response.data.local_ml_recommendations,
      realtime_weather_readings: response.data.realtime_weather_readings,
      report: report
    });
  } catch (err) {
    console.error('Soil Analysis Error:', err.message);
    res.status(500).json({ error: 'Soil Analysis failed: ' + (err.response?.data?.detail || err.message) });
  }
});

// 3. Crop Disease Detection Endpoints
app.get('/api/diseases', async (req, res) => {
  try {
    const rows = await db.all('SELECT * FROM disease_detections ORDER BY created_at DESC');
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/detect-disease', upload.single('file'), async (req, res) => {
  const { crop_name } = req.body;
  const file = req.file;

  if (!file) {
    return res.status(400).json({ error: 'Crop image file is required.' });
  }

  try {
    // 1. Forward the file to the Python FastAPI backend
    const form = new FormData();
    form.append('file', fs.createReadStream(file.path), file.filename);
    if (crop_name) {
      form.append('crop_name', crop_name);
    }

    const response = await axios.post(`${PYTHON_BACKEND_URL}/detect-disease`, form, {
      headers: {
        ...form.getHeaders()
      }
    });

    const analysis = response.data.analysis;
    const finalCropName = response.data.crop_name || crop_name || 'Detected Crop';

    // 2. Save in SQLite Database
    const relativeImagePath = path.relative(__dirname, file.path);
    const insertResult = await db.run(
      `INSERT INTO disease_detections (crop_name, analysis, image_path) 
       VALUES (?, ?, ?)`,
      [finalCropName, analysis, relativeImagePath]
    );

    res.json({
      id: insertResult.id,
      crop_name: finalCropName,
      local_analysis: response.data.local_analysis,
      ai_analysis_report: analysis,
      image_path: relativeImagePath
    });
  } catch (err) {
    console.error('Disease Detection Error:', err.message);
    res.status(500).json({ error: 'Crop disease detection failed: ' + (err.response?.data?.detail || err.message) });
  }
});

// 4. Market Trends Endpoints
app.get('/api/market-trends', async (req, res) => {
  try {
    const rows = await db.all('SELECT * FROM market_trends ORDER BY created_at DESC');
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/market-trends', async (req, res) => {
  const { crop, district } = req.body;

  if (!crop) {
    return res.status(400).json({ error: 'Crop parameter is required.' });
  }

  try {
    // 1. Call Python AI Backend
    const response = await axios.post(`${PYTHON_BACKEND_URL}/market-trends`, {
      crop: crop,
      district: district || 'Tamil Nadu'
    });

    const report = response.data.report;

    // 2. Save in SQLite Database
    const insertResult = await db.run(
      `INSERT INTO market_trends (crop, district, report) 
       VALUES (?, ?, ?)`,
      [crop, district || 'Tamil Nadu', report]
    );

    res.json({
      id: insertResult.id,
      crop: crop,
      district: district || 'Tamil Nadu',
      realtime_weather: response.data.realtime_weather,
      web_search_context: response.data.web_search_context,
      report: report
    });
  } catch (err) {
    console.error('Market Trends Error:', err.message);
    res.status(500).json({ error: 'Market trends analysis failed: ' + (err.response?.data?.detail || err.message) });
  }
});

// 5. Yield Measurements & Efficiency Comparison Endpoints
app.get('/api/measurements', async (req, res) => {
  try {
    const rows = await db.all('SELECT * FROM measurements ORDER BY created_at DESC');
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/measure', async (req, res) => {
  const { district, crop, area, production } = req.body;

  if (!district || !crop || area === undefined || production === undefined) {
    return res.status(400).json({ error: 'district, crop, area, and production parameters are required.' });
  }

  try {
    // 1. Call Python AI Backend
    const response = await axios.post(`${PYTHON_BACKEND_URL}/measure`, {
      district,
      crop,
      area: parseFloat(area),
      production: parseFloat(production)
    });

    const yieldVal = response.data.analysis.farmer_yield;
    const comparisonText = response.data.analysis.comparison;

    // 2. Save in SQLite Database
    const insertResult = await db.run(
      `INSERT INTO measurements (district, crop, area, production, yield, comparison) 
       VALUES (?, ?, ?, ?, ?, ?)`,
      [district, crop, area, production, yieldVal, comparisonText]
    );

    res.json({
      id: insertResult.id,
      realtime_weather: response.data.realtime_weather,
      analysis: response.data.analysis
    });
  } catch (err) {
    console.error('Measurement Error:', err.message);
    res.status(500).json({ error: 'Measurement analysis failed: ' + (err.response?.data?.detail || err.message) });
  }
});

// 6. RAG AI Assistant Endpoints (Multi-Turn Chat History stored in SQLite)
app.get('/api/chat-history/:sessionId', async (req, res) => {
  const { sessionId } = req.params;
  try {
    const rows = await db.all('SELECT role, content, created_at FROM chat_history WHERE session_id = ? ORDER BY id ASC', [sessionId]);
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/chat-message', async (req, res) => {
  const { session_id, message, district } = req.body;

  if (!session_id || !message) {
    return res.status(400).json({ error: 'session_id and message are required.' });
  }

  try {
    // 1. Save user message to database
    await db.run(
      'INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)',
      [session_id, 'user', message]
    );

    // 2. Fetch full conversation history for context
    const historyRows = await db.all(
      'SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id ASC',
      [session_id]
    );
    
    // Map database history format to python endpoint request format
    const chatHistoryPayload = historyRows.slice(0, -1).map(row => ({
      role: row.role,
      content: row.content
    }));

    // 3. Call Python Assistant (RAG + Web Search if appropriate)
    const response = await axios.post(`${PYTHON_BACKEND_URL}/assistant`, {
      message: message,
      chat_history: chatHistoryPayload,
      district: district || 'Tamil Nadu'
    });

    const assistantReply = response.data.response;

    // 4. Save assistant reply to database
    await db.run(
      'INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)',
      [session_id, 'assistant', assistantReply]
    );

    res.json({
      response: assistantReply,
      used_web_search: response.data.used_web_search,
      web_search_query: response.data.web_search_query
    });
  } catch (err) {
    console.error('Assistant Error:', err.message);
    res.status(500).json({ error: 'AI Assistant response failed: ' + (err.response?.data?.detail || err.message) });
  }
});

// Serve uploaded crop disease images statically (if needed)
app.use('/uploads', express.static(uploadDir));

// Start server
app.listen(PORT, () => {
  console.log(`Node.js AgriSmart backend running on port ${PORT}`);
});
