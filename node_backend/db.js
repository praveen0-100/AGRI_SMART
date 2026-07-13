const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const dbPath = path.join(__dirname, 'agri_smart.db');

// Connect to SQLite Database
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error('Error opening database:', err.message);
  } else {
    console.log('Connected to the AgriSmart SQLite database.');
  }
});

// Initialize database tables
db.serialize(() => {
  // Soil Records Table
  db.run(`
    CREATE TABLE IF NOT EXISTS soil_records (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      n REAL,
      p REAL,
      k REAL,
      ph REAL,
      moisture REAL,
      temperature REAL,
      district TEXT,
      report TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Crop Disease Detections Table
  db.run(`
    CREATE TABLE IF NOT EXISTS disease_detections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      crop_name TEXT,
      analysis TEXT,
      image_path TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Market Trends Cache Table
  db.run(`
    CREATE TABLE IF NOT EXISTS market_trends (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      crop TEXT,
      district TEXT,
      report TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Chat History Table for RAG AI Assistant
  db.run(`
    CREATE TABLE IF NOT EXISTS chat_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      role TEXT,
      content TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Crop Yield Measurements Table
  db.run(`
    CREATE TABLE IF NOT EXISTS measurements (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      district TEXT,
      crop TEXT,
      area REAL,
      production REAL,
      yield REAL,
      comparison TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);
});

// Helper functions wrapping sqlite3 callbacks in Promises
function run(sql, params = []) {
  return new Promise((resolve, reject) => {
    db.run(sql, params, function (err) {
      if (err) {
        reject(err);
      } else {
        resolve({ id: this.lastID, changes: this.changes });
      }
    });
  });
}

function all(sql, params = []) {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
}

function get(sql, params = []) {
  return new Promise((resolve, reject) => {
    db.get(sql, params, (err, row) => {
      if (err) {
        reject(err);
      } else {
        resolve(row);
      }
    });
  });
}

module.exports = {
  db,
  run,
  all,
  get
};
