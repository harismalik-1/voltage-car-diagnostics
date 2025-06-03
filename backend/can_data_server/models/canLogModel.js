// models/canLogModel.js
const pool = require("../config/db");

async function getAllCanLogs(limit = 100, offset = 0) {
  const res = await pool.query(
    `SELECT id, filename, description, uploaded_at
     FROM can_logs
     ORDER BY uploaded_at DESC
     LIMIT $1 OFFSET $2`,
    [limit, offset]
  );
  return res.rows;
}

async function getCanLogById(logId) {
  const res = await pool.query(
    `SELECT id, filename, description, uploaded_at
     FROM can_logs
     WHERE id = $1`,
    [logId]
  );
  return res.rows[0] || null;
}

async function createCanLog(filename, description = null) {
  const res = await pool.query(
    `INSERT INTO can_logs (filename, description)
     VALUES ($1, $2)
     RETURNING id, filename, description, uploaded_at`,
    [filename, description]
  );
  return res.rows[0];
}

module.exports = { getAllCanLogs, getCanLogById, createCanLog };
