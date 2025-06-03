// controllers/canLogsController.js
const path = require("path");
const fs = require("fs");
const {
  getAllCanLogs,
  getCanLogById,
  createCanLog,
} = require("../models/canLogModel");

const DATA_DIR = path.join(__dirname, "../data/can_logs");

async function listCanLogs(req, res, next) {
  try {
    // parse optional query params ?limit= & ?offset=
    const limit = parseInt(req.query.limit, 10) || 100;
    const offset = parseInt(req.query.offset, 10) || 0;
    const logs = await getAllCanLogs(limit, offset);
    res.json(logs);
  } catch (err) {
    next(err);
  }
}

async function downloadCanLog(req, res, next) {
  try {
    const logId = parseInt(req.params.id, 10);
    if (isNaN(logId)) {
      return res.status(400).json({ error: "Invalid log ID" });
    }

    const log = await getCanLogById(logId);
    if (!log) {
      return res.status(404).json({ error: "CAN log not found" });
    }

    const filePath = path.join(DATA_DIR, log.filename);
    if (!fs.existsSync(filePath)) {
      return res
        .status(404)
        .json({ error: "File exists in DB but missing on server" });
    }

    // Stream the file back
    res.setHeader("Content-Type", "text/csv");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename="${log.filename}"`
    );
    const stream = fs.createReadStream(filePath);
    stream.pipe(res);
    stream.on("error", (err) => next(err));
  } catch (err) {
    next(err);
  }
}

async function registerCanLog(req, res, next) {
  try {
    // For demo, we just register an existing file in data/can_logs—
    // In production, you'd parse req.file for multipart upload.
    const { filename, description } = req.body;
    if (!filename) {
      return res.status(400).json({ error: "filename is required" });
    }

    const filePath = path.join(DATA_DIR, filename);
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "CSV not found on server" });
    }

    const newLog = await createCanLog(filename, description || null);
    res.status(201).json(newLog);
  } catch (err) {
    // unique constraint violation?
    if (err.code === "23505") {
      return res.status(409).json({ error: "filename already registered" });
    }
    next(err);
  }
}

module.exports = { listCanLogs, downloadCanLog, registerCanLog };

