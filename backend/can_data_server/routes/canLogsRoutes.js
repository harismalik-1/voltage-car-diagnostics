// routes/canLogsRoutes.js
const express = require("express");
const router = express.Router();
const {
  listCanLogs,
  downloadCanLog,
  registerCanLog,
} = require("../controllers/canLogsController");

// GET  /can-logs          → list all registered logs (metadata)
// POST /can-logs          → register a new log (filename + optional description)
// GET  /can-logs/:id      → download the CSV file for that ID
router.get("/", listCanLogs);
router.post("/", registerCanLog);
router.get("/:id", downloadCanLog);

module.exports = router;
