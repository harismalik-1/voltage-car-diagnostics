// app.js
require("dotenv").config();
const express = require("express");
const path = require("path");
const canLogsRoutes = require("./routes/canLogsRoutes");
const errorHandler = require("./middleware/errorHandler");

const app = express();
const cors = require('cors');
app.use(cors());

// 1) Middleware: parse JSON bodies (for POST /can-logs)
app.use(express.json());

// 2) Static health check or landing route
app.get("/", (req, res) => {
  res.json({ message: "CAN-Catalog Service is running." });
});

// 3) Mount the /can-logs router
app.use("/can-logs", canLogsRoutes);

const CAN_DATA_PATH = path.join(__dirname, "data/can_logs/can_data.csv");

// Serve can_data.csv directly
app.get("/can-data", (req, res) => {
  console.log("Serving file from:", CAN_DATA_PATH);
  res.download(CAN_DATA_PATH, "can_data.csv", (err) => {
    if (err) {
      console.error("Download error:", err);
      res.status(404).json({ error: "CAN data file not found" });
    }
  });
});

// 4) Error handler (should live after all routes)
app.use(errorHandler);

// 5) Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server listening on http://localhost:${PORT}`);
});
