import React, { useState } from 'react';
import { Car3D } from './Car3D';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Play, Zap, AlertTriangle } from 'lucide-react';

interface DiagnosticsSectionProps {
  onRunDiagnostics: () => void;
}

interface AnomalyResult {
  window_index: number;
  start_row: number;
  timestamp: string | null;
}

interface DiagnosticsResult {
  total_windows: number;
  num_anomalies: number;
  anomalies: AnomalyResult[];
}

export const DiagnosticsSection: React.FC<DiagnosticsSectionProps> = ({ onRunDiagnostics }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [diagnosticsResult, setDiagnosticsResult] = useState<DiagnosticsResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloadLoading, setDownloadLoading] = useState(false);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file, file.name);

    try {
      const response = await fetch("http://127.0.0.1:8000/evaluate", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to evaluate diagnostics');
      }

      const result = await response.json();
      setDiagnosticsResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error("Error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadCanData = async () => {
    setDownloadLoading(true);
    setError(null);
    try {
      const response = await fetch("http://127.0.0.1:3000/can-data");
      if (!response.ok) {
        throw new Error("Failed to download CAN data");
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "can_data2.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred while downloading CAN data");
    } finally {
      setDownloadLoading(false);
    }
  };

  return (
    <Card className="bg-gray-900/50 border-gray-800 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-xl font-light text-white flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
            <Zap className="w-4 h-4 text-white" />
          </div>
          Vehicle Overview
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <Car3D />
        <div className="text-center space-y-4">
          <div className="space-y-2">
            <h3 className="text-lg font-light text-white">Cybertruck</h3>
            <p className="text-sm text-gray-400">VIN: 5YJ3E1EA9KF123456</p>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-400">Connected</span>
            </div>
          </div>

          {/* File Input (Hidden) */}
          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
            id="diagnostics-file-input"
          />

          {/* Get CAN Data Button */}
          <Button
            onClick={handleDownloadCanData}
            disabled={downloadLoading}
            className="w-full mb-2 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white font-light py-3 rounded-lg transition-all duration-300 transform hover:scale-105 disabled:opacity-50"
          >
            {downloadLoading ? (
              <div className="flex items-center">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Downloading...
              </div>
            ) : (
              <>Get CAN Data</>
            )}
          </Button>

          {/* Run Diagnostics Button */}
          <Button 
            onClick={() => document.getElementById('diagnostics-file-input')?.click()}
            disabled={isLoading}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-light py-3 rounded-lg transition-all duration-300 transform hover:scale-105 disabled:opacity-50"
          >
            {isLoading ? (
              <div className="flex items-center">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Processing...
              </div>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Run Diagnostics
              </>
            )}
          </Button>

          {/* Results Display */}
          {diagnosticsResult && (
            <div className="mt-4 p-4 bg-gray-800/50 rounded-lg">
              <h4 className="text-lg font-light text-white mb-2">Diagnostics Results</h4>
              <div className="space-y-2">
                <p className="text-sm text-gray-300">
                  Total Windows Analyzed: {diagnosticsResult.total_windows}
                </p>
                <p className="text-sm text-gray-300">
                  Anomalies Detected: {diagnosticsResult.num_anomalies}
                </p>
                {diagnosticsResult.num_anomalies > 0 && (
                  <div className="mt-4">
                    <h5 className="text-sm font-medium text-red-400 mb-2 flex items-center">
                      <AlertTriangle className="w-4 h-4 mr-2" />
                      Anomaly Details
                    </h5>
                    <div className="max-h-40 overflow-y-auto">
                      {diagnosticsResult.anomalies.map((anomaly, index) => (
                        <div key={index} className="text-sm text-gray-400 mb-1">
                          Window {anomaly.window_index} 
                          {anomaly.timestamp && ` at ${anomaly.timestamp}`}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-4 bg-red-900/50 rounded-lg">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
