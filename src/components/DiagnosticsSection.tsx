
import React from 'react';
import { Car3D } from './Car3D';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Play, Zap } from 'lucide-react';

export const DiagnosticsSection = () => {
  const handleRunDiagnostics = () => {
    console.log('Running diagnostics...');
    // Add diagnostics logic here
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
          <Button 
            onClick={handleRunDiagnostics}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-light py-3 rounded-lg transition-all duration-300 transform hover:scale-105"
          >
            <Play className="w-4 h-4 mr-2" />
            Run Diagnostics
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};
