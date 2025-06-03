import React from 'react';
import { Activity, Zap, CheckCircle2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export const DiagnosticLoading: React.FC = () => {
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <Card className="bg-gray-900/50 border-gray-800 backdrop-blur-sm w-full max-w-2xl mx-4">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-3 font-light">
            <Zap className="w-5 h-5 text-blue-400 animate-pulse" />
            <span>Running Diagnostics</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-blue-400/10">
                <Activity className="w-5 h-5 text-blue-400 animate-spin" />
              </div>
              <div className="flex-1">
                <p className="text-white font-medium">Analyzing Vehicle Systems</p>
                <div className="w-full h-1 bg-gray-800 rounded-full mt-2">
                  <div className="h-full bg-blue-400 rounded-full animate-[progress_3s_ease-in-out_forwards]"></div>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-purple-400/10">
                <CheckCircle2 className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <p className="text-white font-medium">Preparing Results</p>
                <p className="text-sm text-gray-400">Almost there...</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}; 