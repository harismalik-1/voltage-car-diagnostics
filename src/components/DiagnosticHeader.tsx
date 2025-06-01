import React from 'react';
import { Activity, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const DiagnosticHeader = () => {
  const navigate = useNavigate();

  return (
    <header className="bg-black border-b border-gray-800">
      <div className="container mx-auto px-6 py-6">
        <div className="flex items-center justify-between">
          <div 
            className="flex items-center space-x-4 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate('/')}
          >
            <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg">
              <Zap className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-light text-white tracking-wide">Voltage</h1>
              <p className="text-sm text-gray-400 font-light">Automated Vehicle Diagnostics</p>
            </div>
          </div>
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-300 font-light">System Online</span>
            </div>
            <div className="text-xs text-gray-500">
              Last sync: 2 min ago
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
