import React from 'react';
import { DiagnosticHeader } from '../components/DiagnosticHeader';
import { DiagnosticCards } from '../components/DiagnosticCards';
import { MLInsights } from '../components/MLInsights';

const Diagnostics = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
      <DiagnosticHeader />
      
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 lg:py-12 max-w-7xl">
        <div className="space-y-6">
          <h1 className="text-2xl font-light text-white mb-8">Diagnostic Results</h1>
          
          {/* Real-time Diagnostics and AI Insights */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-6">
              <DiagnosticCards />
            </div>
            <div className="space-y-6">
              <MLInsights />
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default Diagnostics; 