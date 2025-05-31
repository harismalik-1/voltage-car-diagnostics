
import React from 'react';
import { DiagnosticHeader } from '../components/DiagnosticHeader';
import { VehicleStatus } from '../components/VehicleStatus';
import { DiagnosticCards } from '../components/DiagnosticCards';
import { MLInsights } from '../components/MLInsights';
import { CustomerWorkflow } from '../components/CustomerWorkflow';
import { DiagnosticsSection } from '../components/DiagnosticsSection';

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
      <DiagnosticHeader />
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-8">
          <div className="xl:col-span-2">
            <VehicleStatus />
          </div>
          <DiagnosticsSection />
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-8">
          <DiagnosticCards />
          <MLInsights />
        </div>
        <CustomerWorkflow />
      </div>
    </div>
  );
};

export default Index;
