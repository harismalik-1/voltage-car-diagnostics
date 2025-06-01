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
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 lg:py-12 max-w-7xl">
        {/* Top Section - Vehicle Status and Diagnostics */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <VehicleStatus />
          </div>
          <div className="lg:col-span-1">
            <DiagnosticsSection />
          </div>
        </section>

        {/* Middle Section - Diagnostic Cards and ML Insights */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="space-y-6">
            <DiagnosticCards />
          </div>
          <div className="space-y-6">
            <MLInsights />
          </div>
        </section>

        {/* Bottom Section - Customer Workflow */}
        <section className="mt-8">
          <CustomerWorkflow />
        </section>
      </main>
    </div>
  );
};

export default Index;
