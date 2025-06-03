import React, { useState } from 'react';
import { DiagnosticHeader } from '../components/DiagnosticHeader';
import { VehicleStatus } from '../components/Analysis';
import { DiagnosticsSection } from '../components/DiagnosticsSection';
import { CustomerWorkflow } from '../components/CustomerWorkflow';
import { DiagnosticLoading } from '../components/DiagnosticLoading';
import { useNavigate } from 'react-router-dom';
//import { NewComponent } from '../components/NewComponent'; // ← your new component

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleRunDiagnostics = () => {
    setIsLoading(true);
    // Simulate diagnostic process
    setTimeout(() => {
      setIsLoading(false);
      navigate('/diagnostics');
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
      <DiagnosticHeader />

      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 lg:py-12 max-w-7xl">
        {/* Top Section - VehicleStatus and DiagnosticsSection */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="lg:col-span-1">
            <VehicleStatus />
          </div>

          <div className="lg:col-span-1">
            <DiagnosticsSection onRunDiagnostics={handleRunDiagnostics} />
          </div>
        </section>

        {/* Bottom Section - Customer Workflow */}
        <section className="mt-8">
          <CustomerWorkflow />
        </section>
      </main>

      {isLoading && <DiagnosticLoading />}
    </div>
  );
};

export default Index;
