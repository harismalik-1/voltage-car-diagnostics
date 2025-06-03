
import React from 'react';
import { Car, Battery, Thermometer, Settings } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export const VehicleStatus = () => {
  const statusItems = [
    { icon: Car, label: 'Vehicle Health', value: '98%', status: 'excellent', color: 'text-green-400', bgColor: 'bg-green-400/10' },
    { icon: Battery, label: 'Battery Status', value: '87%', status: 'good', color: 'text-blue-400', bgColor: 'bg-blue-400/10' },
    { icon: Thermometer, label: 'Thermal Management', value: 'Optimal', status: 'normal', color: 'text-emerald-400', bgColor: 'bg-emerald-400/10' },
    { icon: Settings, label: 'System Performance', value: '94%', status: 'excellent', color: 'text-purple-400', bgColor: 'bg-purple-400/10' },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {statusItems.map((item, index) => (
        <Card key={index} className="bg-gray-900/50 border-gray-800 hover:bg-gray-900/70 transition-all duration-300 backdrop-blur-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <div className={`flex items-center justify-center w-8 h-8 rounded-lg ${item.bgColor}`}>
              <item.icon className={`h-4 w-4 ${item.color}`} />
            </div>
            <div className="text-right">
              <div className="text-2xl font-light text-white">{item.value}</div>
              <p className="text-xs text-gray-400 uppercase tracking-wider">{item.status}</p>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <CardTitle className="text-sm font-light text-gray-300">{item.label}</CardTitle>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};
