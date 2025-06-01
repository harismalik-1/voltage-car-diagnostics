import React from 'react';
import { Battery, Car, Thermometer, Settings, Activity } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export const VehicleStatus: React.FC = () => {
  const metrics = [
    {
      id: 1,
      system: 'Battery Health',
      value: '92%',
      status: 'excellent',
      icon: Battery,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10'
    },
    {
      id: 2,
      system: 'Range',
      value: '280 mi',
      status: 'optimal',
      icon: Car,
      color: 'text-blue-400',
      bgColor: 'bg-blue-400/10'
    },
    {
      id: 3,
      system: 'Thermal Management',
      value: 'Optimal',
      status: 'normal',
      icon: Thermometer,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-400/10'
    },
    {
      id: 4,
      system: 'System Performance',
      value: '94%',
      status: 'excellent',
      icon: Settings,
      color: 'text-purple-400',
      bgColor: 'bg-purple-400/10'
    }
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'excellent':
        return 'bg-green-400/20 text-green-300 border-green-400/30';
      case 'optimal':
        return 'bg-blue-400/20 text-blue-300 border-blue-400/30';
      case 'normal':
        return 'bg-emerald-400/20 text-emerald-300 border-emerald-400/30';
      default:
        return 'bg-gray-400/20 text-gray-300 border-gray-400/30';
    }
  };

  return (
    <Card className="bg-gray-900/50 border-gray-800 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-white flex items-center space-x-3 font-light">
          <Activity className="w-5 h-5 text-blue-400" />
          <span>Vehicle Status</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {metrics.map((metric) => (
          <div key={metric.id} className="flex items-center justify-between p-4 bg-gray-800/30 rounded-xl border border-gray-700/50 hover:bg-gray-800/50 transition-all duration-200">
            <div className="flex items-center space-x-4">
              <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${metric.bgColor}`}>
                <metric.icon className={`w-5 h-5 ${metric.color}`} />
              </div>
              <div>
                <p className="text-white font-medium">{metric.system}</p>
                <p className="text-sm text-gray-400">Current Status</p>
              </div>
            </div>
            <div className="text-right space-y-1">
              <p className={`text-xl font-bold ${metric.color}`}>{metric.value}</p>
              <Badge className={`border ${getStatusBadge(metric.status)} font-light text-xs`}>
                {metric.status}
              </Badge>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}; 