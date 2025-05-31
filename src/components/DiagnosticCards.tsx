
import React from 'react';
import { AlertTriangle, CheckCircle, Clock, Wrench } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export const DiagnosticCards = () => {
  const diagnostics = [
    {
      id: 1,
      system: 'Autopilot Cameras',
      status: 'active',
      lastCheck: '2 minutes ago',
      issues: 0,
      icon: CheckCircle,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10'
    },
    {
      id: 2,
      system: 'Charging System',
      status: 'warning',
      lastCheck: '5 minutes ago',
      issues: 1,
      icon: AlertTriangle,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-400/10'
    },
    {
      id: 3,
      system: 'Drive Unit',
      status: 'active',
      lastCheck: '1 minute ago',
      issues: 0,
      icon: CheckCircle,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10'
    },
    {
      id: 4,
      system: 'Climate Control',
      status: 'maintenance',
      lastCheck: '10 minutes ago',
      issues: 2,
      icon: Wrench,
      color: 'text-orange-400',
      bgColor: 'bg-orange-400/10'
    }
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-400/20 text-green-300 border-green-400/30';
      case 'warning':
        return 'bg-yellow-400/20 text-yellow-300 border-yellow-400/30';
      case 'maintenance':
        return 'bg-orange-400/20 text-orange-300 border-orange-400/30';
      default:
        return 'bg-gray-400/20 text-gray-300 border-gray-400/30';
    }
  };

  return (
    <Card className="bg-gray-900/50 border-gray-800 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-white flex items-center space-x-3 font-light">
          <Clock className="w-5 h-5 text-blue-400" />
          <span>Real-time Diagnostics</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {diagnostics.map((item) => (
          <div key={item.id} className="flex items-center justify-between p-4 bg-gray-800/30 rounded-xl border border-gray-700/50 hover:bg-gray-800/50 transition-all duration-200">
            <div className="flex items-center space-x-4">
              <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${item.bgColor}`}>
                <item.icon className={`w-5 h-5 ${item.color}`} />
              </div>
              <div>
                <p className="text-white font-medium">{item.system}</p>
                <p className="text-sm text-gray-400">Last check: {item.lastCheck}</p>
              </div>
            </div>
            <div className="text-right space-y-1">
              <Badge className={`border ${getStatusBadge(item.status)} font-light text-xs`}>
                {item.status}
              </Badge>
              {item.issues > 0 && (
                <p className="text-xs text-red-400">{item.issues} issue(s)</p>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};
