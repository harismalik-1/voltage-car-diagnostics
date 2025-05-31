
import React from 'react';
import { Brain, TrendingUp, Cpu } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

export const MLInsights = () => {
  const insights = [
    {
      title: 'Predictive Maintenance',
      description: 'Next service recommendation in 2,847 miles',
      confidence: 94,
      icon: Brain,
      color: 'text-purple-400',
      bgColor: 'bg-purple-400/10'
    },
    {
      title: 'Performance Optimization',
      description: 'Battery efficiency can be improved by 3.2%',
      confidence: 87,
      icon: TrendingUp,
      color: 'text-blue-400',
      bgColor: 'bg-blue-400/10'
    },
    {
      title: 'Anomaly Detection',
      description: 'No unusual patterns detected',
      confidence: 99,
      icon: Cpu,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10'
    }
  ];

  return (
    <Card className="bg-gray-900/50 border-gray-800 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-white flex items-center space-x-3 font-light">
          <Brain className="w-5 h-5 text-purple-400" />
          <span>AI Insights</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {insights.map((insight, index) => (
          <div key={index} className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${insight.bgColor}`}>
                <insight.icon className={`w-5 h-5 ${insight.color}`} />
              </div>
              <div className="flex-1">
                <p className="text-white font-medium">{insight.title}</p>
                <p className="text-sm text-gray-400">{insight.description}</p>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Confidence</span>
                <span className={insight.color}>{insight.confidence}%</span>
              </div>
              <div className="relative">
                <Progress 
                  value={insight.confidence} 
                  className="h-2 bg-gray-800"
                />
                <div 
                  className={`absolute top-0 left-0 h-2 rounded-full bg-gradient-to-r ${
                    insight.confidence > 95 ? 'from-green-400 to-emerald-500' :
                    insight.confidence > 85 ? 'from-blue-400 to-purple-500' :
                    'from-yellow-400 to-orange-500'
                  }`}
                  style={{ width: `${insight.confidence}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};
