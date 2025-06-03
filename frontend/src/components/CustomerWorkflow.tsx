
import React from 'react';
import { MessageSquare, Calendar, Wrench, CheckCircle2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export const CustomerWorkflow = () => {
  const workflowSteps = [
    {
      icon: MessageSquare,
      title: 'Issue Reported',
      description: 'Detected via mobile app',
      status: 'completed',
      time: '10:30 AM',
      color: 'text-green-400',
      bgColor: 'bg-green-400'
    },
    {
      icon: CheckCircle2,
      title: 'Remote Analysis',
      description: 'AI system diagnosis',
      status: 'completed',
      time: '10:32 AM',
      color: 'text-green-400',
      bgColor: 'bg-green-400'
    },
    {
      icon: Wrench,
      title: 'Solution Ready',
      description: 'Software update available',
      status: 'active',
      time: '10:35 AM',
      color: 'text-blue-400',
      bgColor: 'bg-blue-400'
    },
    {
      icon: Calendar,
      title: 'Service Scheduling',
      description: 'Optional technician visit',
      status: 'pending',
      time: 'Pending',
      color: 'text-gray-400',
      bgColor: 'bg-gray-600'
    }
  ];

  return (
    <Card className="bg-gray-900/50 border-gray-800 mt-8 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="text-white font-light text-xl">Customer Experience Pipeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-8">
          {/* Timeline */}
          <div className="relative">
            <div className="flex items-center justify-between">
              {workflowSteps.map((step, index) => (
                <div key={index} className="flex flex-col items-center space-y-3 flex-1">
                  <div className={`flex items-center justify-center w-12 h-12 rounded-full border-2 ${
                    step.status === 'completed' ? 'border-green-400 bg-green-400/20' :
                    step.status === 'active' ? 'border-blue-400 bg-blue-400/20' : 
                    'border-gray-600 bg-gray-600/20'
                  } transition-all duration-300`}>
                    <step.icon className={`w-6 h-6 ${step.color}`} />
                  </div>
                  
                  <div className="text-center max-w-32">
                    <p className="text-white font-medium text-sm">{step.title}</p>
                    <p className="text-xs text-gray-400 mt-1">{step.description}</p>
                    <p className="text-xs text-gray-500 mt-1">{step.time}</p>
                  </div>
                  
                  {index < workflowSteps.length - 1 && (
                    <div className="absolute top-6 w-full h-0.5 bg-gray-700" 
                         style={{ 
                           left: `${((index + 1) / workflowSteps.length) * 100}%`,
                           width: `${(1 / workflowSteps.length) * 100}%`
                         }}>
                      <div className={`h-full ${
                        step.status === 'completed' ? 'bg-green-400' : 'bg-gray-700'
                      } transition-all duration-500`} 
                           style={{ 
                             width: step.status === 'completed' ? '100%' : '0%'
                           }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3 justify-center pt-6 border-t border-gray-800">
            <Button className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white border-0 font-light px-6">
              Apply Update
            </Button>
            <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white font-light px-6">
              Schedule Service
            </Button>
            <Button variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white font-light px-6">
              Contact Customer
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
