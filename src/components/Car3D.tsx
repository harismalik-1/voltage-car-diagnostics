
import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment } from '@react-three/drei';
import { Group } from 'three';

const CarModel = () => {
  const groupRef = useRef<Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.3;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Main Car Body - Lower */}
      <mesh position={[0, 0.4, 0]}>
        <boxGeometry args={[4.2, 0.8, 1.9]} />
        <meshStandardMaterial color="#ffffff" metalness={0.9} roughness={0.1} />
      </mesh>
      
      {/* Car Hood */}
      <mesh position={[1.5, 0.5, 0]} rotation={[0, 0, -0.05]}>
        <boxGeometry args={[1.2, 0.6, 1.8]} />
        <meshStandardMaterial color="#ffffff" metalness={0.9} roughness={0.1} />
      </mesh>
      
      {/* Car Roof/Cabin - Tesla-like sleek design */}
      <mesh position={[-0.2, 1.1, 0]} rotation={[0, 0, 0.02]}>
        <boxGeometry args={[2.8, 0.7, 1.7]} />
        <meshStandardMaterial color="#ffffff" metalness={0.9} roughness={0.1} />
      </mesh>
      
      {/* Windshield */}
      <mesh position={[0.8, 1.3, 0]} rotation={[0, 0, -0.15]}>
        <boxGeometry args={[1.2, 0.6, 1.6]} />
        <meshStandardMaterial color="#87ceeb" metalness={0.1} roughness={0.0} transparent opacity={0.7} />
      </mesh>
      
      {/* Rear Window */}
      <mesh position={[-1.2, 1.2, 0]} rotation={[0, 0, 0.1]}>
        <boxGeometry args={[0.8, 0.5, 1.6]} />
        <meshStandardMaterial color="#87ceeb" metalness={0.1} roughness={0.0} transparent opacity={0.7} />
      </mesh>
      
      {/* Front Wheels */}
      <mesh position={[1.4, 0, 1.0]} rotation={[Math.PI/2, 0, 0]}>
        <cylinderGeometry args={[0.35, 0.35, 0.25]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.2} roughness={0.8} />
      </mesh>
      <mesh position={[1.4, 0, -1.0]} rotation={[Math.PI/2, 0, 0]}>
        <cylinderGeometry args={[0.35, 0.35, 0.25]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.2} roughness={0.8} />
      </mesh>
      
      {/* Rear Wheels */}
      <mesh position={[-1.4, 0, 1.0]} rotation={[Math.PI/2, 0, 0]}>
        <cylinderGeometry args={[0.35, 0.35, 0.25]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.2} roughness={0.8} />
      </mesh>
      <mesh position={[-1.4, 0, -1.0]} rotation={[Math.PI/2, 0, 0]}>
        <cylinderGeometry args={[0.35, 0.35, 0.25]} />
        <meshStandardMaterial color="#1a1a1a" metalness={0.2} roughness={0.8} />
      </mesh>
      
      {/* Tesla-style Front Bumper */}
      <mesh position={[2.2, 0.3, 0]}>
        <boxGeometry args={[0.15, 0.4, 1.6]} />
        <meshStandardMaterial color="#f0f0f0" metalness={0.8} roughness={0.2} />
      </mesh>
      
      {/* Tesla Headlights - More elongated */}
      <mesh position={[2.1, 0.6, 0.7]}>
        <boxGeometry args={[0.1, 0.2, 0.4]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.6} />
      </mesh>
      <mesh position={[2.1, 0.6, -0.7]}>
        <boxGeometry args={[0.1, 0.2, 0.4]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.6} />
      </mesh>
      
      {/* Tesla Logo Area */}
      <mesh position={[2.15, 0.5, 0]}>
        <boxGeometry args={[0.05, 0.1, 0.1]} />
        <meshStandardMaterial color="#cc0000" emissive="#cc0000" emissiveIntensity={0.3} />
      </mesh>
      
      {/* Side Mirrors */}
      <mesh position={[0.5, 1.0, 1.0]}>
        <boxGeometry args={[0.15, 0.1, 0.1]} />
        <meshStandardMaterial color="#ffffff" metalness={0.9} roughness={0.1} />
      </mesh>
      <mesh position={[0.5, 1.0, -1.0]}>
        <boxGeometry args={[0.15, 0.1, 0.1]} />
        <meshStandardMaterial color="#ffffff" metalness={0.9} roughness={0.1} />
      </mesh>
      
      {/* Door Handles */}
      <mesh position={[0, 0.7, 1.0]}>
        <boxGeometry args={[0.3, 0.05, 0.05]} />
        <meshStandardMaterial color="#e0e0e0" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh position={[0, 0.7, -1.0]}>
        <boxGeometry args={[0.3, 0.05, 0.05]} />
        <meshStandardMaterial color="#e0e0e0" metalness={0.8} roughness={0.2} />
      </mesh>
      
      {/* Rear Lights */}
      <mesh position={[-2.1, 0.6, 0.8]}>
        <boxGeometry args={[0.05, 0.15, 0.2]} />
        <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.4} />
      </mesh>
      <mesh position={[-2.1, 0.6, -0.8]}>
        <boxGeometry args={[0.05, 0.15, 0.2]} />
        <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={0.4} />
      </mesh>
    </group>
  );
};

export const Car3D = () => {
  return (
    <div className="w-full h-64 bg-gradient-to-b from-gray-900 to-black rounded-xl overflow-hidden">
      <Canvas camera={{ position: [6, 4, 6], fov: 50 }}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={1.2} />
        <pointLight position={[-10, -10, -10]} intensity={0.6} />
        <pointLight position={[0, 10, 0]} intensity={0.4} color="#ffffff" />
        <CarModel />
        <Environment preset="city" />
      </Canvas>
    </div>
  );
};
