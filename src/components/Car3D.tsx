import React, { useRef, useLayoutEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment, useGLTF, OrbitControls } from '@react-three/drei';
import { Group, Box3, Vector3 } from 'three';

const TeslaModel = () => {
  const groupRef = useRef<Group>(null);
  const { scene } = useGLTF('/models/cybertruck.glb');

  useLayoutEffect(() => {
    if (scene && groupRef.current) {
      // Compute bounding box
      const box = new Box3().setFromObject(scene);
      const size = new Vector3();
      const center = new Vector3();
      box.getSize(size);
      box.getCenter(center);

      // Center the model
      scene.position.x += (scene.position.x - center.x);
      scene.position.y += (scene.position.y - center.y);
      scene.position.z += (scene.position.z - center.z);

      // Scale the model to fit into a 2.5 unit box
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 2.5 / maxDim;
      scene.scale.setScalar(scale);
    }
  }, [scene]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.3;
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={scene} />
    </group>
  );
};

export const Car3D = () => {
  return (
    <div className="w-full h-64 bg-gradient-to-b from-gray-900 to-black rounded-xl overflow-hidden">
      <Canvas camera={{ position: [0, 0, 4], fov: 40 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        <pointLight position={[0, 10, 0]} intensity={0.4} color="#ffffff" />
        <TeslaModel />
        <OrbitControls enableZoom={false} enablePan={false} />
        <Environment preset="city" />
      </Canvas>
    </div>
  );
};
