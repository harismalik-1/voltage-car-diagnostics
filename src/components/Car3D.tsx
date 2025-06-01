import React, { useRef, useLayoutEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Environment, useGLTF, OrbitControls } from '@react-three/drei';
import { Group, Box3, Vector3, PerspectiveCamera, Scene } from 'three';

// Create a persistent scene instance
const persistentScene = new Scene();

const TeslaModel = () => {
  const groupRef = useRef<Group>(null);
  const { scene } = useGLTF('/models/cybertruck.glb');
  const { camera } = useThree();

  useLayoutEffect(() => {
    if (scene && groupRef.current) {
      // Compute bounding box
      const box = new Box3().setFromObject(scene);
      const size = new Vector3();
      const center = new Vector3();
      box.getSize(size);
      box.getCenter(center);

      // Center the model at world origin
      scene.position.x = -center.x;
      scene.position.y = -center.y;
      scene.position.z = -center.z;

      // Scale the model to fit into a 2.5 unit box
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 2.5 / maxDim;
      scene.scale.setScalar(scale);

      // Ensure camera is at the correct position
      if (camera instanceof PerspectiveCamera) {
        camera.position.set(0, 0, 4);
        camera.lookAt(0, 0, 0);
        camera.updateProjectionMatrix();
      }
    }
  }, [scene, camera]);

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
      <Canvas 
        scene={persistentScene}
        camera={{ position: [0, 0, 4], fov: 40 }}
        gl={{ preserveDrawingBuffer: true }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        <pointLight position={[0, 10, 0]} intensity={0.4} color="#ffffff" />
        <TeslaModel />
        <OrbitControls 
          enableZoom={false} 
          enablePan={false}
          minPolarAngle={Math.PI / 2}
          maxPolarAngle={Math.PI / 2}
          enableRotate={false}
        />
        <Environment preset="city" />
      </Canvas>
    </div>
  );
};
