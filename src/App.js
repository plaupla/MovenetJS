//1. install dependencies DONE
//2. import dependencies DONE
//3. setup webcam and canvas DONE
//4. define references video DONE
//5. Load movenet
//6. Detect function
//7. Draw function to UI

import React, {useEffect, useRef, useState} from 'react';
import './App.css';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
// Register one of the TF.js backends.
import '@tensorflow/tfjs-backend-webgl';
import Webcam from 'react-webcam';
//follow shohei github
import '@tensorflow/tfjs-backend-wasm';
import * as tfJsConverter from '@tensorflow/tfjs-converter';
//eslint-disable-next-line
import * as tfJsBackendWebgl from '@tensorflow/tfjs-backend-webgl';
// import {isMobile} from 'react-device-detect';

//const and lets from shohei
let detector;
let startInferenceTime,
    numInferences = 0;
let inferenceTimeSum = 0,
    lastPanelUpdate = 0;
let rafId;
let canvasFullScreen;
let ctxFullScreen;
let model;
let modelType;

const VIDEO_CONSTRAINTS = {
  facingMode: "user",
  deviceId: "",
  frameRate: {max: 60, ideal: 30},
  width:640,
  height: 480
};
const MOVENET_CONFIG = {
  maxPoses: 1,
  type: 'lightning',
  scoreThreshold: 0.3
};

const DEFAULT_LINE_WIDTH = 2;
const DEFAULT_RADIUS = 4;


function App() {
  //set up camera and canvas
  const webcamRef = useRef({}); //use shohei's posenetmovenet
  //const canvasRef = useRef(null); //initial using nick renotte vid
  const [cameraReady, setCameraReady] = useState(false);
  const [displayFps, setDisplayFps] = useState(0); //may use for UI similarity calculation later


  //loading posenet?
  useEffect(() => {
    _loadPoseNet().then();

    // eslint-disable-next-line
  }, []);

  const _loadPoseNet = async () => {
    if (rafId) {
        window.cancelAnimationFrame(rafId);
        detector.dispose();
    }

    detector = await createDetector();
    await renderPrediction();
  }

  //load movenet from shohei

  const createDetector = async () => {
    model = poseDetection.SupportedModels.MoveNet;
    modelType = poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING; //or SINGLEPOSE_THUNDER
    return await poseDetection.createDetector(model, {modelType: modelType});
}

  //load Movenet detector from tf github //error on await
  // const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
  // const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);

  // const poses = await detector.estimatePoses(image);

  //error di await. dari tensorflow.movenet.posenet. probably need to make detector? 
//   const createDetector = async () => {
//     model = poseDetection.SupportedModels.MoveNet;
//     modelType = poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING; //or SINGLEPOSE_THUNDER
//     return await poseDetection.createDetector(model, {modelType: modelType});
// }

//testing using posenet JS

//render prediction

const renderPrediction = async () => {
  await renderResult();
  rafId = requestAnimationFrame(renderPrediction);
}

const renderResult = async () => {
  const video = webcamRef.current && webcamRef.current['video'];

  if (!cameraReady && !video) {
      return;
  }

  if (video.readyState < 2) {
      return;
  }

  //start pose estimation
  beginEstimatePosesStats();
  const poses = await detector.estimatePoses(video, {
      maxPoses: MOVENET_CONFIG.maxPoses, //When maxPoses = 1, a single pose is detected
      flipHorizontal: false
  });
  endEstimatePosesStats();
  drawCtxFullScreen(video);

  if (poses.length > 0) {
      drawResultsFullScreen(poses);
  }
}

const beginEstimatePosesStats = () => {
  startInferenceTime = (performance || Date).now();
}

const endEstimatePosesStats = () => {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;
  const panelUpdateMilliseconds = 1000;

  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
      const averageInferenceTime = inferenceTimeSum / numInferences;
      inferenceTimeSum = 0;
      numInferences = 0;
      setDisplayFps(1000.0 / averageInferenceTime, 120);
      lastPanelUpdate = endInferenceTime;
  }
}

//draw keypoints

const drawCtxFullScreen = (video) => {
  canvasFullScreen = document.getElementById('output-full-screen');
  ctxFullScreen = canvasFullScreen.getContext('2d');

  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  video.width = videoWidth;
  video.height = videoHeight;

  canvasFullScreen.width = videoWidth;
  canvasFullScreen.height = videoHeight;
  ctxFullScreen.fillRect(0, 0, videoWidth, videoHeight);

  ctxFullScreen.translate(video.videoWidth, 0);
  ctxFullScreen.scale(-1, 1);
  ctxFullScreen.drawImage(video, 0, 0, videoWidth, videoHeight);
}

//draw results 

const drawResultsFullScreen = (poses) => {
  for (const pose of poses) {
      drawResult(pose);
  }
}

const drawResult = (pose) => {
  if (pose.keypoints != null) {
      drawKeypoints(pose.keypoints);
      drawSkeleton(pose.keypoints);
  }
}

//draw keypoints

const drawKeypoints = (keypoints) => {
  const keypointInd = poseDetection.util.getKeypointIndexBySide(model);
  ctxFullScreen.fillStyle = 'White';
  ctxFullScreen.strokeStyle = 'White';
  ctxFullScreen.lineWidth = DEFAULT_LINE_WIDTH;

  for (const i of keypointInd.middle) {
      drawKeypoint(keypoints[i]);
  }

  ctxFullScreen.fillStyle = 'Green';

  for (const i of keypointInd.left) {
      drawKeypoint(keypoints[i]);
  }

  ctxFullScreen.fillStyle = 'Orange';

  for (const i of keypointInd.right) {
      drawKeypoint(keypoints[i]);
  }
}
const drawKeypoint = (keypoint) => {
  // If score is null, just show the keypoint.
  const score = keypoint.score != null ? keypoint.score : 1;
  const scoreThreshold = MOVENET_CONFIG.scoreThreshold || 0;

  if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, DEFAULT_RADIUS, 0, 2 * Math.PI);
      ctxFullScreen.fill(circle);
      ctxFullScreen.stroke(circle);
  }
}

//draw skeletons
const drawSkeleton = (keypoints) => {
  ctxFullScreen.fillStyle = 'White';
  ctxFullScreen.strokeStyle = 'White';
  ctxFullScreen.lineWidth = DEFAULT_LINE_WIDTH;
  poseDetection.util.getAdjacentPairs(model).forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j]; // If score is null, just show the keypoint.

      const score1 = kp1.score != null ? kp1.score : 1;
      const score2 = kp2.score != null ? kp2.score : 1;
      const scoreThreshold = MOVENET_CONFIG.scoreThreshold || 0;

      if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
          ctxFullScreen.beginPath();
          ctxFullScreen.moveTo(kp1.x, kp1.y);
          ctxFullScreen.lineTo(kp2.x, kp2.y);
          ctxFullScreen.stroke();
      }
  });
}

const onUserMediaError = () => {
  console.log('ERROR in Camera!');
};

const onUserMedia = () => {
  console.log('Camera loaded!');
  setCameraReady(true);
};


  return (
    <div className='App'> 
    <header className='App-header'> 
    <Webcam
                    className="filter blur-lg"
                    style={{visibility: "hidden"}}
                    ref={webcamRef}
                    audio={false}
                    height={640}
                    width={480}
                    videoConstraints={VIDEO_CONSTRAINTS}
                    onUserMediaError={onUserMediaError}
                    onUserMedia={onUserMedia}/>
      {/* <Webcam style={{position:"absolute", marginLeft:"auto", marginRight:"auto",left:0, right:0, textAlign:"center",zIndex:9, width:640, height:488} } ></Webcam>
      <canvas style={{position:"absolute", marginLeft:"auto", marginRight:"auto",left:0, right:0, textAlign:"center",zIndex:9, width:640, height:488}} ></canvas> */}
      <label className="text-white">FPS: {displayFps}</label> 
     </header> 
    </div>
  );
}

export default App;
