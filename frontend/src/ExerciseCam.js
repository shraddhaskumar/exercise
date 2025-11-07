import React, { useRef, useEffect, useState } from "react";
import * as poseDetection from "@mediapipe/pose";
import * as cam from "@mediapipe/camera_utils";

const ExerciseCam = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [exercise, setExercise] = useState("arm_raise");
  const [instruction, setInstruction] = useState("Stand straight to begin...");
  const [timer, setTimer] = useState(0);
  const [holding, setHolding] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  const [resting, setResting] = useState(false);
  const [breakTimer, setBreakTimer] = useState(0);
  const [holdCompleted, setHoldCompleted] = useState(false);
  const [armLowered, setArmLowered] = useState(false);

  const holdDuration = 10; // seconds
  const breakDuration = 5; // seconds

  // Enable voice after first click
  useEffect(() => {
    const enableVoice = () => setVoiceEnabled(true);
    window.addEventListener("click", enableVoice);
    return () => window.removeEventListener("click", enableVoice);
  }, []);

  // Speak helper
  const speak = (text) => {
    if (!voiceEnabled) return;
    const synth = window.speechSynthesis;
    if (!synth) return;
    synth.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.pitch = 1;
    utter.rate = 1;
    utter.volume = 1;
    synth.speak(utter);
  };

  // Speak whenever instruction changes
  useEffect(() => {
    if (instruction) speak(instruction);
  }, [instruction]);

  // Setup MediaPipe Pose
  useEffect(() => {
    const pose = new poseDetection.Pose({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
    pose.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    pose.onResults(onResults);

    if (typeof cam.Camera !== "undefined") {
      const camera = new cam.Camera(videoRef.current, {
        onFrame: async () => {
          await pose.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, [exercise]);

  const onResults = (results) => {
    if (!results.poseLandmarks) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    checkExercisePose(results.poseLandmarks);

    ctx.restore();
  };

  // Calculate joint angle
  const getAngle = (a, b, c) => {
    const ab = [a.x - b.x, a.y - b.y];
    const cb = [c.x - b.x, c.y - b.y];
    const dot = ab[0] * cb[0] + ab[1] * cb[1];
    const magAB = Math.sqrt(ab[0] ** 2 + ab[1] ** 2);
    const magCB = Math.sqrt(cb[0] ** 2 + cb[1] ** 2);
    const angle = Math.acos(dot / (magAB * magCB));
    return (angle * 180) / Math.PI;
  };

  // Posture logic
  const checkExercisePose = (lm) => {
    if (exercise === "arm_raise") {
      const leftElbow = lm[13];
      const leftShoulder = lm[11];
      const leftHip = lm[23];
      const shoulderAngle = getAngle(leftElbow, leftShoulder, leftHip);

      if (shoulderAngle < 50) {
        // Arm lowered
        if (!armLowered) {
          setArmLowered(true);
        }
        
        if (holdCompleted && !resting) {
          setResting(true);
          setBreakTimer(0);
          setHoldCompleted(false);
        } else if (!resting && !holding) {
          if (instruction !== "Raise your arm upward!") {
            setInstruction("Raise your arm upward!");
          }
        }
        setHolding(false);
        setTimer(0);
      } else {
        // Arm raised
        setArmLowered(false);
        if (!resting) {
          if (!holding) {
            setInstruction("Good! Hold it there...");
            setHolding(true);
          }
          setTimer((t) => {
            const next = t + 0.1;
            if (next >= holdDuration) {
              setHoldCompleted(true);
            }
            return next >= holdDuration ? holdDuration : next;
          });
        }
      }
    }

    if (exercise === "side_stretch") {
      const leftShoulder = lm[11];
      const leftHip = lm[23];
      const rightHip = lm[24];
      const hipAngle = getAngle(leftShoulder, leftHip, rightHip);

      if (hipAngle < 160) {
        // Side bend detected
        if (!resting) {
          if (!holding) {
            setInstruction("Good side bend! Hold steady...");
            setHolding(true);
          }
          setTimer((t) => {
            const next = t + 0.1;
            if (next >= holdDuration) {
              setHoldCompleted(true);
            }
            return next >= holdDuration ? holdDuration : next;
          });
        }
      } else {
        // Back to neutral
        if (holdCompleted && !resting) {
          setResting(true);
          setBreakTimer(0);
          setHoldCompleted(false);
        } else if (!resting) {
          if (instruction !== "Lean gently to your side.") {
            setInstruction("Lean gently to your side.");
          }
        }
        setHolding(false);
        setTimer(0);
      }
    }
  };

  // Break timer logic
  useEffect(() => {
    if (resting) {
      setInstruction("");
      
      const interval = setInterval(() => {
        setBreakTimer((t) => {
          const next = t + 1;
          if (next >= breakDuration) {
            clearInterval(interval);
            setResting(false);
            setTimer(0);
            setHolding(false);
            if (exercise === "arm_raise") {
              setInstruction("Raise your arm upward!");
            } else if (exercise === "side_stretch") {
              setInstruction("Lean gently to your side.");
            }
            return breakDuration;
          }
          return next;
        });
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [resting, exercise]);

  // Check for arm lowering to start relaxation timer
  useEffect(() => {
    if (exercise === "arm_raise" && armLowered && holdCompleted && !resting) {
      setResting(true);
      setBreakTimer(0);
      setHoldCompleted(false);
    }
  }, [armLowered, holdCompleted, resting, exercise]);

  return (
    <div
      style={{
        textAlign: "center",
        marginTop: 20,
        fontFamily: "sans-serif",
        color: "#222",
      }}
    >
      <h2>AI Stress-Relief Coach 🧘‍♀️</h2>

      {!voiceEnabled && (
        <p style={{ color: "gray" }}>
          🔊 Click anywhere on screen to enable voice feedback
        </p>
      )}

      <select
        value={exercise}
        onChange={(e) => {
          setExercise(e.target.value);
          setInstruction("Get ready...");
          setTimer(0);
          setHolding(false);
          setResting(false);
          setBreakTimer(0);
          setHoldCompleted(false);
          setArmLowered(false);
        }}
        style={{
          padding: "10px",
          fontSize: "1em",
          borderRadius: "8px",
          marginBottom: "10px",
        }}
      >
        <option value="arm_raise">Arm Stretch</option>
        <option value="side_stretch">Side Stretch</option>
      </select>

      <p style={{ fontSize: "1.2em", marginTop: "10px" }}>{instruction}</p>

      {holding && (
        <p style={{ fontSize: "1.5em", color: "green" }}>
          Hold Timer: {timer.toFixed(1)}s
        </p>
      )}

      {resting && (
        <p style={{ fontSize: "1.5em", color: "blue" }}>
          Relax Time: {breakTimer}s / {breakDuration}s
        </p>
      )}

      <video ref={videoRef} style={{ display: "none" }}></video>
      <canvas
        ref={canvasRef}
        width="640"
        height="480"
        style={{
          borderRadius: "12px",
          boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
          marginTop: "10px",
        }}
      />
    </div>
  );
};

export default ExerciseCam;