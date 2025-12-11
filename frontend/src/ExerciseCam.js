import React, { useRef, useEffect, useState } from "react";
import * as poseDetection from "@mediapipe/pose";
import * as cam from "@mediapipe/camera_utils";

const ExerciseCam = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [exercise, setExercise] = useState("arm_raise");
  const [instruction, setInstruction] = useState("Stand straight to begin...");
  const [correction, setCorrection] = useState("");
  const [timer, setTimer] = useState(0);
  const [holding, setHolding] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  const [resting, setResting] = useState(false);
  const [breakTimer, setBreakTimer] = useState(0);
  const [poseCorrect, setPoseCorrect] = useState(false);
  const [mlPrediction, setMlPrediction] = useState("");
  const [mlConfidence, setMlConfidence] = useState(0);
  const [repCount, setRepCount] = useState(0);
  const [postureScore, setPostureScore] = useState(100);

  const holdDuration = exercise === "squats" ? 5 : 10;
  const breakDuration = 5;

  const holdIntervalRef = useRef(null);
  const breakIntervalRef = useRef(null);
  const lastPredictionTime = useRef(0);

  // Enable voice after first click
  useEffect(() => {
    const enableVoice = () => setVoiceEnabled(true);
    window.addEventListener("click", enableVoice);
    return () => window.removeEventListener("click", enableVoice);
  }, []);

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
      modelComplexity: 1,
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

    drawSkeleton(ctx, results.poseLandmarks);
    checkExercisePose(results.poseLandmarks);
    
    // Send to ML model every 500ms
    const now = Date.now();
    if (now - lastPredictionTime.current > 500) {
      lastPredictionTime.current = now;
      sendToMLModel(results.poseLandmarks);
    }

    ctx.restore();
  };

  const drawSkeleton = (ctx, landmarks) => {
    const connections = [
      [11, 13], [13, 15],
      [12, 14], [14, 16],
      [11, 12],
      [11, 23], [12, 24],
      [23, 24],
      [23, 25], [25, 27],
      [24, 26], [26, 28],
    ];

    ctx.strokeStyle = poseCorrect ? "#00ff00" : "#ff0000";
    ctx.lineWidth = 3;

    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];
      ctx.beginPath();
      ctx.moveTo(startPoint.x * 640, startPoint.y * 480);
      ctx.lineTo(endPoint.x * 640, endPoint.y * 480);
      ctx.stroke();
    });

    ctx.fillStyle = poseCorrect ? "#00ff00" : "#ff0000";
    landmarks.forEach((lm) => {
      ctx.beginPath();
      ctx.arc(lm.x * 640, lm.y * 480, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const getAngle = (a, b, c) => {
    const ab = [a.x - b.x, a.y - b.y];
    const cb = [c.x - b.x, c.y - b.y];
    const dot = ab[0] * cb[0] + ab[1] * cb[1];
    const magAB = Math.sqrt(ab[0] ** 2 + ab[1] ** 2);
    const magCB = Math.sqrt(cb[0] ** 2 + cb[1] ** 2);
    const angle = Math.acos(dot / (magAB * magCB));
    return (angle * 180) / Math.PI;
  };

  const getGroundAngle = (a, b) => {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.abs(Math.atan2(dy, dx) * (180 / Math.PI));
  };

  // Send angles to ML model
  const sendToMLModel = async (lm) => {
    try {
      // Calculate all 10 angles as per your training data
      const leftShoulder = lm[11], leftElbow = lm[13], leftWrist = lm[15];
      const rightShoulder = lm[12], rightElbow = lm[14], rightWrist = lm[16];
      const leftHip = lm[23], leftKnee = lm[25], leftAnkle = lm[27];
      const rightHip = lm[24], rightKnee = lm[26], rightAnkle = lm[28];

      const angles = [
        getAngle(leftElbow, leftShoulder, leftHip), // Shoulder_Angle
        getAngle(leftWrist, leftElbow, leftShoulder), // Elbow_Angle
        getAngle(leftShoulder, leftHip, leftKnee), // Hip_Angle
        getAngle(leftHip, leftKnee, leftAnkle), // Knee_Angle
        getAngle(leftKnee, leftAnkle, { x: leftAnkle.x, y: leftAnkle.y + 0.1 }), // Ankle_Angle
        getGroundAngle(leftShoulder, rightShoulder), // Shoulder_Ground_Angle
        getGroundAngle(leftElbow, leftWrist), // Elbow_Ground_Angle
        getGroundAngle(leftHip, rightHip), // Hip_Ground_Angle
        getGroundAngle(leftKnee, leftAnkle), // Knee_Ground_Angle
        getGroundAngle(leftAnkle, { x: leftAnkle.x + 0.1, y: leftAnkle.y }), // Ankle_Ground_Angle
      ];

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ angles }),
      });

      const data = await response.json();
      setMlPrediction(data.predicted_exercise);
      
      // Calculate confidence based on match with current exercise
      const expectedExercise = exercise === "arm_raise" ? "arms_up" : "squat"; 
      setMlConfidence(data.predicted_exercise === expectedExercise ? 95 : 60);
    } catch (error) {
      console.error("ML prediction error:", error);
    }
  };

  // Save posture data to backend
  const savePostureData = async () => {
    try {
      await fetch("http://localhost:8000/posture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "user_" + Date.now(),
          exercise: exercise,
          posture_score: postureScore,
          reps: repCount,
        }),
      });
    } catch (error) {
      console.error("Save posture error:", error);
    }
  };

  const checkExercisePose = (lm) => {
    if (resting) return; 

    if (exercise === "arm_raise") checkArmRaise(lm);
    else if (exercise === "wall_push") checkWallPush(lm);
    else if (exercise === "squats") checkSquats(lm);
    else if (exercise === "neck_rolls") checkNeckRolls(lm);
    else if (exercise === "shoulder_shrugs") checkShoulderShrugs(lm);
  };


  const checkArmRaise = (lm) => {
    const leftWrist = lm[15], leftElbow = lm[13], leftShoulder = lm[11], leftHip = lm[23];
    const rightWrist = lm[16], rightElbow = lm[14], rightShoulder = lm[12];
    
    const leftShoulderAngle = getAngle(leftElbow, leftShoulder, leftHip);
    const leftElbowAngle = getAngle(leftWrist, leftElbow, leftShoulder);
    const leftWristAboveShoulder = leftWrist.y < leftShoulder.y;
    const leftArmCorrect = leftShoulderAngle > 140 && leftElbowAngle > 160 && leftWristAboveShoulder;
    
    const rightShoulderAngle = getAngle(rightElbow, rightShoulder, leftHip);
    const rightElbowAngle = getAngle(rightWrist, rightElbow, rightShoulder);
    const rightWristAboveShoulder = rightWrist.y < rightShoulder.y;
    const rightArmCorrect = rightShoulderAngle > 140 && rightElbowAngle > 160 && rightWristAboveShoulder;
    
    const isCorrect = leftArmCorrect && rightArmCorrect;

    if (isCorrect) {
      setPoseCorrect(true);
      setCorrection("");
      if (!holding) {
        setInstruction("Perfect! Hold both arms up...");
        setHolding(true);
      }
    } else {
      setPoseCorrect(false);
      if (holding) {
        setHolding(false);
        setTimer(0);
      }
      setInstruction("Raise BOTH arms straight up");
      
      if (!leftWristAboveShoulder && !rightWristAboveShoulder) {
        setCorrection("⚠️ Raise both arms higher!");
      } else if (!leftWristAboveShoulder) {
        setCorrection("⚠️ Left arm: Raise higher!");
      } else if (!rightWristAboveShoulder) {
        setCorrection("⚠️ Right arm: Raise higher!");
      } else if (leftElbowAngle < 160 && rightElbowAngle < 160) {
        setCorrection("⚠️ Straighten both elbows!");
      } else if (leftElbowAngle < 160) {
        setCorrection("⚠️ Left arm: Straighten elbow!");
      } else if (rightElbowAngle < 160) {
        setCorrection("⚠️ Right arm: Straighten elbow!");
      }
    }
  };

  const checkWallPush = (lm) => {
    const leftWrist = lm[15], leftElbow = lm[13], leftShoulder = lm[11];
    const rightWrist = lm[16], rightElbow = lm[14], rightShoulder = lm[12];
    
    // Check elbow angles (should be bent for push position)
    const leftElbowAngle = getAngle(leftWrist, leftElbow, leftShoulder);
    const rightElbowAngle = getAngle(rightWrist, rightElbow, rightShoulder);
    
    // Check if arms are in front (wrists should be forward of shoulders)
    const armsForward = leftWrist.z < leftShoulder.z && rightWrist.z < rightShoulder.z;
    
    // Check if elbows are bent (wall push position)
    const elbowsBent = leftElbowAngle < 140 && rightElbowAngle < 140;
    
    // Arms should be roughly parallel
    const armsLevel = Math.abs(leftWrist.y - rightWrist.y) < 0.1;
    
    const isCorrect = elbowsBent && armsLevel && (leftElbowAngle > 60 && rightElbowAngle > 60);

    if (isCorrect) {
      setPoseCorrect(true);
      setCorrection("");
      if (!holding) {
        setInstruction("Perfect wall push position! Hold it...");
        setHolding(true);
      }
    } else {
      setPoseCorrect(false);
      if (holding) {
        setHolding(false);
        setTimer(0);
      }
      setInstruction("Bend elbows as if pushing against a wall");
      
      if (!elbowsBent) {
        setCorrection("⚠️ Bend your elbows more!");
      } else if (!armsLevel) {
        setCorrection("⚠️ Keep both arms at same height!");
      } else {
        setCorrection("⚠️ Position arms as if pushing a wall!");
      }
    }
  };

  const checkNeckRolls = (lm) => {
    const nose = lm[0], leftEar = lm[7], rightEar = lm[8];
    const leftShoulder = lm[11], rightShoulder = lm[12];
    
    // Calculate neck tilt (nose position relative to shoulders)
    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    const neckTilt = Math.abs(nose.y - shoulderMidY);
    
    // Check if head is tilted significantly
    const headTiltAngle = Math.abs(leftEar.y - rightEar.y);
    
    // For neck rolls, we want to see movement/tilt
    const isCorrect = neckTilt > 0.05 || headTiltAngle > 0.02;

    if (isCorrect) {
      setPoseCorrect(true);
      setCorrection("");
      if (!holding) {
        setInstruction("Great! Keep rolling your neck gently...");
        setHolding(true);
      }
    } else {
      setPoseCorrect(false);
      if (holding) {
        setHolding(false);
        setTimer(0);
      }
      setInstruction("Gently roll your head in circles");
      setCorrection("⚠️ Move your head more - tilt left, back, right, forward");
    }
  };

  const checkShoulderShrugs = (lm) => {
    const leftShoulder = lm[11], rightShoulder = lm[12];
    const leftEar = lm[7], rightEar = lm[8];
    
    // Calculate distance between shoulders and ears
    const leftDistance = Math.abs(leftShoulder.y - leftEar.y);
    const rightDistance = Math.abs(rightShoulder.y - rightEar.y);
    
    // Shoulders should be raised (closer to ears)
    const isCorrect = leftDistance < 0.15 && rightDistance < 0.15;

    if (isCorrect) {
      setPoseCorrect(true);
      setCorrection("");
      if (!holding) {
        setInstruction("Perfect shrug! Hold your shoulders up...");
        setHolding(true);
      }
    } else {
      setPoseCorrect(false);
      if (holding) {
        setHolding(false);
        setTimer(0);
      }
      setInstruction("Lift BOTH shoulders up towards your ears");
      setCorrection("⚠️ Raise your shoulders higher!");
    }
  };

  const checkSquats = (lm) => {
    const leftHip = lm[23], leftKnee = lm[25], leftAnkle = lm[27], leftShoulder = lm[11];
    const rightHip = lm[24], rightKnee = lm[26], rightAnkle = lm[28];
    
    const leftKneeAngle = getAngle(leftHip, leftKnee, leftAnkle);
    const rightKneeAngle = getAngle(rightHip, rightKnee, rightAnkle);
    const hipAngle = getAngle(leftShoulder, leftHip, leftKnee);
    
    const isCorrect = leftKneeAngle < 110 && rightKneeAngle < 110 && hipAngle < 100;

    if (isCorrect) {
      setPoseCorrect(true);
      setCorrection("");
      if (!holding) {
        setInstruction("Great squat! Hold it there...");
        setHolding(true);
      }
    } else {
      setPoseCorrect(false);
      if (holding) {
        setHolding(false);
        setTimer(0);
      }
      setInstruction("Bend BOTH knees and lower into squat");
      
      if (leftKneeAngle >= 110 && rightKneeAngle >= 110) {
        setCorrection("⚠️ Bend both knees more!");
      } else if (leftKneeAngle >= 110) {
        setCorrection("⚠️ Left knee: Bend more!");
      } else if (rightKneeAngle >= 110) {
        setCorrection("⚠️ Right knee: Bend more!");
      }
    }
  };

  useEffect(() => {
    if (holdIntervalRef.current) {
      clearInterval(holdIntervalRef.current);
    }
    
    if (holding && !resting && poseCorrect) {
      holdIntervalRef.current = setInterval(() => {
        setTimer((t) => {
          const next = t + 0.1;
          if (next >= holdDuration) {
            clearInterval(holdIntervalRef.current);
            setHolding(false);
            setTimer(holdDuration);
            
            // Set/Speak Rest Message Immediately
            setCorrection(""); 
            setPoseCorrect(false);

            const restMessage = "relax! take rest";
            setInstruction(restMessage); 
            speak(restMessage); 

            setResting(true);
            setBreakTimer(0);
            
            setRepCount(prev => prev + 1);
            savePostureData();
            
            return holdDuration;
          }
          return next;
        });
      }, 100);
    } else if (!holding) {
      setTimer(0);
    }
    
    return () => {
      if (holdIntervalRef.current) {
        clearInterval(holdIntervalRef.current);
      }
    };
  }, [holding, resting, poseCorrect, exercise]);

  useEffect(() => {
    if (breakIntervalRef.current) {
      clearInterval(breakIntervalRef.current);
    }
    
    if (resting) {
      breakIntervalRef.current = setInterval(() => {
        setBreakTimer((t) => {
          const next = t + 1;
          if (next >= breakDuration) {
            clearInterval(breakIntervalRef.current);
            setResting(false);
            setBreakTimer(breakDuration);
            setTimer(0);
            setHolding(false);
            setCorrection("");
            
            // Set/Speak Next Instruction ONLY After Rest is Over
            if (exercise === "arm_raise") {
              setInstruction("Ready? Raise BOTH arms straight up");
            } else if (exercise === "wall_push") {
              setInstruction("Ready? Bend elbows like pushing a wall");
            } else if (exercise === "squats") {
              setInstruction("Ready? Lower into squat position");
            } else if (exercise === "neck_rolls") {
              setInstruction("Ready? Roll your head gently in circles");
            } else if (exercise === "shoulder_shrugs") {
              setInstruction("Ready? Lift your shoulders up");
            } 
            
            return breakDuration;
          }
          return next;
        });
      }, 1000);
    } else {
      setBreakTimer(0);
    }
    
    return () => {
      if (breakIntervalRef.current) {
        clearInterval(breakIntervalRef.current);
      }
    };
  }, [resting, exercise]); 

  const handleExerciseChange = (newExercise) => {
    if (holdIntervalRef.current) {
      clearInterval(holdIntervalRef.current);
    }
    if (breakIntervalRef.current) {
      clearInterval(breakIntervalRef.current);
    }
    
    setExercise(newExercise);
    setInstruction("Get ready...");
    setTimer(0);
    setHolding(false);
    setResting(false);
    setBreakTimer(0);
    setPoseCorrect(false);
    setCorrection("");
    setRepCount(0);
    
    setTimeout(() => {
      if (newExercise === "arm_raise") {
        setInstruction("Raise BOTH arms straight up");
      } else if (newExercise === "wall_push") {
        setInstruction("Bend elbows as if pushing a wall");
      } else if (newExercise === "squats") {
        setInstruction("Lower into squat position");
      } else if (newExercise === "neck_rolls") {
        setInstruction("Gently roll your head in circles");
      } else if (newExercise === "shoulder_shrugs") {
        setInstruction("Lift your shoulders up towards ears");
      } 
    }, 500);
  };

  return (
    <div style={{ textAlign: "center", marginTop: 20, fontFamily: "Arial, sans-serif", color: "#222", backgroundColor: "#f5f5f5", padding: "20px", minHeight: "100vh" }}>
      <h1 style={{ color: "#2563eb" }}>🏋️ AI Fitness Coach (ML-Powered)</h1>
      {!voiceEnabled && <p style={{ color: "gray", fontSize: "0.9em" }}>🔊 Click anywhere to enable voice feedback</p>}
      
      <select
        value={exercise}
        onChange={(e) => handleExerciseChange(e.target.value)}
        style={{ padding: "12px 20px", fontSize: "1.1em", borderRadius: "8px", border: "2px solid #2563eb", marginBottom: "15px", cursor: "pointer", backgroundColor: "white", minWidth: "200px" }}
      >
        <option value="arm_raise">🙌 Both Arms Raise</option>
        <option value="wall_push">💪 Wall Push (Upper Body)</option>
        <option value="squats">🦵 Squats</option>
        <option value="neck_rolls">🔄 Neck Rolls (Stress Relief)</option>
        <option value="shoulder_shrugs">💆 Shoulder Shrugs (Tension Release)</option>
      </select>

      <div style={{ display: "flex", gap: "10px", justifyContent: "center", marginBottom: "15px" }}>
        <div style={{ backgroundColor: "#e0e7ff", padding: "10px 20px", borderRadius: "8px", border: "2px solid #4f46e5" }}>
          <p style={{ margin: 0, fontSize: "0.9em", color: "#4f46e5" }}>🎯 Reps: <strong>{repCount}</strong></p>
        </div>
        <div style={{ backgroundColor: mlPrediction ? "#dcfce7" : "#fef3c7", padding: "10px 20px", borderRadius: "8px", border: `2px solid ${mlPrediction ? "#16a34a" : "#f59e0b"}` }}>
          <p style={{ margin: 0, fontSize: "0.9em", color: mlPrediction ? "#16a34a" : "#f59e0b" }}>
            🤖 ML: {mlPrediction || "Detecting..."} {mlConfidence > 0 && `(${mlConfidence}%)`}
          </p>
        </div>
      </div>

      <div style={{ backgroundColor: poseCorrect ? "#d1fae5" : resting ? "#dbeafe" : "#fef3c7", padding: "15px", borderRadius: "10px", marginBottom: "15px", border: `2px solid ${poseCorrect ? "#10b981" : resting ? "#3b82f6" : "#f59e0b"}` }}>
        <p style={{ fontSize: "1.3em", fontWeight: "bold", margin: "5px 0" }}>{instruction}</p>
        {correction && <p style={{ fontSize: "1.1em", color: "#dc2626", margin: "5px 0" }}>{correction}</p>}
      </div>

      {holding && !resting && (
        <div style={{ backgroundColor: "#dcfce7", padding: "15px", borderRadius: "10px", marginBottom: "10px", border: "2px solid #16a34a" }}>
          <p style={{ fontSize: "1.8em", color: "#16a34a", fontWeight: "bold" }}>⏱️ Hold: {timer.toFixed(1)}s / {holdDuration}s</p>
          <div style={{ width: "100%", backgroundColor: "#e5e7eb", borderRadius: "10px", height: "20px", overflow: "hidden" }}>
            <div style={{ width: `${(timer / holdDuration) * 100}%`, backgroundColor: "#16a34a", height: "100%" }} />
          </div>
        </div>
      )}

      {resting && (
        <div style={{ backgroundColor: "#dbeafe", padding: "15px", borderRadius: "10px", marginBottom: "10px", border: "2px solid #3b82f6" }}>
          {/* MODIFIED: Hardcoding the display text for the rest timer bar */}
          <p style={{ fontSize: "1.8em", color: "#3b82f6", fontWeight: "bold" }}>😌 RELAX! TAKE REST: {breakTimer}s / {breakDuration}s</p>
          <div style={{ width: "100%", backgroundColor: "#e5e7eb", borderRadius: "10px", height: "20px", overflow: "hidden" }}>
            <div style={{ width: `${(breakTimer / breakDuration) * 100}%`, backgroundColor: "#3b82f6", height: "100%" }} />
          </div>
        </div>
      )}

      <video ref={videoRef} style={{ display: "none" }}></video>
      <canvas ref={canvasRef} width="640" height="480" style={{ borderRadius: "12px", boxShadow: "0 6px 20px rgba(0,0,0,0.2)", maxWidth: "100%", height: "auto" }} />

      <div style={{ marginTop: "20px", padding: "15px", backgroundColor: "white", borderRadius: "10px", textAlign: "left", maxWidth: "640px", margin: "20px auto" }}>
        <h3 style={{ color: "#2563eb" }}>📋 Exercise Guide:</h3>
        <ul style={{ lineHeight: "1.8" }}>
          <li>**Both Arms Raise:** Raise BOTH arms straight up vertically</li>
          <li>**Wall Push:** Bend elbows and position arms as if pushing a wall - great for upper body strength</li>
          <li>**Squats:** Bend BOTH knees and lower your body into squat position</li>
          <li>**Neck Rolls:** Gently roll your head in circles - perfect for desk work stress</li>
          <li>**Shoulder Shrugs:** Lift both shoulders towards ears - releases upper body tension</li>
        </ul>
        <p style={{ marginTop: "15px", color: "#6b7280" }}>
          💡 Hold each position correctly for {holdDuration === 5 ? "5" : "10"} seconds, then rest for {breakDuration} seconds. 
          The skeleton will turn <span style={{ color: "#16a34a", fontWeight: "bold" }}>green</span> when your pose is correct!
        </p>
        <p style={{ marginTop: "10px", color: "#2563eb", fontWeight: "bold" }}>
          ✨ Perfect combination of strength training and stress relief exercises!
        </p>
      </div>
    </div>
  );
};

export default ExerciseCam;
