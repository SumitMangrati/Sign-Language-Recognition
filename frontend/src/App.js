import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

const App = () => {
  const webcamRef = useRef(null);
  const [prediction, setPrediction] = useState("Waiting...");
  const [error, setError] = useState("");

  useEffect(() => {
    const interval = setInterval(captureAndSend, 1000); // Capture every second
    return () => clearInterval(interval);
  }, []);

  const captureAndSend = async () => {
    if (!webcamRef.current) {
      console.error("Webcam not found!");
      return;
    }

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) {
      setError("Failed to capture image.");
      return;
    }

    const blob = await fetch(imageSrc).then((res) => res.blob());
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(response.data.prediction);
      setError("");
    } catch (err) {
      console.error("Error sending frame:", err);
      setError("Error sending frame to server.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Sign Language Detection</h1>
      <Webcam ref={webcamRef} screenshotFormat="image/jpeg" width={640} height={480} videoConstraints={videoConstraints} />
      <h2>Prediction: {prediction}</h2>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default App;
