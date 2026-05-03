import cv2
import tensorflow as tf
import numpy as np

# ---------------- MODEL LOAD ----------------
interpreter = tf.lite.Interpreter(model_path="rice_classifier_final.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = [
    'basmati2000Z1',
    'basmatipakZ',
    'chinabZ1',
    'kissanbasmatiZ',
    'punjabZ1',
    'super2019Z1',
    'supergoldZ1'
]

# ---------------- PREDICTION FUNCTION ----------------
def predict_frame(frame):
    # FIXED preprocessing (IMPORTANT)
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    index = np.argmax(output)
    confidence = output[index]

    return classes[index], confidence

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Press 's' to capture & predict | 'z' to exit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error")
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------------- EXIT ----------------
    if key == ord('z'):
        print("🛑 Stopping program...")
        break

    # ---------------- CAPTURE & PREDICT ----------------
    if key == ord('s'):
        print("📸 Image captured!")

        label, conf = predict_frame(frame)

        print(f"🌾 Prediction: {label} | Confidence: {conf*100:.2f}%")

        # show result on frame
        result_frame = frame.copy()
        text = f"{label} ({conf*100:.2f}%)"

        cv2.putText(result_frame, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Prediction Result", result_frame)
        cv2.waitKey(2000)

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()