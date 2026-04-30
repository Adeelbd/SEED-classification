import cv2
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="rice_classifier_final.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#
classes = [ 'basmati2000Z1', 
    'basmatipakZ', 
    'chinabZ1', 
    'kissanbasmatiZ', 
    'punjabZ1', 
    'super2019Z1', 
    'supergoldZ1']


def predict_frame(frame):
    img = cv2.resize(frame, (224, 224))   # ⚠️ Change if your model uses different size
    img = img.astype(np.float32)

    

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = classes[np.argmax(output)]

    return predicted_class


def live_stream_camera(width=1920, height=1080):

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"Camera running at {width}x{height}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        prediction = predict_frame(frame)

        cv2.putText(frame, f"Seed: {prediction}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Live Seed Detection", frame)

        print("Detected:", prediction)

      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


live_stream_camera(1920, 1080)