import cv2
import mediapipe as mp
import tensorflow as tf
from keras_preprocessing.image import img_to_array
import numpy as np


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model("emotionmodel.h5")


with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height,image_width,_ = image.shape


    if results.detections:
      for detection in results.detections:
          bboxC = detection.location_data.relative_bounding_box
          x, y, w, h = int(bboxC.xmin * image_width), int(bboxC.ymin * image_height), int(
              bboxC.width * image_width), int(bboxC.height * image_height)

          # Crop the detected face
          face = image[y:y + h, x:x + w]
          face = cv2.resize(face, (224, 224))

          img_array = img_to_array(face)
          img_array = np.expand_dims(img_array, axis=0)
          img_array = img_array / 255.0
          predictions = model.predict(img_array)
          predicted_class = np.argmax(predictions, axis=1)[0]
          class_labels = ['angry', 'happy', 'sad']
          result = class_labels[predicted_class]


          image = cv2.flip(image, 1)
          cv2.putText(image, result, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
          # Display the cropped face
          cv2.imshow('Detected Face', cv2.flip(face,1))
          mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()