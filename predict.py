

import cv2
from fer import FER

# Initialize the FER detector with MTCNN for better accuracy
detector = FER(mtcnn=True)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Detect emotions in the frame
    result = detector.detect_emotions(frame)

    # Draw bounding boxes and emotion labels on the frame
    for face in result:
        (x, y, w, h) = face['box']
        emotions = face['emotions']
        
        # Get the top emotion and its score
        top_emotion = max(emotions, key=emotions.get)
        score = emotions[top_emotion]
        
        # Draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Label the top emotion
        label = f"{top_emotion}: {score:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()

