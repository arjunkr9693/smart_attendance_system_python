import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load known faces
known_face_encodings = []
known_face_names = ["arjun", "Elon_Musk"]
for person in known_face_names:
    image_path = f"Photos/{person}.png"
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)

# Create a copy of the known face names for expected students
expected_students = known_face_names.copy()

# Initialize variables for face locations and encodings
face_locations = []
face_encodings = []

# Get the current time
now = datetime.now()
current_date = now.strftime("%d-%m-%y")

# Open a CSV file for writing attendance
with open(f"{current_date}.csv", "w+", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Time"])

    # Start the main video capture loop
    while True:
        _, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Recognize faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            name = "Unknown"

            # Draw a rectangle around the face
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Display the name above the rectangle
                cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)

                # Mark attendance if the person is not already marked
                if name in expected_students:
                    cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    expected_students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    writer.writerow([name, current_time])
                else:
                    # Show 'Already Present' above the rectangle if attendance is already marked
                    cv2.putText(frame, "Already Present", (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("Attendance System", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
