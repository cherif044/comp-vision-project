import matplotlib

# Set Windows-compatible backend
matplotlib.use('TkAgg')  # Ensure TkAgg for Windows compatibility
import matplotlib.pyplot as plt
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os
import time
import shutil

# Create a temporary directory for storing face images
temp_dir = "temp_faces"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Initialize matplotlib for visualizations
plt.ion()  # Interactive mode for real-time updates

def preprocess_face(img, face_idx, show_visualizations=True):
    print(f"Preprocessing Face {face_idx}...")
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if show_visualizations:
        plt.figure(figsize=(5,5))
        plt.imshow(img_rgb)
        plt.title(f'Original Image (Face {face_idx})')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        print(f"Displayed original image for Face {face_idx}")

    # Face detection (already done, but crop face for preprocessing)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print(f"No faces detected in preprocessing for Face {face_idx}.")
        return None, None

    # Take the first detected face (since this is a cropped face region)
    (x, y, w, h) = faces[0]
    face = img_rgb[y:y+h, x:x+w]

    if show_visualizations:
        plt.figure(figsize=(5,5))
        plt.imshow(face)
        plt.title(f"Detected & Cropped Face (Face {face_idx})")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        print(f"Displayed cropped face for Face {face_idx}")

    # Apply median filter
    median_filtered = cv2.medianBlur(face, 5)

    # Apply bilateral filter
    bilateral_filtered = cv2.bilateralFilter(median_filtered, d=9, sigmaColor=75, sigmaSpace=75)

    if show_visualizations:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(face); axes[0].set_title(f"Before Denoising (Face {face_idx})"); axes[0].axis('off')
        axes[1].imshow(bilateral_filtered); axes[1].set_title(f"After Median + Bilateral (Face {face_idx})"); axes[1].axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        print(f"Displayed denoising steps for Face {face_idx}")

    # Enhancement: CLAHE and sharpening
    lab = cv2.cvtColor(bilateral_filtered, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    enhanced_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    kernel_sharpening = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_clahe, -1, kernel_sharpening)

    if show_visualizations:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(bilateral_filtered); axes[0].set_title(f"After Median + Bilateral (Face {face_idx})"); axes[0].axis('off')
        axes[1].imshow(enhanced_clahe); axes[1].set_title(f"After CLAHE (Face {face_idx})"); axes[1].axis('off')
        axes[2].imshow(sharpened); axes[2].set_title(f"After Sharpening (Face {face_idx})"); axes[2].axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        print(f"Displayed CLAHE and sharpening for Face {face_idx}")

    # Histogram Equalization
    ycrcb = cv2.cvtColor(sharpened, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    enhanced_final = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2RGB)

    if show_visualizations:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sharpened); axes[0].set_title(f"Before Equalization (Face {face_idx})"); axes[0].axis('off')
        axes[1].imshow(enhanced_final); axes[1].set_title(f"After Equalization (Face {face_idx})"); axes[1].axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        print(f"Displayed histogram equalization for Face {face_idx}")

    # Custom preprocessing pipeline
    if show_visualizations:
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    else:
        fig, axes = None, [None] * 5

    step0 = enhanced_final
    if show_visualizations:
        axes[0].imshow(step0)
        axes[0].set_title(f"0. Input (Equalized & Sharpened) (Face {face_idx})")
        axes[0].axis('off')

    # Resize to model's expected input
    resized = cv2.resize(enhanced_final, (227, 227))

    if show_visualizations:
        plt.figure(figsize=(5,5))
        plt.imshow(resized)
        plt.title(f"Preprocessed Face (227x227) (Face {face_idx})")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.001)
        print(f"Displayed final preprocessed face for Face {face_idx}")

    # Return original face and enhanced face for DeepFace
    original_face = cv2.resize(face, (227, 227))
    return original_face, resized

def get_pred_label(result):
    if not result or not result[0]:
        return "No prediction"
    gender_probs = result[0]['gender']
    predicted_gender = max(gender_probs, key=gender_probs.get)
    predicted_age = result[0]['age']
    predicted_emotion = result[0]['dominant_emotion']
    predicted_race = result[0]['dominant_race']
    return f"{predicted_gender}, {predicted_age:.0f} yrs\n{predicted_emotion}, {predicted_race}"

def draw_label(image, label):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, line in enumerate(label.split('\n')):
        cv2.putText(img_bgr, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# User choice: webcam or upload
print("Choose input method:")
print("1. Webcam (track faces, press 's' to capture)")
print("2. Upload a single picture")
choice = input("Enter 1 or 2: ")

if choice == "2":
    # File upload mode
    try:
        from google.colab import files
        uploaded = files.upload()
        img_path = list(uploaded.keys())[0]
        img = cv2.imread(img_path)
    except:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename(title="Select an image")
        img = cv2.imread(img_path)

    if img is None:
        print("Error: Could not load the image.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        exit()

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade classifier.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        exit()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
    print("Displayed original image for uploaded file")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No faces detected.")
        plt.close('all')
        shutil.rmtree(temp_dir, ignore_errors=True)
        exit()

    results = []
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        original_face, enhanced_face = preprocess_face(face, i+1)
        if original_face is None or enhanced_face is None:
            continue

        original_path = os.path.join(temp_dir, f"original_face_{i}.jpg")
        enhanced_path = os.path.join(temp_dir, f"enhanced_face_{i}.jpg")
        Image.fromarray(original_face).save(original_path)
        Image.fromarray(enhanced_face).save(enhanced_path)

        actions = ['age', 'gender', 'emotion', 'race']
        try:
            result_original = DeepFace.analyze(img_path=original_path, actions=actions, enforce_detection=False)
            result_enhanced = DeepFace.analyze(img_path=enhanced_path, actions=actions, enforce_detection=False)
            results.append((original_face, enhanced_face, result_original, result_enhanced))
        except Exception as e:
            print(f"DeepFace error for Face {i+1}: {e}")
        finally:
            os.remove(original_path)
            os.remove(enhanced_path)

    n_faces = len(results)
    if n_faces > 0:
        fig, axes = plt.subplots(n_faces, 2, figsize=(12, 6 * n_faces))
        if n_faces == 1:
            axes = [axes]
        for i, (original_face, enhanced_face, result_original, result_enhanced) in enumerate(results):
            label_original = get_pred_label(result_original)
            label_enhanced = get_pred_label(result_enhanced)
            original_labeled = draw_label(original_face, label_original)
            enhanced_labeled = draw_label(enhanced_face, label_enhanced)

            axes[i][0].imshow(original_labeled)
            axes[i][0].set_title(f"Original Face {i+1} Prediction")
            axes[i][0].axis('off')
            axes[i][1].imshow(enhanced_labeled)
            axes[i][1].set_title(f"Enhanced Face {i+1} Prediction")
            axes[i][1].axis('off')

            print(f"Face {i+1} - Original Image Prediction:\n", label_original)
            print(f"Face {i+1} - Enhanced Image Prediction:\n", label_enhanced)
        plt.tight_layout()
        plt.show(block=True)  # Block to keep final plots open
        print("Displayed final prediction plots for uploaded file")
    else:
        print("No valid faces processed.")

    plt.close('all')
    shutil.rmtree(temp_dir, ignore_errors=True)
    exit()

# Webcam mode
def open_webcam():
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam opened successfully on index {index}.")
            return cap, index
    print("Error: Could not open webcam on indices 0, 1, or 2. Check permissions or device.")
    shutil.rmtree(temp_dir, ignore_errors=True)
    exit()

cap, webcam_index = open_webcam()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Could not load Haar Cascade classifier.")
    shutil.rmtree(temp_dir, ignore_errors=True)
    exit()

stored_faces = []
capture_triggered = False
error_message = ""
last_capture_time = time.time()  # Track time for 15-second interval
last_predictions = {}  # Cache predictions for each face

print("Starting webcam... Press 's' to capture detected faces manually, 'q' to quit.")

def is_same_face(face1, face2, threshold=50):
    """Check if two faces are the same based on coordinates."""
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    return (abs(x1 - x2) < threshold and
            abs(y1 - y2) < threshold and
            abs(w1 - w2) < threshold and
            abs(h1 - h2) < threshold)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Check if 15 seconds have passed and faces are detected
    current_time = time.time()
    auto_capture = (current_time - last_capture_time >= 15.0) and len(faces) > 0

    # Process faces for display and capture
    if len(faces) > 0:
        frame_predictions = {}
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

            face_key = f"{x}_{y}_{w}_{h}"
            # Check if this face matches a previously detected face
            matched_key = None
            for prev_key in last_predictions:
                prev_x, prev_y, prev_w, prev_h = map(int, prev_key.split('_'))
                if is_same_face((x, y, w, h), (prev_x, prev_y, prev_w, prev_h)):
                    matched_key = prev_key
                    break

            if matched_key and not auto_capture:
                # Reuse cached predictions for the same face
                label_original, label_enhanced = last_predictions[matched_key]
            else:
                # Process new or updated face
                label_original = "Processing..."
                label_enhanced = "Processing..."
                if auto_capture:
                    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    original_face, enhanced_face = preprocess_face(face_rgb, i+1, show_visualizations=False)
                    if original_face is None or enhanced_face is None:
                        label_original = "Processing Error"
                        label_enhanced = "Processing Error"
                    else:
                        original_path = os.path.join(temp_dir, f"original_face_{i}.jpg")
                        enhanced_path = os.path.join(temp_dir, f"enhanced_face_{i}.jpg")
                        Image.fromarray(original_face).save(original_path)
                        Image.fromarray(enhanced_face).save(enhanced_path)

                        actions = ['age', 'gender', 'emotion', 'race']
                        try:
                            result_original = DeepFace.analyze(img_path=original_path, actions=actions, enforce_detection=False)
                            result_enhanced = DeepFace.analyze(img_path=enhanced_path, actions=actions, enforce_detection=False)
                            label_original = get_pred_label(result_original)
                            label_enhanced = get_pred_label(result_enhanced)
                            stored_faces.append((original_face, enhanced_face, result_original, result_enhanced))
                        except Exception as e:
                            print(f"DeepFace error for Face {i+1}: {e}")
                            label_original = "Error in Prediction"
                            label_enhanced = "Error in Prediction"
                        finally:
                            try:
                                os.remove(original_path)
                                os.remove(enhanced_path)
                            except OSError as e:
                                print(f"Error deleting temporary file: {e}")

            frame_predictions[face_key] = (label_original, label_enhanced)

            # Display predictions
            y_offset = y - 100 if y - 100 > 0 else 20
            for j, line in enumerate(label_original.split('\n')):
                cv2.putText(frame, f"O: {line}", (x, y_offset + j*20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2, cv2.LINE_AA)
            for j, line in enumerate(label_enhanced.split('\n')):
                cv2.putText(frame, f"E: {line}", (x, y_offset + (j+2)*20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 0), 2, cv2.LINE_AA)

        if auto_capture:
            last_predictions = frame_predictions
            last_capture_time = current_time
            print("Processed and captured faces after 15 seconds.")

        error_message = ""
    else:
        error_message = "No Face Detected"
        cv2.putText(frame, error_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Face Tracking - Press s to Capture, q to Quit', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if len(faces) > 0:
            capture_triggered = True
            for i, (x, y, w, h) in enumerate(faces):
                face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                original_face, enhanced_face = preprocess_face(face_rgb, i+1, show_visualizations=False)
                if original_face is None or enhanced_face is None:
                    continue

                original_path = os.path.join(temp_dir, f"original_face_{i}.jpg")
                enhanced_path = os.path.join(temp_dir, f"enhanced_face_{i}.jpg")
                Image.fromarray(original_face).save(original_path)
                Image.fromarray(enhanced_face).save(enhanced_path)

                actions = ['age', 'gender', 'emotion', 'race']
                try:
                    result_original = DeepFace.analyze(img_path=original_path, actions=actions, enforce_detection=False)
                    result_enhanced = DeepFace.analyze(img_path=enhanced_path, actions=actions, enforce_detection=False)
                    label_original = get_pred_label(result_original)
                    label_enhanced = get_pred_label(result_enhanced)
                    face_key = f"{x}_{y}_{w}_{h}"
                    last_predictions[face_key] = (label_original, label_enhanced)
                    stored_faces.append((original_face, enhanced_face, result_original, result_enhanced))
                except Exception as e:
                    print(f"DeepFace error for Face {i+1}: {e}")
                finally:
                    try:
                        os.remove(original_path)
                        os.remove(enhanced_path)
                    except OSError as e:
                        print(f"Error deleting temporary file: {e}")
            print("Manual capture triggered.")
        else:
            error_message = "Error: No face detected to capture"
            print(error_message)
            cv2.putText(frame, error_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Face Tracking - Press s to Capture, q to Quit', frame)
            cv2.waitKey(1000)

# Release webcam
cap.release()
cv2.destroyAllWindows()

# Display results for captured faces
if stored_faces:
    results = []
    for i, (original_face, enhanced_face, result_original, result_enhanced) in enumerate(stored_faces):
        results.append((original_face, enhanced_face, result_original, result_enhanced))

    # Display results
    n_faces = len(results)
    if n_faces > 0:
        fig, axes = plt.subplots(n_faces, 2, figsize=(12, 6 * n_faces))
        if n_faces == 1:
            axes = [axes]
        for i, (original_face, enhanced_face, result_original, result_enhanced) in enumerate(results):
            label_original = get_pred_label(result_original)
            label_enhanced = get_pred_label(result_enhanced)
            original_labeled = draw_label(original_face, label_original)
            enhanced_labeled = draw_label(enhanced_face, label_enhanced)

            axes[i][0].imshow(original_labeled)
            axes[i][0].set_title(f"Original Face {i+1} Prediction")
            axes[i][0].axis('off')
            axes[i][1].imshow(enhanced_labeled)
            axes[i][1].set_title(f"Enhanced Face {i+1} Prediction")
            axes[i][1].axis('off')

            print(f"Face {i+1} - Original Image Prediction:\n", label_original)
            print(f"Face {i+1} - Enhanced Image Prediction:\n", label_enhanced)
        plt.tight_layout()
        plt.show(block=True)  # Block to keep final plots open
        print("Displayed final prediction plots for captured faces")
    else:
        print("No valid faces processed.")
else:
    print("No faces captured for processing.")

plt.close('all')
shutil.rmtree(temp_dir, ignore_errors=True)
