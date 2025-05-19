from flask import Flask, render_template, Response, jsonify
import cv2
import logging
import mediapipe as mp
import numpy as np
import pickle
import time
from threading import Lock

app = Flask(__name__)

processing = False  
cap = None
is_alphabet_mode = False
current_model = None
labels_dict = None
thread_lock = Lock()
sign_buffer = []
last_detection_time = time.time()
reset_threshold = 3.0
current_sign_start = None  
last_added_character = None  
last_character_time = None  

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

def load_models():
    global current_model, labels_dict
    try:
       
        general_model_dict = pickle.load(open('./general_sign_model.p', 'rb'))
        general_model = general_model_dict['model']
        
        
        alphabet_model_dict = pickle.load(open('./alphabet_model.p', 'rb'))
        alphabet_model = alphabet_model_dict['model']
        
        
        current_model = general_model
        labels_dict =   {0: 'Hello', 1: 'Goodbye', 2: 'No', 3: 'Why?', 4: 'Thanks',
                        5: 'Food', 6: 'Drink', 7: 'You', 8: 'Please', 9: 'Yes', 10: 'No',
                        11: 'I Love You', 12: 'Girl', 13: 'Sorry', 14: 'Thanks', 15: 'See', 16: 'Please',
                        17: 'Yes', 18: 'Never', 19: 'YesHear'}
        
        return general_model, alphabet_model
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return None, None

def generate_frames():
    global cap, processing, sign_buffer, last_detection_time, current_sign_start
    global last_added_character, last_character_time
    
    while True:
        if not processing:
            break
            
        with thread_lock:
            if cap is None or not cap.isOpened():
                break
                
            success, frame = cap.read()
            if not success:
                break
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                current_time = time.time()
                
                if results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)
                        
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))
                    
                    if x_ and y_:
                        x1 = int(min(x_) * frame.shape[1]) - 10
                        y1 = int(min(y_) * frame.shape[0]) - 10
                        x2 = int(max(x_) * frame.shape[1]) - 10
                        y2 = int(max(y_) * frame.shape[0]) - 10
                        
                        if current_model is not None and len(data_aux) > 0:
                            prediction_prob = current_model.predict_proba([np.asarray(data_aux)])[0]
                            prediction_idx = np.argmax(prediction_prob)
                            accuracy = prediction_prob[prediction_idx] * 100
                            predicted_character = labels_dict.get(prediction_idx, "Unknown")
                            
                            # Handle fingerspelling mode
                            if is_alphabet_mode and accuracy > 60:
                                if current_sign_start is None:
                                    # Start timing for new sign
                                    current_sign_start = current_time
                                    last_detection_time = current_time
                                elif (current_time - current_sign_start >= 2.0):
                                    # After 3 seconds, add the character
                                    if (last_added_character != predicted_character or 
                                        last_character_time is None or 
                                        current_time - last_character_time > 1.0):
                                        sign_buffer.append(predicted_character)
                                        last_added_character = predicted_character
                                        last_character_time = current_time
                                        current_sign_start = None
                            else:
                                
                                current_sign_start = None
                            
                           
                            color = (0, min(255, accuracy * 2.55), 0)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                            text = f"{predicted_character} ({accuracy:.1f}%)"
                            
                       
                            if is_alphabet_mode and current_sign_start is not None:
                                hold_time = current_time - current_sign_start
                                if hold_time < 2.0:
                                    text += f" Hold: {hold_time:.1f}s"
                            
                            cv2.putText(frame, text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3,
                                        cv2.LINE_AA)
                else:
                   
                    if is_alphabet_mode:
                        sign_buffer = []
                    current_sign_start = None
                
               
                if is_alphabet_mode and sign_buffer:
                    current_word = ''.join(sign_buffer)
                    cv2.putText(frame, f"Current: {current_word}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue




@app.route('/reset_word', methods=['POST'])
def reset_word():
    global sign_buffer
    with thread_lock:
        sign_buffer = []
    return jsonify({"success": True})

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/start')
def start():
    global cap, processing
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Could not open camera", 500
    processing = True
    return render_template('start.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    processing = False
    return jsonify({"success": True}), 200

@app.route('/quit', methods=['POST'])
def quit_camera():
    global cap, processing
    processing = False
    if cap and cap.isOpened():
        cap.release()
        cap = None
    return render_template('main.html') 

@app.route('/toggle_fingerspelling', methods=['POST'])
def toggle_fingerspelling():
    global current_model, labels_dict, is_alphabet_mode
    with thread_lock:
        is_alphabet_mode = not is_alphabet_mode
        logging.debug(f"Switched to {'Alphabet' if is_alphabet_mode else 'General'} mode")
        if is_alphabet_mode:
            current_model = alphabet_model
            labels_dict = { 0: 'A', 1: 'B', 2: 'K', 3: 'L', 4: 'M', 5: 'N', 6: 'O', 7: 'P', 8: 'R', 9: 'S', 
                            10: 'T', 11: 'C', 12: 'U', 13: 'V', 14: 'W', 15: 'X', 16: 'Y', 17: 'Q', 18: 'D', 
                            19: 'E', 20: 'F', 21: 'G', 22: 'H', 23: 'I', 24: 'J', 25: 'Z', 26: ' ' }
        else:
            current_model = general_model
            labels_dict = {0: 'Hello', 1: 'Goodbye', 2: 'Where?', 3: 'Why?', 4: 'Girl',
                           5: 'Food', 6: 'Drink', 7: 'You', 8: 'Please', 9: 'Yes', 10: 'No',
                           11: 'I Love You', 12: 'Like', 13: 'Sorry', 14: 'Thanks', 15: 'See', 16: 'Please',
                           17: 'Yes', 18: 'Boy', 19: 'Hear'}
            
    return jsonify({"fingerspelling_mode": is_alphabet_mode})

if __name__ == '__main__':

    general_model, alphabet_model = load_models()
    if general_model is None or alphabet_model is None:
        print("Error: Could not load models. Please check model files exist.")
    else:
        app.run(debug=True, threaded=True)