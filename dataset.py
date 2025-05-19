import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


GENERAL_SIGN_DIR = './data/General_Sign'
ALPHABET_DIR = './data/Alphabet'

def collect_data_from_directory(data_dir):
    data = []
    labels = []
    for dir_ in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, dir_)):
            continue  
        
        for img_path in os.listdir(os.path.join(data_dir, dir_)):
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  
            
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(data_dir, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
               
                hand_landmarks = results.multi_hand_landmarks[0]
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

                data.append(data_aux)
                labels.append(dir_)  

 
    assert len(data) == len(labels), "Mismatch between collected data and labels."
    return data, labels




choice = input("Enter the data to collect ('alphabet', 'general' or 'both'): ").strip().lower()

if choice == 'both':
    alphabet_data, alphabet_labels = collect_data_from_directory(ALPHABET_DIR)
    with open('alphabet_data.pickle', 'wb') as f:
        pickle.dump({'data': alphabet_data, 'labels': alphabet_labels}, f)
    
    general_sign_data, general_sign_labels = collect_data_from_directory(GENERAL_SIGN_DIR)
    with open('general_sign_data.pickle', 'wb') as f:
        pickle.dump({'data': general_sign_data, 'labels': general_sign_labels}, f)
    
    print("saved as alphabet_data.pickle and general_sign_data.pickle")

elif choice == 'alphabet':
    alphabet_data, alphabet_labels = collect_data_from_directory(ALPHABET_DIR)
    with open('alphabet_data.pickle', 'wb') as f:
        pickle.dump({'data': alphabet_data, 'labels': alphabet_labels}, f)
    
    print("Alphabet data collection complete and saved as 'alphabet_data.pickle'.")
elif choice == 'general':
    general_sign_data, general_sign_labels = collect_data_from_directory(GENERAL_SIGN_DIR)
    with open('general_sign_data.pickle', 'wb') as f:
        pickle.dump({'data': general_sign_data, 'labels': general_sign_labels}, f)
    
    print("General Sign data collection complete and saved as 'general_sign_data.pickle'.")
else:
    print("Invalid choice! Please enter 'alphabet' or 'general'.")
