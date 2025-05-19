import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter

def preprocess_landmarks(data_dict):
    processed_data = []
    labels = []
    
    for sample, label in zip(data_dict['data'], data_dict['labels']):
       
        if isinstance(sample, list) and len(sample) == 42:  
            processed_data.append(sample)
            labels.append(label)
    
   
    assert len(processed_data) == len(labels), "Mismatch between data and labels."

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    

    with open('label_encoder.p', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return np.array(processed_data), encoded_labels

def train_model(data, labels, model_name):
   
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )
    

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    

    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    print(f"{model_name} Model Accuracy: {score * 100:.2f}%")
    
  
    with open('label_encoder.p', 'rb') as f:
        label_encoder = pickle.load(f)
    decoded_predictions = label_encoder.inverse_transform(y_predict)
    decoded_actual = label_encoder.inverse_transform(y_test)
    
  
    with open(f'{model_name.lower().replace(" ", "_")}_model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    
    print(f"{model_name} model saved successfully!")


choice = input("Enter the model to train ('alphabet', 'general' or 'both'): ").strip().lower()

if choice == 'both':
 
    general_sign_data_dict = pickle.load(open('./general_sign_data.pickle', 'rb'))
    general_data, general_labels = preprocess_landmarks(general_sign_data_dict)
    train_model(general_data, general_labels, "General Sign")
    
    
    alphabet_data_dict = pickle.load(open('./alphabet_data.pickle', 'rb'))
    alphabet_data, alphabet_labels = preprocess_landmarks(alphabet_data_dict)
    train_model(alphabet_data, alphabet_labels, "Alphabet")

elif choice == 'alphabet':
    alphabet_data_dict = pickle.load(open('./alphabet_data.pickle', 'rb'))
    alphabet_data, alphabet_labels = preprocess_landmarks(alphabet_data_dict)
    train_model(alphabet_data, alphabet_labels, "Alphabet")

elif choice == 'general':
    general_sign_data_dict = pickle.load(open('./general_sign_data.pickle', 'rb'))
    general_data, general_labels = preprocess_landmarks(general_sign_data_dict)
    train_model(general_data, general_labels, "General Sign")
else:
    print("Invalid choice! Please enter 'alphabet' or 'general'.")
