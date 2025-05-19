import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


categories = {
    'General_Sign': 20,  
    'Alphabet': 27      
}

dataset_size = int(input("Enter the number of images per class: "))

print("Available categories:")
for idx, category in enumerate(categories.keys(), start=1):
    print(f"{idx}. {category}")

selected_categories = input("Enter the numbers of categories you want to collect (comma separated): ")
selected_categories = [int(i.strip()) for i in selected_categories.split(',')]


for idx in selected_categories:
    category = list(categories.keys())[idx - 1]
    category_path = os.path.join(DATA_DIR, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    existing_classes = len([name for name in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, name))])
    for i in range(existing_classes, categories[category]):
        class_folder = os.path.join(category_path, str(i))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

cap = cv2.VideoCapture(0)


for idx in selected_categories:
    category = list(categories.keys())[idx - 1]
    print(f"Collecting data for {category}")
    
    for class_label in range(categories[category]):
 
        class_folder = os.path.join(DATA_DIR, category, str(class_label))
        if len(os.listdir(class_folder)) >= dataset_size:
            print(f"Class {class_label} in {category} already has enough data. Skipping...")
            continue

        print(f"Collecting data for class {class_label} in {category}")
     
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to access the camera.")
                break
            
            cv2.putText(frame, f'Ready? Press "S" to collect for class {class_label}!', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):  
                break


        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to access the camera.")
                break

            cv2.putText(frame, f'Capturing {counter + 1}/{dataset_size} for class {class_label}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            image_path = os.path.join(class_folder, f'{counter}.jpg')
            cv2.imwrite(image_path, frame)
            counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):  
                print("Capture stopped by user.")
                break

print("Data collection complete!")
cap.release()
cv2.destroyAllWindows()
