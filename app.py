from flask import Flask, render_template, jsonify
import os
import threading
import time
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import send_from_directory

# Define the path for the Excel file
excel_file_path = 'D:/ANAND Project/Haldex ABA Project/ABA photos latest/UI ABA Project/Result Excel File/processing_results.xlsx'

# Create the Excel file with the necessary columns if it doesn't exist
if not os.path.exists(excel_file_path):
    df = pd.DataFrame(columns=['Serial No', 'Image name', 'Bush', 'Nipple', 'O-ring','M4 Screw','Timestamp'])
    df.to_excel(excel_file_path, index=False)
    

# Load the pre-trained models
bush_model = load_model('bush_model.h5')
#marking_model = load_model('marking_model.h5')
nipple_model = load_model('nipple_model.h5')
oring_model = load_model('oring_model.h5')

m4_model = load_model('m4_model.h5')

# Coordinates for each component
bush_coordinates = {"x": 714.5384615384617, "y": 274.11538461538464, "width": 264.0, "height": 264.0}
#marking_coordinates = {"x": 332.53846153846166, "y": 1578.1153846153848, "width": 428.0, "height": 390.0}
nipple_coordinates = {"x": 356.53846153846166, "y": 1183.1153846153848, "width": 246.0, "height": 246.0}
oring_coordinates = {"x": 765.5384615384617, "y": 1859.6153846153848, "width": 618.0, "height": 615.0}

m4_coordinates = {"x": 690.0384615384617, "y": 1865.1153846153848, "width": 1195.0, "height": 872.0}

LMoS_coordinates = {"x": 205.6822742474917, "y": 1645.1153846153848, "width": 112.28762541806009, "height": 128.0}
LTS_coordinates = {"x": 501.1822742474917, "y": 1567.546822742475, "width": 131.2876254180601, "height": 130.86287625418026} 
LMS_coordinates = {"x": 372.82608695652175, "y": 1892.546822742475, "width": 130.0, "height": 126.86287625418026}
LBS_coordinates = {"x": 545.1822742474917, "y": 2202.1153846153848, "width": 140.7123745819399, "height": 132.0}
RTS_coordinates = {"x": 946.5384615384617, "y": 1509.1153846153848, "width": 128.0, "height": 138.0}
RMS_coordinates = {"x": 1166.5384615384617, "y": 1847.6153846153848, "width": 140.0, "height": 139.0}
RBS_coordinates = {"x": 968.0384615384617, "y": 2207.6153846153848, "width": 137.0, "height": 137.0}


app = Flask(__name__)

# Global variable to store the latest image path
dynamic_latest_image_path = ""
absolute_latest_image_path =""

###############################################################################################################################
#Home page Code 
# Define a route to serve the HTML file at the root URL
@app.route('/')
def index():
    return render_template('Trial.html')  # Ensure you have a 'Trial.html' file in a 'templates' folder

#################################################################################################################################

# Route to fetch the latest image from a specified folder
@app.route('/latest-image')
def latest_image():
    global dynamic_latest_image_path  # Declare the global variable
    global absolute_latest_image_path
    
    IMAGE_FOLDER = 'D:/ANAND Project/Haldex ABA Project/ABA photos latest/UI ABA PROJECT/static/images/'  # Update with your actual image folder path
    files = sorted(
        (f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))),
        key=lambda f: os.path.getmtime(os.path.join(IMAGE_FOLDER, f)),
        reverse=True
    )
    if files:
        dynamic_latest_image_path = f"/static/images/{files[0]}" #Dynamic path can not be used for the processing the image
        absolute_latest_image_path = os.path.join(IMAGE_FOLDER, files[0]) #Absolute path
        print("Latest Image Path updated in function:", dynamic_latest_image_path)  # Print the latest image path inside the function
        return jsonify({"imagePath": dynamic_latest_image_path})
    
    dynamic_latest_image_path = ""  # Set to empty if no images found
    print("No images found in the directory.")
    return jsonify({"imagePath": ""})

###############################################################################################################################

last_processed_image = ""


@app.route('/process-latest-image')
def process_latest_image():
    global absolute_latest_image_path, last_processed_image  # Access the global variable
            
     # Only proceed if there is a new image
    if absolute_latest_image_path and absolute_latest_image_path != last_processed_image:
        # Update the last processed image path
        last_processed_image = absolute_latest_image_path

        
        # Process the image using latest_image_path
        print("Processing image at path:", absolute_latest_image_path)
        
        output_folder = 'D:/ANAND Project/Haldex ABA Project/ABA photos latest/UI ABA PROJECT/Result/'
        img = cv2.imread(absolute_latest_image_path)
        image_name = os.path.basename(absolute_latest_image_path)
        
        # Prediction and annotation functions (from your code)
        def predict_and_draw(img, model, coordinates, label):
            x, y, width, height = int(coordinates['x']), int(coordinates['y']), int(coordinates['width']), int(coordinates['height'])
            roi = img[y:y + height, x:x + width]
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0  # Normalize

            prediction = model.predict(np.expand_dims(roi_resized, axis=0))
            confidence = np.max(prediction) * 100
            predicted_class = np.argmax(prediction, axis=1)[0]
            display_text = f'{label}: {"Okay" if predicted_class == 1 else "Not Okay"} ({confidence:.2f}%)'

            color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 10)
            cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
            
            return predicted_class
        
        def predict_screw(img, model, coordinates, label):
            x, y, width, height = int(coordinates['x']), int(coordinates['y']), int(coordinates['width']), int(coordinates['height'])
            roi = img[y:y + height, x:x + width]
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0  # Normalize

            prediction = model.predict(np.expand_dims(roi_resized, axis=0))
            confidence = np.max(prediction) * 100
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            return predicted_class
        
        def draw(img, coordinates, label):
            predicted_class = 1 if m4_result == "Okay" else "Not Okay"
            x, y, width, height = int(coordinates['x']), int(coordinates['y']), int(coordinates['width']), int(coordinates['height'])
            
            display_text = f'{label}: {"Okay" if predicted_class == 1 else "Not Okay"}'
            
            color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 10)
            cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
            #return m4_result
        
               
        # Apply the models to the image and get results for each component
        bush_result = "Okay" if predict_and_draw(img, bush_model, bush_coordinates, 'bush') == 1 else "Not Okay"
        #marking_result = "Okay" if predict_and_draw(img, marking_model, marking_coordinates, 'Marking') == 1 else "Not Okay"
        nipple_result = "Okay" if predict_and_draw(img, nipple_model, nipple_coordinates, 'Nipple') == 1 else "Not Okay"
        oring_result = "Okay" if predict_and_draw(img, oring_model, oring_coordinates, 'Oring') == 1 else "Not Okay"
        
        
        LMoS_result = "Okay" if predict_screw(img, m4_model, LMoS_coordinates, 'screw') == 1 else "Not Okay"
        LTS_result = "Okay" if predict_screw(img, m4_model, LTS_coordinates, 'screw') == 1 else "Not Okay"
        LMS_result = "Okay" if predict_screw(img, m4_model, LMS_coordinates, 'screw') == 1 else "Not Okay"
        LBS_result = "Okay" if predict_screw(img, m4_model, LBS_coordinates, 'screw') == 1 else "Not Okay"
        RTS_result = "Okay" if predict_screw(img, m4_model, RTS_coordinates, 'screw') == 1 else "Not Okay"
        RMS_result = "Okay" if predict_screw(img, m4_model, RMS_coordinates, 'screw') == 1 else "Not Okay"
        RBS_result = "Okay" if predict_screw(img, m4_model, RBS_coordinates, 'screw') == 1 else "Not Okay"
        
        m4_result = "Okay" if all(result == "Okay" for result in [LMoS_result, LTS_result, LMS_result, LBS_result, RTS_result, RMS_result, RBS_result]) else "Not Okay"
        
        #m4_result == "Okay"
        draw(img, m4_coordinates, 'screw')    
                
        # Save the processed image with the same name as the original image
        output_path = os.path.join(output_folder, os.path.basename(absolute_latest_image_path))
        cv2.imwrite(output_path, img)
        
        # Load the existing Excel file and append the new row
        df = pd.read_excel(excel_file_path)
        new_row = {
            'Serial No': len(df) + 1,
            'Image name': image_name,
            'Bush': bush_result,
            #'Marking': marking_result,
            'Nipple': nipple_result,
            'O-ring': oring_result,
            'M4 Screw' : m4_result,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format as desired
        }
        df = df._append(new_row, ignore_index=True)
        
        # Save the updated DataFrame back to the Excel file
        df.to_excel(excel_file_path, index=False)
                           
        print("Processed image at path:", absolute_latest_image_path)
        # Add any processing logic here
        # Determine overall result based on individual component results

        #overall_status = "Okay" if all(result == "Okay" for result in [bush_result, marking_result, nipple_result, oring_result, m4_result]) else "Not Okay"
        overall_status = "Okay" if all(result == "Okay" for result in [bush_result, nipple_result, oring_result, m4_result]) else "Not Okay"

        os.remove(absolute_latest_image_path)
        
        # Include the overall status in the response
        return jsonify({"message": f"Processed the latest image at path: {absolute_latest_image_path}", "overallStatus": overall_status})
    else:
        return "No latest image available to process."

#########################################################################################################################################
        
# Route to fetch the latest processed image from the result folder
@app.route('/latest-result-image')
def latest_result_image():
    result_folder = 'D:/ANAND Project/Haldex ABA Project/ABA photos latest/UI ABA PROJECT/Result/'  # Path to the result folder
    files = sorted(
        (f for f in os.listdir(result_folder) if os.path.isfile(os.path.join(result_folder, f))),
        key=lambda f: os.path.getmtime(os.path.join(result_folder, f)),
        reverse=True
    )
    if files:
        latest_result_image_path = f"/result/{files[0]}"
        print("Latest Result Image Path:", latest_result_image_path)
        return jsonify({"imagePath": latest_result_image_path})
    
    return jsonify({"imagePath": ""})

@app.route('/result/<path:filename>')
def serve_result_image(filename):
    result_directory = "D:/ANAND Project/Haldex ABA Project/ABA photos latest/UI ABA PROJECT/Result/"
    return send_from_directory(result_directory, filename)


if __name__ == '__main__':
    app.run(debug=False)
