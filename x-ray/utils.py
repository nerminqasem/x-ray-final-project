from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import pandas as pd
import os
import cv2
import shutil
from flask import request
import torch
import torchvision.transforms as T
from collections import Counter

# Read chest X-ray from a DICOM file
def read_xray(path, voi_lut=True, fix_monochrome=True):
    """
    Reads a chest X-ray from a DICOM file.
    """
    dicom = pydicom.read_file(path)

    # Apply VOI LUT
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # MONOCHROME1 correction
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # Scale data to [0, 255]
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data

# Resize image
def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    """
    Resizes the image to the specified size.
    """
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im

# Draw advanced bounding box (with disease type)
def draw_bbox_with_disease(image, xmin, ymin, xmax, ymax, text=None, line_thickness=20, disease_type=None, confidence=None):
    """
    Draws a bounding box on the image and indicates the disease type.
    """
    w = xmax - xmin
    h = ymax - ymin

    # Determine color based on disease type
    if disease_type == "Pneumonia":
        bbox_color = (255, 0, 0)  # Red
    elif disease_type == "Consolidation":
        bbox_color = (0, 255, 0)  # Green
    elif disease_type == "Atelectasis":
        bbox_color = (0, 0, 255)  # Blue
    elif disease_type == "Edema":
        bbox_color = (255, 255, 0)  # Yellow
    elif disease_type == "Nodule":
        bbox_color = (255, 0, 255)  # Magenta
    else:
        bbox_color = (255, 255, 255)  # White (default)

    # Draw the box
    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    bbox_thickness = line_thickness

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)

    # Add the confidence score
    confidence_text = f"{int(confidence*100)}%" if confidence else ""
    
    # Add a background if there is text
    if text or confidence_text:
        # Background rectangle
        cv2.rectangle(image, (xmin, ymin - 70), (xmin + w, ymin), (0, 0, 0), -1)

        # Add the text
        text_color = (255, 255, 255)  # White
        font = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 2
        thickness = 4

        # Opacity number and confidence
        display_text = f"{text} {confidence_text}" if text else confidence_text
        origin = (xmin + 10, ymin - 30)
        image = cv2.putText(image, display_text, origin, font,
                            fontScale, text_color, thickness, cv2.LINE_AA)
    
    # Also add the disease type
    if disease_type:
        cv2.putText(image, disease_type, (xmin + 10, ymin - 60), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, bbox_color, 3, cv2.LINE_AA)

    return image

# Draw a simple bounding box (for backward compatibility)
def draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=2):
    """
    Draws a simple bounding box on the image (for backward compatibility).
    """
    color = (255, 0, 0)  # Blue
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
    return image

# Pad the image to make it square
def pad_image_to_square(image):
    """
    Pads the image on the sides to make it square.
    """
    shape_tuple = image.shape
    height = image.shape[0]
    width = image.shape[1]

    def pad_image_channel(image_channel, height, width):
        pad_amt = abs(height - width)

        if height == width:
            pad_channel = image_channel
        elif height > width:  # Pad to the right
            pad_channel = np.pad(image_channel, [(0, 0), (0, pad_amt)], mode='constant')
        else:  # Pad at the bottom
            pad_channel = np.pad(image_channel, [(0, pad_amt), (0, 0)], mode='constant')

        return pad_channel

    if len(shape_tuple) == 2:  # Grayscale
        padded_image = pad_image_channel(image, height, width)
    elif len(shape_tuple) == 3:  # Multi-channel
        num_channels = image.shape[2]

        for j in range(num_channels):
            image_channel = image[:, :, j]
            padded_channel = pad_image_channel(image_channel, height, width)

            if j == 0:
                padded_image = padded_channel
            else:
                padded_image = np.dstack((padded_image, padded_channel))

    return padded_image

# Process the list of images
def process_images(image_file_list):
    """
    Processes the list of images, pads them to square, and saves them.
    """
    if not os.path.isdir('static/proc_images_dir'):
        proc_images_dir = os.path.join('static', 'proc_images_dir')
        os.mkdir(proc_images_dir)

    for i, image_fname in enumerate(image_file_list):
        ext = image_fname.split('.')[-1]
        path = 'uploads/' + image_fname

        if ext in ['dcm', 'dicom']:
            image = read_xray(path)
            image = pad_image_to_square(image)
            new_fname = image_fname.replace(ext, 'png')
            new_path = 'static/proc_images_dir/' + new_fname
            cv2.imwrite(new_path, image)
        else:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            image = pad_image_to_square(image)
            new_path = 'static/proc_images_dir/' + image_fname
            cv2.imwrite(new_path, image)

# Process FasterRCNN predictions
def process_fasterrcnn_preds(pred, conf_threshold):
    """
    Processes the predictions coming from the FasterRCNN model.
    """
    pred_scores_list = list(pred[0]['scores'].detach().cpu().numpy())
    pred_labels_list = list(pred[0]['labels'].detach().cpu().numpy())
    pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    pred_boxes_list = [list(item) for item in pred_boxes]

    fin_scores_list = []
    fin_labels_list = []
    fin_boxes_list = []

    if pred_scores_list:
        for i, item in enumerate(pred_scores_list):
            if item > conf_threshold:
                fin_scores_list.append(item)
                fin_labels_list.append(pred_labels_list[i])
                fin_boxes_list.append(pred_boxes_list[i])

    pred_dict = {
        'pred_scores': fin_scores_list,
        'pred_labels': fin_labels_list,
        'pred_boxes': fin_boxes_list
    }

    return pred_dict

# Create a prediction dataframe
def create_pred_dataframe(pred_dict, fname, image_height, image_width):
    """
    Creates a pandas DataFrame from the prediction results.
    """
    num_labels = len(pred_dict['pred_labels'])

    if num_labels == 0:
        empty_dict = {
            'xmin': [0],
            'ymin': [0],
            'xmax': [0],
            'ymax': [0],
            'pred_score': [0],
            'pred_labels': [2],
            'fname': fname,
            'orig_image_height': image_height,
            'orig_image_width': image_width
        }
        df = pd.DataFrame(empty_dict)
    else:
        df1 = pd.DataFrame(pred_dict)
        df1['fname'] = fname

        boxes_np = np.array(list(df1['pred_boxes']))
        cols = ['xmin', 'ymin', 'xmax', 'ymax']
        df2 = pd.DataFrame(boxes_np, columns=cols)

        df = pd.concat([df1, df2], axis=1)
        df = df.drop('pred_boxes', axis=1)
        df['orig_image_height'] = image_height
        df['orig_image_width'] = image_width

    # Reset coordinates for label 2 (normal)
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    for i in range(0, len(df)):
        pred_label = df.loc[i, 'pred_labels']
        xmin = df.loc[i, 'xmin']
        ymin = df.loc[i, 'ymin']
        xmax = df.loc[i, 'xmax']
        ymax = df.loc[i, 'ymax']

        if pred_label == 2:
            xmin_list.append(0)
            ymin_list.append(0)
            xmax_list.append(0)
            ymax_list.append(0)
        else:
            xmin_list.append(xmin)
            ymin_list.append(ymin)
            xmax_list.append(xmax)
            ymax_list.append(ymax)

    df['xmin'] = xmin_list
    df['ymin'] = ymin_list
    df['xmax'] = xmax_list
    df['ymax'] = ymax_list

    return df

# Determine the disease type
def determine_disease_type(pred_labels):
    """
    Determines the disease type based on the predicted labels.
    """
    # Labels:
    # 1: Opacity (general), 2: Normal, 3: Lung, 4: Pneumonia,
    # 5: Consolidation, 6: Atelectasis, 7: Edema, 8: Nodule
    
    disease_mapping = {
        1: "Opacity",
        4: "Pneumonia",
        5: "Consolidation",
        6: "Atelectasis",
        7: "Edema",
        8: "Nodule"
    }
    
    detected_diseases = []
    for label in pred_labels:
        if label in disease_mapping:
            detected_diseases.append(disease_mapping[label])
    
    if not detected_diseases:
        return "Normal"
    
    # Return the most frequent disease
    disease_counts = Counter(detected_diseases)
    return disease_counts.most_common(1)[0][0]

# Determine the location (a simple estimate)
def determine_location(xmin, ymin, xmax, ymax, image_width, image_height):
    """
    Roughly determines the location of the bounding box.
    """
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    # Horizontal position
    if center_x < image_width / 3:
        horizontal = "left"
    elif center_x > 2 * image_width / 3:
        horizontal = "right"
    else:
        horizontal = "center"
    
    # Vertical position
    if center_y < image_height / 3:
        vertical = "upper"
    elif center_y > 2 * image_height / 3:
        vertical = "lower"
    else:
        vertical = "middle"
    
    # Lobe estimate
    if vertical == "upper":
        lob = "upper lobe"
    elif vertical == "lower":
        lob = "lower lobe"
    else:
        lob = "middle lobe"
    
    return f"{horizontal} {lob}"

# Run prediction on all images
def predict_on_all_images(model_list, device, image_list, threshold):
    """
    Runs prediction on all images and returns the results.
    """
    # Create the pred_images_dir folder
    if not os.path.isdir('static/pred_images_dir'):
        pred_images_dir = 'static/pred_images_dir'
        os.mkdir(pred_images_dir)

    print('Prediction starting...')
    print(os.listdir('static/proc_images_dir'))

    model_path_0 = f"TRAINED_MODEL_FOLDER/{model_list[0]}"

    # Load the model
    model = torch.load(model_path_0, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    model.to(device)

    num_preds_dict = {}
    disease_types_dict = {}
    detailed_results = {}  # Dictionary for detailed results

    for i, fname in enumerate(image_list):
        print(f'Predicting image {i+1}/{len(image_list)}...')

        orig_fname = fname
        ext = fname.split('.')[1]
        dicom_ext_list = ['dcm', 'dicom']

        if ext in dicom_ext_list:
            fname = fname.replace(ext, 'png')

        path = 'static/proc_images_dir/' + fname

        # Load the image
        image = Image.open(path).convert("L")
        image1 = np.array(image)
        image_height = image1.shape[0]
        image_width = image1.shape[1]

        # Convert to tensor
        my_transform = T.Compose([T.ToTensor()])
        image_tensor = my_transform(image)
        image_tensor = image_tensor.to(device)

        # Run prediction
        pred = model([image_tensor])
        pred_dict = process_fasterrcnn_preds(pred, threshold)

        # Create a DataFrame
        df1 = create_pred_dataframe(pred_dict, fname, image_height, image_width)

        if i == 0:
            df_fin = df1
        else:
            df_fin = pd.concat([df_fin, df1], axis=0)

        # Draw the bounding boxes
        path = 'static/proc_images_dir/' + fname
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        pred_boxes_list = pred_dict['pred_boxes']
        pred_labels_list = pred_dict['pred_labels']
        pred_scores_list = pred_dict['pred_scores']

        ignore_labels_list = [2, 3]
        lungs_detected = 'no'
        new_label_list = []
        new_bbox_list = []
        new_scores_list = []

        for j in range(0, len(pred_labels_list)):
            label = pred_labels_list[j]
            bbox = pred_boxes_list[j]
            score = pred_scores_list[j]

            if label == 3:
                lungs_detected = 'yes'

            if label not in ignore_labels_list:
                new_label_list.append(label)
                new_bbox_list.append(bbox)
                new_scores_list.append(score)

        num_preds = len(new_label_list)
        num_preds_dict[orig_fname] = (num_preds, lungs_detected)
        
        # Determine the disease type
        disease_type = determine_disease_type(new_label_list)
        disease_types_dict[orig_fname] = disease_type
        
        # Prepare the detailed results
        locations = []
        for j in range(0, len(new_bbox_list)):
            coords_list = new_bbox_list[j]
            xmin, ymin, xmax, ymax = coords_list
            location = determine_location(xmin, ymin, xmax, ymax, image_width, image_height)
            locations.append(location)
        
        detailed_results[orig_fname] = {
            'num_opacities': num_preds,
            'disease_type': disease_type,
            'confidence_scores': new_scores_list,
            'locations': locations,
            'labels': new_label_list
        }

        for j in range(0, len(new_bbox_list)):
            coords_list = new_bbox_list[j]
            xmin = int(coords_list[0])
            ymin = int(coords_list[1])
            xmax = int(coords_list[2])
            ymax = int(coords_list[3])
            
            # Determine the disease type separately for each opacity
            current_disease = determine_disease_type([new_label_list[j]])
            # Use the advanced bounding box function
            image = draw_bbox_with_disease(image, xmin, ymin, xmax, ymax, 
                            text=f"O{j+1}", 
                            line_thickness=2, 
                            disease_type=current_disease,
                            confidence=new_scores_list[j] if j < len(new_scores_list) else None)

        # Save the image
        dst = os.path.join('static/pred_images_dir/', fname)
        cv2.imwrite(dst, image)

    # Save the DataFrame
    path = 'df_fin_preds.csv'
    df_fin.to_csv(path, index=False)

    print('Prediction completed.')

    # Sort the images (opacities first)
    list1 = []
    list2 = []

    for key, value in num_preds_dict.items():
        if value[0] > 0:
            list1.append(key)
        else:
            list2.append(key)

    sorted_image_list = list1 + list2
    assert len(sorted_image_list) == len(image_list)

    return num_preds_dict, sorted_image_list, disease_types_dict, detailed_results

# Delete user-submitted data
def delete_user_submitted_data():
    """
    Clears the data uploaded by the user.
    """
    if os.path.isdir('uploads') == True:
        shutil.rmtree('uploads')

# Hide all bounding boxes
def hide_all_bboxes(image_fname):
    """
    Hides all bounding boxes.
    """
    if os.path.isdir('static/analysis_images_dir') == True:
        shutil.rmtree('static/analysis_images_dir')
        print('Folder deleted.')

    if not os.path.isdir('static/analysis_images_dir'):
        analysis_images_dir = 'static/analysis_images_dir'
        os.mkdir(analysis_images_dir)

    path = os.path.join('static/proc_images_dir', image_fname)
    image = cv2.imread(path)
    image_fname = 'no_bboxes_' + image_fname
    k = str(99)

    new_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/analysis_images_dir/{k}/{image_fname}" height="500" alt="Chest X-ray">"""

    if new_image_str != 'None':
        print('User clicked on the image.')
        if not os.path.isdir(f'static/analysis_images_dir/{k}'):
            analysis_images_dir = f'static/analysis_images_dir/{k}'
            os.mkdir(analysis_images_dir)
        dst = os.path.join(f'static/analysis_images_dir/{k}', image_fname)
        cv2.imwrite(dst, image)

    output = {'new_image_str': new_image_str}
    return output

# Show all bounding boxes
def show_all_bboxes(image_fname):
    """
    Shows all bounding boxes.
    """
    image_fname = image_fname.replace('no_bboxes_', '')
    new_image_str = f"""<img id="selected-image" onclick="get_click_coords(event, this.src)" class="w3-round unblock" src="/static/pred_images_dir/{image_fname}" height="500" alt="Chest X-ray">"""

    if new_image_str != 'None':
        print('User clicked on the image.')
        output = {'new_image_str': new_image_str}
    else:
        output = {'new_image_str': 'None'}
    return output