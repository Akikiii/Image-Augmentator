import json
import cv2
import os
import albumentations as A
import random
import shutil

# Directories for augmented images
output_image_dir = 'C:/Users/Admin/Desktop/Coding/Python/Image Augmentator/images/augmented_images'
train_dir = os.path.join(output_image_dir, 'train')
val_dir = os.path.join(output_image_dir, 'val')
test_dir = os.path.join(output_image_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load COCO JSON file
with open('C:/Users/Admin/Desktop/Coding/Python/Image Augmentator/result.json') as f:
    coco_data = json.load(f)

# Define a robust set of augmentations (without cropping or cutout)
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.3)
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_area=0, min_visibility=0.3))

# New target size
target_width = 800
target_height = 600

# Shuffle images for random split
random.shuffle(coco_data['images'])

# Split data into 80% training, 10% validation, 10% testing
total_images = len(coco_data['images'])
train_split = int(0.8 * total_images)
val_split = int(0.1 * total_images)

train_images = coco_data['images'][:train_split]
val_images = coco_data['images'][train_split:train_split + val_split]
test_images = coco_data['images'][train_split + val_split:]

# Ensure bounding boxes are within image bounds
def clip_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    w = max(0, min(w, img_width - x))
    h = max(0, min(h, img_height - y))
    return [x, y, w, h]

# Helper function to save images and annotations
def process_images(image_list, output_dir, coco_data, new_json_path):
    new_coco_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']}

    for img_data in image_list:
        img_id = img_data['id']
        img_path = img_data['file_name']  # Original image path

        # Load the image using OpenCV
        image = cv2.imread(img_path)

        # Collect bounding boxes and categories from annotations
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        bboxes = []
        category_ids = []

        for ann in annotations:
            bboxes.append(ann['bbox'])  # COCO bounding box format
            category_ids.append(ann['category_id'])  # Category label

        # Apply augmentations
        augmented = augmentations(image=image, bboxes=bboxes, category_ids=category_ids)

        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']  # Adjusted bounding boxes

        # Resize the augmented image to 800x600
        resized_image = cv2.resize(augmented_image, (target_width, target_height))  # (width, height)

        # Calculate the scaling factors for the bounding boxes
        x_scale = target_width / augmented_image.shape[1]
        y_scale = target_height / augmented_image.shape[0]

        # Resize and scale bounding boxes
        new_bboxes = []
        for bbox in augmented_bboxes:
            x, y, w, h = bbox
            # Scale bounding box coordinates to the new image size
            new_x = x * x_scale
            new_y = y * y_scale
            new_w = w * x_scale
            new_h = h * y_scale
            # Ensure the bounding box is within the image bounds
            new_bboxes.append(clip_bbox([new_x, new_y, new_w, new_h], target_width, target_height))

        # Save augmented image in the appropriate directory
        image_name = os.path.basename(img_path)
        aug_img_path = os.path.join(output_dir, image_name)

        cv2.imwrite(aug_img_path, resized_image)  # Save resized augmented image

        # Update the image metadata
        img_data['width'] = target_width
        img_data['height'] = target_height
        img_data['file_name'] = aug_img_path  # Update path to the new image

        # Update the annotations with new bbox coordinates
        for i, ann in enumerate(annotations):
            ann['bbox'] = new_bboxes[i]  # Update the bounding boxes
            ann['area'] = new_bboxes[i][2] * new_bboxes[i][3]  # Update area based on resized bbox

        new_coco_data['images'].append(img_data)
        new_coco_data['annotations'].extend(annotations)

    # Save new COCO JSON file for the split
    with open(new_json_path, 'w') as f:
        json.dump(new_coco_data, f)

# Process training data
train_json_path = os.path.join(train_dir, 'train_annotations.json')
process_images(train_images, train_dir, coco_data, train_json_path)

# Process validation data
val_json_path = os.path.join(val_dir, 'val_annotations.json')
process_images(val_images, val_dir, coco_data, val_json_path)

# Process testing data
test_json_path = os.path.join(test_dir, 'test_annotations.json')
process_images(test_images, test_dir, coco_data, test_json_path)

print(f"Train data saved to: {train_dir}")
print(f"Validation data saved to: {val_dir}")
print(f"Test data saved to: {test_dir}")
