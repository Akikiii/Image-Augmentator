import json
import cv2
import os
import albumentations as A #Install Albumentations before running code


output_image_dir = 'C:\\Users\\Admin\\Desktop\\Coding\\Python\\Image Augmentator\\images\\augmented_images'
os.makedirs(output_image_dir, exist_ok=True)

with open('result.json') as f:
    coco_data = json.load(f)

augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),                # Flip horizontally
    A.VerticalFlip(p=0.5),                  # Flip vertically
    A.RandomBrightnessContrast(p=0.2),       # Random brightness and contrast
    A.Rotate(limit=30, p=0.5)                # Random rotation
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Loop over the images in COCO format
for img_data in coco_data['images']:
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
    
    # Resize the augmented image to 600x800
    resized_image = cv2.resize(augmented_image, (800, 600))  # Note: (width, height)
    
    # Draw bounding boxes and labels on the resized image
    for bbox, category_id in zip(augmented_bboxes, category_ids):
        x, y, w, h = map(int, bbox)  # Convert bounding box to integer
        stage_label = coco_data['categories'][category_id]['name']  # Get the stage name
        
        # Resize bounding box coordinates according to the resized image
        x_scale = 800 / augmented_image.shape[1]
        y_scale = 600 / augmented_image.shape[0]
        x, y, w, h = int(x * x_scale), int(y * y_scale), int(w * x_scale), int(h * y_scale)

        # Draw the bounding box
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put label text above the bounding box
        cv2.putText(resized_image, stage_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the augmented image with the same name in the new directory
    image_name = os.path.basename(img_path)  # Extract the image name from path
    aug_img_path = os.path.join(output_image_dir, image_name)  # New augmented image path
    
    cv2.imwrite(aug_img_path, resized_image)  # Save resized augmented image
    
    # Update the image file path in the COCO JSON for the augmented image
    img_data['file_name'] = aug_img_path  # Update path in JSON

    # Update the annotations with new bbox coordinates
    for i, ann in enumerate(annotations):
        ann['bbox'] = augmented_bboxes[i]  # Update the bounding boxes

# Save the updated COCO annotations with new augmented data
new_coco_json_path = os.path.join(output_image_dir, 'augmented_annotations.json')
with open(new_coco_json_path, 'w') as f:
    json.dump(coco_data, f)

print(f"Augmented images saved to: {output_image_dir}")
print(f"Updated COCO annotations saved to: {new_coco_json_path}")
