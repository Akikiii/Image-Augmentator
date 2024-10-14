import json
import matplotlib.pyplot as plt
from PIL import Image
import os
# C:\Users\Admin\Desktop\Coding\Python\Image Augmentator\images\augmented_images\augmented_annotations.json
# Load the COCO annotations
with open(r'C:\Users\Admin\Desktop\Coding\Python\Image Augmentator\images\augmented_images\augmented_annotations.json') as f:
    coco_data = json.load(f)

# Create a mapping from image_id to filename
image_dict = {image['id']: image['file_name'] for image in coco_data['images']}

# Create a mapping from annotation id to bounding box details
annotations_dict = {}
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    if image_id not in annotations_dict:
        annotations_dict[image_id] = []
    annotations_dict[image_id].append(annotation['bbox'])  # You can also include other details if needed

# Function to plot images with bounding boxes
def plot_image_with_bboxes(image_id):
    # Get the image path
    image_path = image_dict[image_id]
    img = Image.open(image_path)

    # Plot the image
    plt.imshow(img)
    plt.axis('off')

    # Draw bounding boxes
    if image_id in annotations_dict:
        for bbox in annotations_dict[image_id]:
            x, y, width, height = bbox
            rect = plt.Rectangle((x, y), width, height, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)

    plt.show()

# Example usage: Plot the first image and its bounding boxes
first_image_id = coco_data['images'][25]['id']
plot_image_with_bboxes(first_image_id)
