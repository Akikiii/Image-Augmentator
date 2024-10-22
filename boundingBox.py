import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import os

# Load the COCO annotations
#C:\Users\Admin\Desktop\Coding\Python\Image Augmentator\result.json
# C:\Users\Admin\Desktop\Coding\Python\Image Augmentator\images\augmented_images\test\test_annotations.json
with open(r'C:\Users\Admin\Desktop\Coding\Python\Image Augmentator\images\augmented_images\test\test_annotations.json') as f:
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

# Initialize the current image index
current_image_idx = 0

# Function to plot images with bounding boxes
def plot_image_with_bboxes(image_id):
    # Get the image path
    image_path = image_dict[image_id]
    img = Image.open(image_path)

    # Clear the previous plot
    plt.clf()

    # Plot the image
    plt.imshow(img)
    plt.axis('off')

    # Draw bounding boxes
    if image_id in annotations_dict:
        for bbox in annotations_dict[image_id]:
            x, y, width, height = bbox
            rect = plt.Rectangle((x, y), width, height, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)

    plt.draw()

# Callback function for "Next" button
def next_image(event):
    global current_image_idx
    current_image_idx = (current_image_idx + 1) % len(coco_data['images'])  # Circular navigation
    next_image_id = coco_data['images'][current_image_idx]['id']
    plot_image_with_bboxes(next_image_id)

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

# Plot the first image
first_image_id = coco_data['images'][current_image_idx]['id']
plot_image_with_bboxes(first_image_id)

# Add a "Next" button
ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])  # Position for the button
btn_next = Button(ax_next, 'Next')
btn_next.on_clicked(next_image)

plt.show()
