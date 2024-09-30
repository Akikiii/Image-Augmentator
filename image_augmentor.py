import Augmentor   
import numpy as np
from PIL import Image 
import cv2 #OpenCV 



#Gaussian Noise Simulator Not Editable lods except sa probabilty(Since yung probability is based on picking picture pref wag na galawin)

#P.S. Incompatible ang NumPy with Augmentor so naglagay tayo ng convert to PIL sa dulo para ma process(thanks gpt HAHA)

class GaussianNoise(Augmentor.Operations.Operation):
    def __init__(self, probability, mean=0, stddev=0.1):
        super(GaussianNoise, self).__init__(probability)
        self.mean = mean
        self.stddev = stddev

    def perform_operation(self, images):
        processed_images = []
        for image in images:
            np_img = np.array(image)
            
            np_img = np.float32(np_img) / 255.0
            gaussian_noise = np.random.normal(self.mean, self.stddev, np_img.shape)
            noisy_image = np_img + gaussian_noise
            noisy_image = np.clip(noisy_image, 0, 1)
            noisy_image = np.uint8(noisy_image * 255)

            noisy_image = Image.fromarray(noisy_image)
            processed_images.append(noisy_image)
        
        return processed_images

p = Augmentor.Pipeline("") #Image Path goes here example : C:/Users/Admin/Desktop/ThesisPics/FINAL


p.add_operation(GaussianNoise(probability=1.0, mean=0, stddev=0.1))

# Docs : https://augmentor.readthedocs.io/en/stable/
# Ang Probability is based kung mapipili ba yung picture or hindi *to limit duplicates* sa augmentaion process

p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=8)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)    
p.flip_left_right(probability=0.8)  
p.flip_top_bottom(probability=0.3)  
p.crop_random(probability=1, percentage_area=0.5)   
p.resize(probability=1.0, width=800, height=600)    

p.process() #lahat ng nasa folder scan
#p.sample(*ilang picture # goes here*)


