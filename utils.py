import cv2
import tensorflow as tf
import os
import numpy as np

BASE_DIR = "/content/drive/MyDrive/Occluded Facial Expression Recognition/"
OCCLUSION_SIZE = (100, 100)
OCCLUSION_PATH = os.path.join(BASE_DIR, 'datasets/Occlusions')
LANDMARK_PREDICTOR_PATH = os.path.join(BASE_DIR, 'util_files/shape_predictor_68_face_landmarks.dat')

class OcclusionCreator():
  def __init__(self):
    path = OCCLUSION_PATH
    self.occlusion_paths = []
    for subdir, dirs, files in os.walk(path):
      for f in files:
        image_path = os.path.join(subdir, f)
        self.occlusion_paths.append(image_path)
    self.index = 0
  
  def resize(self, image_1, image_2):
    x1 = image_1.shape[1]
    y1 = image_1.shape[0]
    x2 = image_2.shape[1]
    y2 = image_2.shape[0]
    r = x2 // y2

    if r == 0:
      r = y2 // x2
      f = np.random.rand(30, 55)
      factor = np.random.randint(10, 15)
      factor = factor / 10
      seedY = np.random.randint(30, 55)
      seedX = int((seedY / r)*factor)
    elif r in (0.8, 1.2):
      seedX = np.random.randint(30, 55)
      seedY = int((seedX / r)*factor)     
    else:
      f = np.random.rand(30, 55)
      factor = np.random.randint(10, 15)
      factor = factor / 10
      seedX = np.random.randint(30, 55)
      seedY = int((seedX / r)*factor)

    if(x2 > x1 or y2 > y1 or x2 > x1 // 2 or y2 > x2 // 2):
      image_2 = cv2.resize(image_2, (seedX, seedY), interpolation = cv2.INTER_AREA) 
    return image_2

  def generate_coordinates(self, image_1, image_2):
    x1 = image_1.shape[1]
    y1 = image_1.shape[0]
    x2 = image_2.shape[1]
    y2 = image_2.shape[0]
    try:
      seedX = np.random.randint(0, (x1 - x2))
      seedY = np.random.randint(0, (y1 - y2))
    except:
      seedX = 50
      seedY = 50
    return seedX, seedY

  def overlay(self, image, occlusion, coordinates):
    x_offset, y_offset = coordinates[0], coordinates[1]
    x1, x2 = x_offset, x_offset + occlusion.shape[1]
    y1, y2 = y_offset, y_offset + occlusion.shape[0]
    occ_alpha = (occlusion[:, :, 3] / 255.0)
    img_alpha = 1.0 - occ_alpha
    for c in range(0, 3):
      image[y1:y2, x1:x2, c] = (occ_alpha*occlusion[:, :, c] + img_alpha*image[y1:y2, x1:x2, c])
    return image

  def impose(self, x_batch):
    occluded_images = []
    for i in range(x_batch.shape[0]):
      image = x_batch[i]
      occlusion_path = self.occlusion_paths[self.index]
      self.index = (self.index + 1) % len(self.occlusion_paths)
      occlusion = cv2.imread(occlusion_path, cv2.IMREAD_UNCHANGED)
      occlusion = self.resize(image, occlusion)
      coordinates = self.generate_coordinates(image, occlusion)
      occluded_image = self.overlay(image, occlusion, coordinates)
      occluded_images.append(occluded_image)
    return tf.convert_to_tensor(occluded_images)
