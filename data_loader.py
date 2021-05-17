import tensorflow as tf
import os
import dlib
import cv2
import numpy as np
import random
import pickle5 as pickle
from matplotlib import pyplot as plt

WORKERS = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = (224, 224, 3)
OCCLUSION_SIZE = (110, 110)
BATCH_SIZE = 32
POOL_SIZE = 6
PATCH_SIZE = 24
BUFFER_SIZE = 10000

BASE_DIR = "/content/drive/MyDrive/Occluded Facial Expression Recognition/"
RAFDB_PATH = os.path.join(BASE_DIR, 'datasets/RAF-DB/Images')
AFFECT_PATH_train = os.path.join(BASE_DIR, 'datasets/affect_net_data')
AFFECT_PATH_val = os.path.join(BASE_DIR, 'datasets/affect_net_validation')
OCCLUSION_PATH = os.path.join(BASE_DIR, 'datasets/Occlusions')
FEDRO_PATH = os.path.join(BASE_DIR, 'datasets/FED-RO/FED-RO_crop')
LANDMARK_PREDICTOR_PATH = os.path.join(BASE_DIR, 'util_files/shape_predictor_68_face_landmarks.dat')

class Dataset():
  def __init__(self, batch_size = BATCH_SIZE, image_shape = IMAGE_SIZE, patch_size = PATCH_SIZE, create_occlusion = False, rafdb = False, affectnet = False, fedro = False):
    self.batch_size = batch_size
    self.image_height = image_shape[0]
    self.image_width = image_shape[1]
    self.label_dict = {'surprise': 0, 'fear': 1, 'disgust': 2, 'happiness': 3, 'sadness': 4, 'anger': 5, 'neutral': 6}
    self.num_classes = len(list(self.label_dict))
    self.x_train, self.x_test, self.y_train, self.y_test = [], [], [], []
    self.create_occlusion = create_occlusion
    
    if rafdb == True:
      for subdir, dirs, files in os.walk(RAFDB_PATH):
        for f in files:
          image_path = os.path.join(subdir, f)
          if f[:5] == 'train':
            self.x_train.append(image_path)
            label = image_path.split('/')[-2]
            self.y_train.append(self.label_dict[label])
          else:
            self.x_test.append(image_path)
            label = image_path.split('/')[-2]
            self.y_test.append(self.label_dict[label])

    if affectnet == True:
      for subdir, dirs, files in os.walk(AFFECT_PATH_train):
        for f in files:
          image_path = os.path.join(subdir, f)      
          label = image_path.split('/')[-2].lower()
          if self.label_dict.get(label) is None: 
            continue
          self.x_train.append(image_path)
          self.y_train.append(self.label_dict[label])
        
      for subdir, dirs, files in os.walk(AFFECT_PATH_val):
        for f in files:
          image_path = os.path.join(subdir, f)      
          label = image_path.split('/')[-2].lower()
          if self.label_dict.get(label) is None:
            continue
          self.x_test.append(image_path)
          self.y_test.append(self.label_dict[label])
    
    if fedro == True:
      for subdir, dirs, files in os.walk(FEDRO_PATH):
        for f in files:
          image_path = os.path.join(subdir, f)      
          label = image_path.split('/')[-2].lower()
          if self.label_dict.get(label) is None:
            continue
          self.x_test.append(image_path)
          self.y_test.append(self.label_dict[label])
    
    path = OCCLUSION_PATH
    self.occlusion_paths = []
    for subdir, dirs, files in os.walk(path):
      for f in files:
        image_path = os.path.join(subdir, f)
        self.occlusion_paths.append(image_path)

    self.detector = dlib.get_frontal_face_detector()
    predictor_path = LANDMARK_PREDICTOR_PATH
    self.predictor = dlib.shape_predictor(predictor_path)
    self.points = [19, 22, 23, 26, 39, 37, 44, 46, 28, 30, 49, 51, 53, 55, 59, 57]
    
    self.occlusion_size = OCCLUSION_SIZE[0]
    self.index = 0 

  def parse_function(self, image_path, image_label):
    def impose_occlusion(image):
      image = image.numpy()
      image = image.astype('uint8')
      occlusion_path = self.occlusion_paths[self.index]
      self.index = (self.index + 1) % len(self.occlusion_paths)

      occlusion = cv2.imread(occlusion_path, cv2.IMREAD_UNCHANGED)
      
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = self.detector(gray_image)
      if len(faces) == 0:
        faces = dlib.rectangles()
        faces.append(dlib.rectangle(0, 0, image.shape[0], image.shape[0]))
      for face in faces:
        landmarks_predictor = self.predictor(gray_image, face)
        n = self.points[np.random.randint(0, len(self.points) - 1)]
        x = landmarks_predictor.part(n).x
        y = landmarks_predictor.part(n).y
      
      x_left = max(0, x - self.occlusion_size // 2)  
      x_right = min(image.shape[1], x + self.occlusion_size // 2)
      y_top = max(0, y - self.occlusion_size // 2)
      y_bottom = min(image.shape[0], y + self.occlusion_size // 2)

      occlusion = cv2.resize(occlusion, (x_right - x_left, y_bottom - y_top), interpolation = cv2.INTER_AREA)

      occ_alpha = occlusion[:, :, 3] / 255.
      img_alpha = 1 - occ_alpha

      for c in range(3):
        image[y_top:y_bottom, x_left:x_right, c] = occ_alpha*occlusion[:, :, c] + img_alpha*image[y_top:y_bottom, x_left:x_right, c]
      return image 

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [self.image_width, self.image_height])
    if self.create_occlusion == True:
      image = tf.py_function(func = impose_occlusion, inp = [image], Tout = tf.float32)
      image.set_shape((224, 224, 3))
    label = tf.one_hot(image_label, self.num_classes)
    return image, label
  
  def get_train_ds(self):
    BUFFER_SIZE = len(self.x_train)
    ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.repeat()
    ds = ds.map(self.parse_function, num_parallel_calls = WORKERS)
    ds = ds.batch(self.batch_size, drop_remainder = True)
    ds = ds.prefetch(buffer_size = WORKERS)
    return ds
  
  def get_test_ds(self):
    BUFFER_SIZE = len(self.x_test)
    ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.repeat(1)
    ds = ds.map(self.parse_function, num_parallel_calls = WORKERS)
    ds = ds.batch(self.batch_size, drop_remainder = True)
    ds = ds.prefetch(buffer_size = WORKERS)
    return ds

  def get_fedro_ds(self):
    BUFFER_SIZE = len(self.x_test)
    ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.repeat(1)
    ds = ds.map(self.parse_function, num_parallel_calls = WORKERS)
    ds = ds.batch(self.batch_size, drop_remainder = True)
    ds = ds.prefetch(buffer_size = WORKERS)
    return ds

if __name__ == '__main__':
  dataset = Dataset(create_occlusion = True, rafdb = True)
  train_ds = dataset.get_train_ds()
  test_ds = dataset.get_test_ds()
  for x_batch, y_batch in train_ds:
    x_batch = x_batch.numpy()
    print(x_batch.shape)
    print(y_batch.shape)
    for i in range(5):
      image = x_batch[i].astype('uint8')
      print(image.shape)
      plt.imsave('image_%s.jpg'%(i), image)
    break
