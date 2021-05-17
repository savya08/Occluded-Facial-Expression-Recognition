import tensorflow as tf
import numpy as np
from utils import OcclusionCreator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

RANDOM_SEED = 100
tf.random.set_seed(RANDOM_SEED)

class BaseModel(tf.keras.Model):
  def __init__(self):
    super(BaseModel, self).__init__()
    self.input_layer = tf.keras.Input(shape = (224, 224, 3))
    self.base = tf.keras.applications.resnet50.ResNet50(weights = 'imagenet', input_shape = (224, 224, 3), include_top = False)
    for layer in self.base.layers:
      if layer.__class__.__name__ == 'BatchNormalization':
        layer.trainable = False
    
    self.dropout = tf.keras.layers.Dropout(rate = 0.4, seed = 7)
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(units = 7)

    output = self.base.get_layer('conv3_block4_3_conv').output
    self.model = tf.keras.Model(inputs = [self.base.input], outputs = [output])

  def call(self, x):
    x = tf.keras.applications.resnet50.preprocess_input(x)
    z = self.base(x)
    z = self.dropout(z)
    z = self.flatten(z)
    z = self.dense(z)
    z = tf.keras.activations.softmax(z)
    return z

  def feature_map(self, x):
    return self.model(x)


class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
    self.conv2 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
    self.conv3 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
    self.conv4 = tf.keras.layers.Conv2D(filters = 1024, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
    self.conv5 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), strides = (2, 2), padding = 'same')

  def call(self, h):
    z = self.conv1(h)
    z = tf.nn.leaky_relu(z)

    z = self.conv2(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.nn.leaky_relu(z)

    z = self.conv3(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.nn.leaky_relu(z)

    z = self.conv4(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.nn.leaky_relu(z)

    z = self.conv5(z)
    z = tf.squeeze(z)
    return z

  
class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    self.deconv1 = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
    self.deconv2 = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
    self.deconv3 = tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = (3, 3), strides = (2, 2), padding = 'same')
    
  def call(self, h):
    z = self.deconv1(h)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.activations.relu(z)

    z = self.deconv2(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.activations.relu(z)

    z = self.deconv3(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.activations.relu(z)

    reconstructed_image = tf.math.scalar_mul(255, z)
    return reconstructed_image


class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.occluded_net = BaseModel()
    self.non_occluded_net = BaseModel()
    self.discriminator = Discriminator()
    self.decoder = Decoder()
    self.occ_model = None
    self.occlusion_creator_1 = OcclusionCreator()
    self.occlusion_creator_2 = OcclusionCreator()

  def build_architecture(self):
    x = tf.keras.Input(shape = (224, 224, 3))
    h = tf.keras.Input(shape = (28, 28, 512))
    y = self.occluded_net(x)
    x_rec = self.decoder(h)
    model = tf.keras.Model(inputs = [x, h], outputs = [y, x_rec])
    return model

  def evaluate(self, dataset):
    scores = self.occluded_net.evaluate(dataset)[1]
    return scores

if __name__ == '__main__':
  model = Decoder()
  x_train = np.random.random((10, 28, 28, 512))
  y_pred = model(x_train)
  print(y_pred.shape)
  