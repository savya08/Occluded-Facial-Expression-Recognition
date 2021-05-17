import tensorflow as tf
from data_loader import Dataset, BATCH_SIZE
from models import BaseModel, Model
from utils import OcclusionCreator
import sys
import os
import cv2
import numpy as np
import warnings
import time
import multiprocessing


BASE_DIR = '/content/drive/MyDrive/Occluded Facial Expression Recognition/'
NON_OCC_CKPT = os.path.join(BASE_DIR, 'checkpoints/non_occ_net.ckpt')
OCC_CKPT = os.path.join(BASE_DIR, 'checkpoints/occ_net.ckpt')
MODEL_CKPT = os.path.join(BASE_DIR, 'checkpoints/full_model.ckpt')
MODEL_OCC_CKPT = os.path.join(BASE_DIR, 'checkpoints/full_model_occ.ckpt')
MODEL_OUTPUT_LOG_FILE = os.path.join(BASE_DIR, 'logs/full_model_log')
OCC_NET_OUTPUT_LOG_FILE = os.path.join(BASE_DIR, 'logs/occ_model_log')

BASE_OCC_LR = 0.001
NON_OCC_LR = 0.001
DISC_LR = 0.0001
OCC_LR = 0.0001
K1 = 1
K2 = 2
BASE_NUM_EPOCHS = 20
STEPS_PER_EPOCHS = 200
NUM_EPOCHS = 100
LAMBDAS = tf.constant([0.2, 0.175, 0.25, 0.25])
CLASS_WEIGHTS = {4: 1.81, 6: 1.48, 2: 1.97, 5: 1.83, 0: 1.89, 3: 1.00, 1: 1.95}

def train_non_occluded_net(train_ds, test_ds, num_epochs = BASE_NUM_EPOCHS, learning_rate = NON_OCC_LR, batch_size = BATCH_SIZE):
    non_occluded_net = BaseModel()
    optimizer = tf.keras.optimizers.SGD(learning_rate = 2.0*learning_rate, momentum = 0.9)

    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta = 0.00001, mode = 'min', patience = 10)
    ckpt_path = NON_OCC_CKPT
    checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only = True, save_best_only = True)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate*tf.math.exp(-0.08*epoch))

    non_occluded_net.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    if os.path.exists(ckpt_path + '.index'):
        non_occluded_net.load_weights(ckpt_path).expect_partial()

    non_occluded_net.fit(train_ds, epochs = num_epochs, steps_per_epoch = STEPS_PER_EPOCHS, 
                         validation_data = test_ds,
                         callbacks = [checkpoint, lr_scheduler],
                         class_weight = CLASS_WEIGHTS,
                         verbose = 2,
                         max_queue_size = 5000, workers = multiprocessing.cpu_count(), use_multiprocessing = True)

    non_occluded_net.load_weights(ckpt_path).expect_partial()

def train_occluded_net(train_ds, test_ds, num_epochs = BASE_NUM_EPOCHS, learning_rate = BASE_OCC_LR, batch_size = BATCH_SIZE):
    occluded_net = BaseModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta = 0.00001, mode = 'min', patience = 10)
    ckpt_path = OCC_CKPT
    checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only = True, save_best_only = True)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate*tf.math.exp(-0.08*epoch))
    
    occluded_net.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    if os.path.exists(ckpt_path + '.index'):
        occluded_net.load_weights(ckpt_path).expect_partial()

    occluded_net.fit(train_ds, epochs = num_epochs, steps_per_epoch = STEPS_PER_EPOCHS, 
                     validation_data = test_ds, 
                     callbacks = [checkpoint, lr_scheduler],
                     class_weight = CLASS_WEIGHTS,
                     verbose = 2,
                     max_queue_size = 5000, workers = multiprocessing.cpu_count(), use_multiprocessing = True)

    occluded_net.load_weights(ckpt_path).expect_partial()

def train(train_ds, test_ds, lambdas = LAMBDAS, num_epochs = NUM_EPOCHS, k1 = K1, k2 = K2, disc_lr = DISC_LR, occ_lr = OCC_LR, batch_size = BATCH_SIZE):
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate = disc_lr)
    occ_optimizer = tf.keras.optimizers.Adam(learning_rate = occ_lr)
    disc_loss = tf.keras.metrics.Mean(name = 'disc_loss')
    occ_loss = tf.keras.metrics.Mean(name = 'occ_loss')
    occ_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'occ_accuracy')
    disc_loss_log = []
    occ_loss_log = []
    accuracy_log = []
    fedro_accuracy = tf.Variable(0.0)

    model = Model()
    occluded_net = model.occluded_net
    occluded_net.compile(loss = 'categorical_crossentropy', optimizer = occ_optimizer, metrics = ['accuracy'])
    occluded_net.load_weights(OCC_CKPT).expect_partial()
    non_occluded_net = model.occluded_net
    non_occluded_net.compile(loss = 'categorical_crossentropy', optimizer = occ_optimizer, metrics = ['accuracy'])
    non_occluded_net.load_weights(NON_OCC_CKPT).expect_partial()
    discriminator = model.discriminator
    decoder = model.decoder
    occlusion_creator_1 = model.occlusion_creator_1
    occlusion_creator_2 = model.occlusion_creator_2

    for epoch in range(num_epochs):
        start_time = time.time()
        disc_loss.reset_states()
        batch_iter = iter(train_ds)
        for k in range(k1):
            x_batch, _ = next(batch_iter)
            occ_x_batch = occlusion_creator_1.impose(x_batch.numpy())
            ho_batch = occluded_net.feature_map(occ_x_batch)
            hc_batch = non_occluded_net.feature_map(x_batch)
            with tf.GradientTape() as tape:
                d_batch = tf.zeros((batch_size, ))
                d_pred = discriminator(ho_batch)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(d_batch, d_pred)

                d_batch = tf.ones((batch_size, ))
                d_pred = discriminator(hc_batch)
                loss += tf.nn.sigmoid_cross_entropy_with_logits(d_batch, d_pred)

            gradients = tape.gradient(loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
            disc_loss(loss)
        disc_loss_log.append(disc_loss.result())

        template = 'Epoch {}/{} \n discriminator_loss: {:.4f}'
        print(template.format(epoch + 1, num_epochs, disc_loss.result()))
        with open(MODEL_OUTPUT_LOG_FILE, 'a') as f:
            f.write(template.format(epoch + 1, num_epochs, disc_loss.result()) + '\n')

        occ_loss.reset_states()
        occ_accuracy.reset_states()
        batch_iter = iter(train_ds)
        for k in range(k2):
            x_batch, y_batch = next(batch_iter)
            occ_x_batch = occlusion_creator_2.impose(x_batch.numpy())
            ho = occluded_net.feature_map(occ_x_batch)
            with tf.GradientTape() as tape:
                yo = occluded_net((occ_x_batch))
                yc = non_occluded_net(x_batch)
                d_ho = discriminator(ho)
                x_rec = decoder(ho)

                l_sup = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_batch, yo))
                l_sim = tf.reduce_mean(tf.keras.losses.MSE(yo, yc))
                l_lir = tf.math.maximum(0, tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_batch, yc)) - l_sup)
                l_adv = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(d_ho) + 1e-10))
                l_rec = tf.reduce_mean(tf.keras.losses.MSE(tf.reshape(x_batch, (batch_size, -1)), tf.reshape(x_rec, (batch_size, -1))))
                
                loss = l_sup + lambdas[0]*l_sim + lambdas[1]*l_lir + lambdas[2]*l_adv + lambdas[3]*l_rec
            trainable_variables = occluded_net.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            occ_optimizer.apply_gradients(zip(gradients, trainable_variables))

            occ_loss(loss)
            occ_accuracy(y_batch, yo)
        occ_loss_log.append(occ_loss.result())
        accuracy_log.append(occ_accuracy.result())
        end_time = time.time()

        template = ' model_loss: {:.4f} - train_accuracy: {:.4f} - test_accuracy: {:.4f} - time: {:.2f}'
        print(template.format(occ_loss.result(), occ_accuracy.result(), occluded_net.evaluate(test_ds, verbose = 0)[1], end_time - start_time))
        with open(MODEL_OUTPUT_LOG_FILE, 'a') as f:
            f.write(template.format(occ_loss.result(), occ_accuracy.result(), occluded_net.evaluate(test_ds, verbose = 0)[1], end_time - start_time) + '\n\n')

if __name__ == '__main__':
    if sys.argv[-1] == '0':
        print('Training non-occluded net...\n')
        dataset = Dataset(rafdb = True, affectnet = False, fedro = False)
        train_ds = dataset.get_train_ds()
        test_ds = dataset.get_test_ds()
        train_non_occluded_net(train_ds, test_ds)

    if sys.argv[-1] == '1':
        print('Training occluded net...\n')
        dataset = Dataset(create_occlusion = True, rafdb = True, affectnet = False, fedro = False)
        train_ds = dataset.get_train_ds()
        test_ds = dataset.get_test_ds()
        train_occluded_net(train_ds, test_ds)

    if sys.argv[-1] == '2':
        print('Training full model...\n')
        dataset = Dataset(create_occlusion = True, rafdb = True, affectnet = False, fedro = False)
        train_ds = dataset.get_train_ds()
        test_ds = dataset.get_test_ds()
        train(train_ds, test_ds)
