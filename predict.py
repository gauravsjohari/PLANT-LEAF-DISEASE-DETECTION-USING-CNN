# -*- coding: utf-8 -*-

import tensorflow as tf 
classifierLoad = tf.keras.models.load_model('model.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('late_blight.JPG', target_size = (200,200))
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifierLoad.predict(test_image)
if result[0][1] == 1:  
    print("Potato___Early_blight")
elif result[0][0] == 1:
    print("Potato___healthy")
elif result[0][2] == 1:
    print("Potato___Late_blight")




