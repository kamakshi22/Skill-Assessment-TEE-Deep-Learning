
import tensorflow as tf
import numpy as np
from VGG import Vgg16
from datetime import datetime
import os
import sys

import pandas as pd
import numpy as np

batch_size = 64

num_preprocess_threads = 1
min_queue_examples = 1
dropout_rate =1
learning_rate=1e-4

selectedType = '1'


args = sys.argv
if len(args) > 0:
    selectedType = args[1]  
     

if selectedType == '1':
    GENERAL_EX = './checkpoints/Gen_Impressions_original_data_frames/'
    PERCENTAGE = './checkpoints/Percentage_original_data_frames/'
    OUTPUT_NAME = 'original_data_frames'
    DATA_FOLDER = '../original_data_frames'
    print('Orig') 

elif selectedType == '2':
    GENERAL_EX = './checkpoints/Gen_Impressions_canny_data_frames/'
    PERCENTAGE = './checkpoints/Percentage_canny_data_frames/'
    OUTPUT_NAME = 'canny_data_frames'
    DATA_FOLDER = '../canny_data_frames'
    print('Canny')
    
elif selectedType == '3':
    GENERAL_EX = './checkpoints/Gen_Impressions_enhanced_data_frames/'
    PERCENTAGE = './checkpoints/Percentage_enhanced_data_frames/'    
    OUTPUT_NAME = 'enhanced_data_frames'
    DATA_FOLDER = '../enhanced_data_frames'
    print('enhanced')

elif selectedType == '4':
    GENERAL_EX = './checkpoints/Gen_Impressions_segmentation_data_frames/'
    PERCENTAGE = './checkpoints/Percentage_segmentation_data_frames/'
    OUTPUT_NAME = 'segmentation_data_frames'
    DATA_FOLDER = '../segmentation_data_frames'
    print ('segmentation')

elif selectedType == '5':
    GENERAL_EX = './checkpoints/Gen_Impressions_segmentation_contour_data_frames/'
    PERCENTAGE = './checkpoints/Percentage_segmentation_contour_data_frames/'
    OUTPUT_NAME = 'segmentation_contour_data_frames'
    DATA_FOLDER = '../segmentation_contour_data_frames'
    print ('segmentation_contour')

def getImageLabel(path, Name):
    filename_queue = tf.train.string_input_producer([path])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    record_defaults = [["no.jpg"], [1], [1.], [1.], [1.]]
    Img_Path, Class_Category, Criteria_Avg, Criteria_Per, Gen_Impression = tf.decode_csv(value,
                                                                                  record_defaults=record_defaults)
    d={
        "Criteria_Per": Criteria_Per,
        "Gen_Impression": Gen_Impression,
        "Class_Category": Class_Category
         }
    Label= d[Name]

    img = tf.image.decode_jpeg(tf.read_file(DATA_FOLDER +'/'+Img_Path))

    img = tf.image.resize_images(img, [186, 224])
    img= tf.image.resize_image_with_crop_or_pad(img,224, 224)


    img.set_shape([224, 224, 3])
    img = img
    img = tf.image.convert_image_dtype(img, tf.float32)

    image_b, label_b = tf.train.batch([img, Label],
                                      batch_size=batch_size,
                                      num_threads=num_preprocess_threads,
                                      capacity=min_queue_examples + 3 * batch_size,
                                      )
    label_b = tf.expand_dims(label_b, 1)

    return image_b, label_b



def Regression_criteria_percentage():
    num_classes=1


    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)

    model = Vgg16()
    model.build(x)
    score = tf.layers.dense(model.fc8, 1)
    saver = tf.train.Saver()


    with tf.Session() as sess:

        image_bt, label_bt = getImageLabel("../test_6people_copy.csv", Name= "Criteria_Per")

        print('inside Evaluation')
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess, PERCENTAGE+"model_epoch60.ckpt")

        predicted_values= []
        ground_truth=[]

        for step in range(0, 41):


            imageBt, labelBt = sess.run([image_bt, label_bt])

            pred = sess.run(score, feed_dict={x: imageBt, keep_prob: 1.})
            predf= np.clip(pred,0,100)
            predicted_values= np.append(predicted_values, np.array([predf]))
            ground_truth= np.append(ground_truth, np.array([labelBt]))

        coord.request_stop()
        coord.join(threads)

        print("Predicted values :")
        print(predicted_values)
        print("Ground truth : ")
        print(ground_truth)
        mse = (((predicted_values - ground_truth) ** 2).mean(axis=None))
        rmse = np.sqrt(((predicted_values - ground_truth) ** 2).mean(axis=None))
        print("Root mean squared error is : " + str(rmse))
        print("mean squared error is : " + str(mse))

        return predicted_values


def Regression_Gen_Expressions():

    tf.reset_default_graph()
    num_classes=1

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)

    model = Vgg16()
    model.build(x)
    score = tf.layers.dense(model.fc8, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        image_bt, label_bt = getImageLabel("../test_6people_copy.csv",Name= "Gen_Impression")

        print('inside Evaluation')

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess, GENERAL_EX+"model_epoch60.ckpt")

        predicted_values= []
        ground_truth=[]

        for step in range(0, 41):


            imageBt, labelBt = sess.run([image_bt, label_bt])

            pred = sess.run(score, feed_dict={x: imageBt, keep_prob: 1.})
            predf= np.clip(pred,0,100)
            predicted_values= np.append(predicted_values, np.array([predf]))
            ground_truth= np.append(ground_truth, np.array([labelBt]))

        coord.request_stop()
        coord.join(threads)

        print("Predicted values :")
        print(predicted_values)
        print("Ground truth : ")
        print(ground_truth)
        mse = (((predicted_values - ground_truth) ** 2).mean(axis=None))
        rmse = np.sqrt(((predicted_values - ground_truth) ** 2).mean(axis=None))
        print("Root mean squared error is : " + str(rmse))
        print("mean squared error is : " + str(mse))

        return predicted_values




a= Regression_criteria_percentage()
print("\n")
b= Regression_Gen_Expressions()
print("\n")



print("the new")

#Alexnet= pd.read_csv("../predicted_Alexnet.csv")
test= pd.read_csv("../test_6people_copy.csv")
l = len(test)
predicted= pd.DataFrame()
predicted["Select_id"]= test["Select_id"]
predicted["Predicted_Class"]=test["Class"]
predicted["Predicted_Criteria_Pct"]= a[0:l]
predicted["Predicted_Gen_Impression"]= b[0:l]
predicted["Ground_truth_Criteria_Pct"] = test["Criteria_Pct"]
predicted["Ground_truth_Gen_Impression"] = test["Gen_Impression"]

predicted.to_csv(OUTPUT_NAME+"/predicted_VGG.csv", encoding='utf-8')



