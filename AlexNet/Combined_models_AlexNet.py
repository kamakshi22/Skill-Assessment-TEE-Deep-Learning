
import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from datetime import datetime
import os
import sys

import pandas as pd
import numpy as np

batch_size = 128
num_preprocess_threads = 1
min_queue_examples = 1
dropout_rate =1
learning_rate=1e-4


selectedType = '1'

CLASSIFICATION = './checkpoints/Classification/'

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

def getImageLabel(path, Name,DATA_FOLDER):
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
    #pre-processing

    img = tf.image.resize_images(img, [189, 227])
    img= tf.image.resize_image_with_crop_or_pad(img,227, 227)


    img.set_shape([227, 227, 3])
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



    num_classes=10


    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    model = AlexNet(x, keep_prob, num_classes,[])
    score = model.fc8
    score = tf.layers.dense(model.fc8, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        image_bt, label_bt = getImageLabel("../test_6people_new.csv", Name= "Criteria_Per",DATA_FOLDER=DATA_FOLDER)

        print('inside Evaluation')
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess, PERCENTAGE+"model_epoch16.ckpt")

        predicted_values= []
        ground_truth=[]

        for step in range(0, 21):


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

    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    model = AlexNet(x, keep_prob, num_classes,[])

    score = model.fc8
    score = tf.layers.dense(model.fc8, 1)
    saver = tf.train.Saver()


    with tf.Session() as sess:

        image_bt, label_bt = getImageLabel("../test_6people_new.csv",Name= "Gen_Impression",DATA_FOLDER=DATA_FOLDER)

        print('inside Evaluation')
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess, GENERAL_EX+"model_epoch16.ckpt")

        predicted_values= []
        ground_truth=[]

        for step in range(0, 21):


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


def Classification():

    tf.reset_default_graph()
    num_classes = 10
    


    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batch_size, 10])
    with tf.name_scope('keepprob'):
        keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = AlexNet(x, keep_prob, num_classes, [])

    # Link variable to model output
    score = model.fc8
    softmax = tf.nn.softmax(score)

    correct_pred = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()


    with tf.Session() as sess:
        
        DATA_FOLDER ="../original_data_frames"
        image_bt, label_bt = getImageLabel("../test_6people_new.csv", Name="Class_Category",DATA_FOLDER= DATA_FOLDER )

        print('inside Evaluation')
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # define activation of last layer as score
        saver.restore(sess, CLASSIFICATION+"model_epoch16.ckpt")

        predicted_values = []
        ground_truth = []
        acc_list = []

        for step in range(0, 21):

            imageBt, labelBt = sess.run([image_bt, label_bt])
            one_hot_labels_test = np.zeros((batch_size, num_classes))
            for i in range(len(labelBt)):
                one_hot_labels_test[i][int(labelBt[i]) - 1] = 1
            pred = sess.run([softmax], feed_dict={x: imageBt, keep_prob: 1.})
            acc = sess.run([accuracy], feed_dict={x: imageBt, y: one_hot_labels_test, keep_prob: 1.})


            acc_list = np.append(acc_list, acc)
            predicted_values = np.append(predicted_values, np.array([np.argmax(pred[0], axis=1).astype(np.float)]))

            ground_truth = np.append(ground_truth, np.array([labelBt]))

        coord.request_stop()
        coord.join(threads)

        new_predicted_values = [x + 1 for x in predicted_values]

        print("Predicted values :")
        print(new_predicted_values)
        print("Ground truth : ")
        print(ground_truth)
        mse = (((new_predicted_values - ground_truth) ** 2).mean(axis=None))
        rmse = np.sqrt(((new_predicted_values - ground_truth) ** 2).mean(axis=None))
        average_acc = sum(acc_list) / 21
        print("Test accuracy is : %f" % (average_acc * 100))
        print("Root mean squared error is: " + str(rmse))
        print("Mean squared error is: " + str(mse))

        return new_predicted_values


a= Regression_criteria_percentage()
print("\n")
b= Regression_Gen_Expressions()
print("\n")
c= Classification()
print("\n")


print("the new")


test= pd.read_csv("../test_6people_new.csv")
l = len(test)
predicted= pd.DataFrame()
predicted["Select_id"]= test["Select_id"]
predicted["Class"]= test["Class"]
predicted["class"]=c[0:l]
predicted["Predicted_Criteria_Pct"]= a[0:l]
predicted["Predicted_Gen_Impression"]= b[0:l]
predicted["Ground_truth_Criteria_Pct"] = test["Criteria_Pct"]
predicted["Ground_truth_Gen_Impression"] = test["Gen_Impression"]

predicted.to_csv(OUTPUT_NAME+"/predicted_Alexnet.csv", encoding='utf-8')



