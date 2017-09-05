# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from datetime import datetime
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES']='0'

batch_size = 128
num_epochs = 20
num_preprocess_threads = 1
min_queue_examples = 1
dropout_rate =1
learning_rate=1e-4
num_classes=10
TRAINING = 0 # 0 for test

selectedType = '1'

args = sys.argv
if len(args) > 1:
    selectedType = args[1]    

if selectedType == '1':
    filewriter_path = './logs/Percentage_original_data_frames'
    checkpoint_path = './checkpoints/Percentage_original_data_frames'
    OUTPUT_NAME = 'original_data_frames'
    DATA_FOLDER = '../original_data_frames'

elif selectedType == '2':
    filewriter_path = './logs/Percentage_canny_data_frames'
    checkpoint_path = './checkpoints/Percentage_canny_data_frames'
    OUTPUT_NAME = 'canny_data_frames'
    DATA_FOLDER = '../canny_data_frames'
    
elif selectedType == '3':
    filewriter_path = './logs/Percentage_enhanced_data_frames'
    checkpoint_path = './checkpoints/Percentage_enhanced_data_frames'    
    OUTPUT_NAME = 'enhanced_data_frames'
    DATA_FOLDER = '../enhanced_data_frames'

elif selectedType == '4':
    filewriter_path = './logs/Percentage_segmentation_data_frames'
    checkpoint_path = './checkpoints/Percentage_segmentation_data_frames'
    OUTPUT_NAME = 'segmentation_data_frames'
    DATA_FOLDER = '../segmentation_data_frames'

elif selectedType == '5':
    filewriter_path = './logs/Percentage_segmentation_contour_data_frames'
    checkpoint_path= './checkpoints/Percentage_segmentation_contour_data_frames'
    OUTPUT_NAME = 'segmentation_contour_data_frames'
    DATA_FOLDER = '../segmentation_contour_data_frames'


# # Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
   os.mkdir(checkpoint_path)

display_step=1

def getImageLabel(path):
    filename_queue = tf.train.string_input_producer([path])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    record_defaults = [["no.jpg"], [1], [1.], [1.], [1.]]
    Img_Path, Class_Category, Criteria_Avg, Label, Gen_Impression = tf.decode_csv(value,
                                                                                  record_defaults=record_defaults)

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
                                      capacity=min_queue_examples + 3 * batch_size)

    #image_b, label_b = tf.train.shuffle_batch([img, Label],
                                      #batch_size=batch_size,
                                      #num_threads=num_preprocess_threads,
                                      #capacity=min_queue_examples + 3 * batch_size,min_after_dequeue= 1)

    label_b = tf.expand_dims(label_b, 1)

    return image_b, label_b


if (TRAINING==1):

    image_b, label_b = getImageLabel("../train_Labels366_new.csv")

    # num_training_samples = 12913
    num_training_samples = 14800

    train_batches_per_epoch = np.ceil(num_training_samples / batch_size).astype(np.int32)

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batch_size,1])
    keep_prob = tf.placeholder(tf.float32)
    train_layers = ['fc8', 'fc7']
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]


    # Initialize model
    model = AlexNet(x, keep_prob, num_classes,train_layers)

    score = tf.layers.dense(model.fc8, 1)

    with tf.name_scope("mean_sq"):
        loss= tf.losses.mean_squared_error(y, score)

    with tf.name_scope("train"):
        # Create optimizer and apply gradient descent to the trainable variables
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Add the loss to summary
    tf.summary.scalar('Sq Error Loss', loss)


    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)
    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)
#
    # Initialize an saver for store model checkpoints
#

elif(TRAINING==0):
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    #Initialize model
    model = AlexNet(x, keep_prob, num_classes,[])
    #
    # Link variable to model output
    score = model.fc8
    score = tf.layers.dense(model.fc8, 1)

saver = tf.train.Saver()

writer = tf.summary.FileWriter(filewriter_path)

with tf.Session() as sess:

    if(TRAINING==1):
        # Required to get the filename matching to run.

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        model.load_initial_weights(sess)
        # Coordinate the loading of image files.

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))

        for epoch in range(0,num_epochs):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

            for step in range(0, train_batches_per_epoch):
                print("{} Batch number: {}".format(datetime.now(), step + 1))

                # Get an image tensor and print its value.
                imageB, labelB = sess.run([image_b, label_b])
                np.reshape(labelB,(batch_size,1))


                _, trainloss = sess.run([train_op, loss], feed_dict={x: imageB,
                                                 y: labelB,
                                                 keep_prob: dropout_rate})
                print(trainloss)

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: imageB,
                                                 y: labelB,
                                                 keep_prob: dropout_rate})
                    writer.add_summary(s, epoch * train_batches_per_epoch + step)
                    # writer.add_summary(image_summary)


            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

    elif(TRAINING==0):

        image_bt, label_bt = getImageLabel("../test_Labels366_new.csv")

        print('inside Evaluation')
        #tf.local_variables_initializer()
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # define activation of last layer as score

        saver.restore(sess, checkpoint_path+"/model_epoch20.ckpt")

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
