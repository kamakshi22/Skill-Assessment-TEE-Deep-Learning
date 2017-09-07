# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np
from VGG import Vgg16
from datetime import datetime
import os


batch_size = 64
num_epochs = 60
num_preprocess_threads = 1
min_queue_examples = 1
dropout_rate =1
learning_rate=1e-6
num_classes=10
TRAINING = 0# 0 for test

DATA_FOLDER = '../original_data_frames'
filewriter_path = "./logs/Classification_original_data_frames"
checkpoint_path = "./checkpoints/Classification_original_data_frames/"

# # Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
   os.mkdir(checkpoint_path)
# #
display_step=1

def getImageLabel(path):
    filename_queue = tf.train.string_input_producer([path])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    record_defaults = [["no.jpg"], [1], [1.], [1.], [1.]]
    Img_Path, Label, Criteria_Avg, Criteria_Per, Gen_Impression = tf.decode_csv(value,
                                                                                     record_defaults=record_defaults)

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


if (TRAINING==1):


    image_b, label_b = getImageLabel("../train_Labels366_new.csv")

    num_training_samples = 13485
    #num_training_samples = 14800


    train_batches_per_epoch = np.ceil(num_training_samples / batch_size).astype(np.int32)
#
    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None,10])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = Vgg16()
    model.build(x)

    score = model.fc8
    score= model.prob
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

    with tf.name_scope("train"):
        # Create optimizer and apply gradient descent to the trainable variables
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)


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
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope('keepprob'):
        keep_prob = tf.placeholder(tf.float32)


    #Initialize model
    model = Vgg16()
    model.build(x)
    #
    # Link variable to model output
    score = model.fc8
    # score = tf.layers.dense(model.fc8, 1)
    softmax = tf.nn.softmax(score)

    correct_pred = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
saver = tf.train.Saver()
#
writer = tf.summary.FileWriter(filewriter_path)
#
#
with tf.Session() as sess:

    if(TRAINING==1):
        # Required to get the filename matching to run.

        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

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
                one_hot_labels = np.zeros((batch_size, num_classes))
                for i in range(len(labelB)):
                    one_hot_labels[i][int(labelB[i])-1] = 1


                _, trainloss = sess.run([train_op, loss], feed_dict={x: imageB,
                                                 y: one_hot_labels,
                                                 keep_prob: dropout_rate})

                print(trainloss)

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: imageB,
                                                 y: one_hot_labels,
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

        image_bt, label_bt = getImageLabel("../test_6people.csv")



        print('inside Evaluation')
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess, "./checkpoints/Classification_original_data_frames/model_epoch40.ckpt")

        predicted_values= []
        ground_truth=[]
        acc_list=[]

        for step in range(0, 41):


            imageBt, labelBt = sess.run([image_bt, label_bt])
            one_hot_labels_test = np.zeros((batch_size, num_classes))
            for i in range(len(labelBt)):
                one_hot_labels_test[i][int(labelBt[i]) - 1] = 1

            pred= sess.run([softmax], feed_dict={x: imageBt, keep_prob: 1.})
            acc = sess.run([accuracy], feed_dict={x: imageBt, y: one_hot_labels_test, keep_prob: 1.})

            # predf= np.clip(pred,0,100)
            acc_list= np.append(acc_list, acc)
            predicted_values= np.append(predicted_values, np.array([np.argmax(pred[0],axis=1).astype(np.float)]))

            ground_truth= np.append(ground_truth, np.array([labelBt]))


        coord.request_stop()
        coord.join(threads)

        new_predicted_values = [x + 1 for x in predicted_values]

        print("Predicted values :")
        print(new_predicted_values)
        print("Ground truth : ")
        print(ground_truth)
        mse = (((new_predicted_values - ground_truth) ** 2).mean(axis=None))
        rmse = np.sqrt(((new_predicted_values - ground_truth) ** 2).mean(axis=None))
        average_acc = sum(acc_list) / 41
        print("Test accuracy is : %f" % (average_acc * 100))
        print("Root mean squared error is: " + str(rmse))
        print("Mean squared error is: " + str(mse))

