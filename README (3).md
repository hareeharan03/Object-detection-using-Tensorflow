
# Fine-grained image classification

Fine-grained Image classification is an object recognition method that aims to identify and classify the models of same category


## Documentation
   NOTE : all the cell can be executed but need to be executed after changing the path since the zip file contains all the data 
1. Open colab and mount drive 

2. mount google drive and upload the dataset zip file

3. open the G14.ipynb file also change your runtime type to GPU

4. !unzip /content/drive/MyDrive/Dataset/CUB_200_2011/images.zip -d /content/drive/MyDrive/Dataset/CUB_200_2011

3. Unzip the  dataset, tensorflow- (Faster R-CNN files) folder and place it in /content/drive/MyDrive/ in colab

    ├── Drive
    ├── MyDrive                 # MyDrive root directory
    │   ├── dataset             # dataset which contains splitted data and all other 
    │   ├── tensorflow1         # tensorflow1 that contains all the inbuilt library and .py files

4. open the G14.ipynb file and run cell by cell and make sure all the directory location are correct 

    note : few cells take longer time to execute in data exploration 

5. while running data augmentation make sure you have executed TRAIN_TEST_SPLIT cell and make different file path if the file path is same as mentioned that will resulted in creating duplicates.

6. Implementation part can be executed from the beggining or even can run the predication part by loading the saved weights for both algorithm

7. In approach 2 tensorflow1 object detection dataset prepration need to be executed after changing filename

8. data augmentation part can executed after changing paths in augmented_images_df = image_aug(df, 'PATH TO INPUT IMAGES FOLDER', 'PATH TO DESTINATION FOLDER', 'aug1_', aug)

9. run all the cell in Implementation of Faster R-CNN to set up the environment for tensorflow object detection dont forget after restarting your runtime after installing numpy will leads to lose all variable 

10. can run the predication part directly using saved weights and we have trained for 100 class due to lack of resource

11. open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the .pbtxt file which will be used for training

13. Then, generate the TFRecord files by issuing these commands from the \object_detection folder
    python generate_tfrecord.py --csv_input=PATH TO TRAINING .CSV FILE --image_dir=images\train --output_path=train.record
    python generate_tfrecord.py --csv_input=PATH TO TESTING .CSV FILE --image_dir=images\test --output_path=test.record

14. Create Label Map and Configure Training
    A.The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the /content/drive/MyDrive/tensorflow1/models/research/object_detection/inference_graph_100_class_aug_final/. (Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map
    B.Navigate to the /content/drive/MyDrive/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/faster_rcnn_inception_v2_pets.config make these changes
        Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 100
        Line 106. Change fine_tune_checkpoint to:
        fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
        Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:
        input_path : "C:/tensorflow1/models/research/object_detection/train.record"
        label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
        Line 130. Change num_examples to the number of images you have in the \images\test directory.
        Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:
        input_path : "C:/tensorflow1/models/research/object_detection/test.record"
        label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

15. train the model !python /content/drive/MyDrive/tensorflow1/models/research/object_detection/train.py --logtostderr --train_dir="PATH FOR TRAINING FOLDER" --pipeline_config_path=/content/drive/MyDrive/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/faster_rcnn_inception_v2_pets.config

16. then use the !python /content/drive/MyDrive/tensorflow1/models/research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /content/drive/MyDrive/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix /content/drive/MyDrive/tensorflow1/models/research/object_detection/100_class_aug_final/model.ckpt-???? --output_directory inference_graph to save the weights with recent trained model

17. finally the saved weights inference graph can be used for predication

18. tensorboard is used for visalization


##THANK YOU