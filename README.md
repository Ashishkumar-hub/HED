## Holistically-nested edge detection (HED)

Holistically-nested edge detection (HED) is a deep learning model that uses fully convolutional neural networks and deeply-supervised nets to do image-to-image prediction. HED develops rich hierarchical representations automatically (directed by deep supervision on side replies) that are critical for resolving ambiguity in edge and object boundary detection.

##Model Architecture

The model is VGGNet with few modifications-

Side output layer is connected to the last convolutional layer in each stage, respectively conv1_2, conv2_2, conv3_3, conv4_3,conv5_3. The receptive field size of each of these convolutional layers is identical to the corresponding side-output layer.
Last stage of VGGNet is removed including the 5th pooling layer and all the fully connected layers.
The final HED network architecture has 5 stages, with strides 1, 2, 4, 8 and 16, respectively, and with different receptive field sizes, all nested in the VGGNet. 

##Why HED?

The proposed holistically-nested edge detector (HED) tackles two critical issues: 

Holistic image training and prediction, inspired by fully convolutional neural networks  for image-to-image classification (the system takes an image as input, and directly produces the edge map image as output)
 Nested multi-scale feature learning, inspired by deeply-supervised nets that performs deep layer supervision to “guide” early classification results.

### Code for edge detection using pretrained hed model(caffe) using OpenCV

Command to run the edge detection model on video

    python edge.py --input video.mp4 --prototxt deploy.prototxt --caffemodel hed_pretrained_bsds.caffemodel 
    --width 300 --height 300

Command to run the edge detection model on image

    python edge_detector.py --input image.png --prototxt deploy.prototxt --caffemodel hed_pretrained_bsds.caffemodel
    --width 300 --height 300 
