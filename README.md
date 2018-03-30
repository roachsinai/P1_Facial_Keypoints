[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applicatoions include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project will be broken up into a few main parts in three Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) will be graded**:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and the Trained CNN from Part 2


## Project Instructions

All of the starting code and resources you'll need to compete this project are in this Github repository. Before you can get started coding, you'll have to make sure that you have all the libraries and dependencies required to support this project.

*Note that this project does not require the use of GPU, so this repo does not include instructions for GPU setup.*


### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/cezannec/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `ai` with Python 3.6 and the `numpy` and `pandas` packages for data loading and transformation. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n ai python=3.6 numpy pandas
	source activate ai
	```
	- __Windows__: 
	```
	conda create --name ai python=3.6 numpy pandas
	activate ai
	```
	
	At this point your command line should look something like: `(ai) <User>:P1_Facial_Keypoints <user>$`. The `(ai)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the P1_Facial_Keypoints repo, in the subdirectory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data, and you're encouraged to look trough these folders on your own, too.


## Notebooks

1. Navigate back to the repo. (Also, your source environment should still be activated at this point)
```shell
cd
cd P1_Facial_Keypoints
```

2. Open the first notebook and follow the instructions.


__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality and answer all of the questions included in the notebook. __Unless requested, it's suggested that you do not modify code that has already been included.__


## Evaluation

Your project will be reviewed against the project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


## Project Submission

When you are ready to submit your project, collect all of your project files -- all executed notebooks, and python files -- and compress them into a single zip archive for upload.

Alternatively, your submission could consist of only the **GitHub link** to your repository with all of the completed files.

<a id='rubric'></a>
## Project Rubric

### `models.py`

#### Specify the CNN architecture
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Define a CNN in `models.py`. |  Define a convolutional neural network with at least one convolutional layer, i.e. self.conv1 = nn.Conv2d(1, 32, 5). The network should take in a grayscale, square image. |


### Notebook 2

#### Define the loss and optimization functions
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  Select a loss function and optimizer for training the model. |  The loss and optimization functions should be appropriate for keypoint detection, which is a regression problem. |


#### Train the CNN

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Train your model.  |  Train your CNN after defining its loss and optimization functions. You are encouraged, but not required, to visualize the loss over time/epochs by printing it out occasionally and/or plotting the loss over time. Save your best trained model. |


#### Answer questions about model architecture

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| All questions about model and loss choices are answered.  | After training, all questions in notebook 2 about model architecture and choice of loss function are answered. |


#### Visualize one or more learned feature maps

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Apply a learned convolutional kernel to an image and see its effects. |  Your CNN "learns" to recognize features and this step requires that you extract at least one convolutional filter fro the trained model, apply it to an image, and see what effect this filter has on the image. |


#### Answer question about feature visualization
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
|  After visualizing a convolution kernel, applied to an image, answer: what do you think it detects? | This answer should be informed by how the filtered image (from the step above) looks. |



### Notebook 3

#### Detect faces in a given image
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Use a haar cascade face detector to detect faces in a given image. | The submission successfully employs OpenCV's face detection. |

#### Process each image of a face so that it can be input into your trained model
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Turn each detected image of a face into an appropriate input Tensor. | You should transform ay face into a suqare grayscale image and then a Tensor for your model to take it as input. |

#### Complete the pipeline
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Predict and display the keypoints on each detected face. | After face detection with a Haar cascade and face pre-processing, apply your trained model to each detected face, and display the predicted keypoints on each face in the image. |



__For testers:__

Please answer the following questions, as I'm aiming to improve and add to this project:

* How could this project be improved?
* Where were the instructions most confusing?

* Would you like the ability to overlay simple graphics on a face -- i.e sunglasses on detected eye keypoints?
* Would you like to learn how to group faces by pose (facing left/right/etc) using unsupervised clustering like k-means?

Thank you so much for your time!
