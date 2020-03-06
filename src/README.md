## Requirements
 - Python 3
 - Cython
 - networkx
 - numpy
 - scikit-learn
 - scikit-image
 - PIL (pillow)
 - keras (keras-gpu)
 - pandas
  
## Installation

- Install GMatch4py by running
``` sh
>> cd GMatch4py
>> python setup.py install
```
- Setup PYTHONPATH by running
``` sh
>> . ./set_python_path.sh
```

## Workflow 

The workflow for preparing the data, and training and evaluating the digit 
recognizers used in the paper is described below.
This workflow assumes that the used character datasets for trainig, 
validation and testing are formatted using keras' folder structure in which all
samples of the same class are in the same folder as shown below:

```
train_50
    +---- 0
          +---- 0.png
          ...
    +---- 1
          +---- 0.png
          ...
    +---- 2
    +---- 3
    +---- 4
    +---- 5
    +---- 6
    +---- 7
    +---- 8
    +---- 9
```

The classes should be numbered from 0 to 9 and the scripts assume that there are
only 10 classes.

### Extract Graphs

Graphs are extracted from all images in a given dataset folder structure using 
the `extract_graphs.py` script. Assuming you want to extract the graphs from
all images in a folder `../train_50` into a folder `../train_50_graphs`, this
can be done as follows:

``` sh
>> mkdir ../train_50_graphs
>> python3 tools/extract_graphs.py ../train_50 ../train_50_graphs
```

### Compute Graph Edit Distances

As a next step, the distances between the extracted graphs need to be computed.
For training the Siamese network to predict the graph edit distance, all 
pairwise distances between the graphs of the provided training set need to be
computed. For evaluating the performance of the graph edit distance for 
classifying character graphs, the pairwise distances between the graphs of the
training set and the graphs of the testing set need to be computed.

#### Pairwise Distances for Training

The pairwise distances for training the Siamese network are comptued using the
`compute_graph_distances_parallel.py` script. Assuming you want to compute the
pairwise distances for the extracted graphs in `../train_50_graphs` and store
those distances in the file `../train_50_graph_distances.npz`, this can be done
as follows:

``` sh
>> python3 compute_graph_distances_parallel.py ../train_50_graphs ../train_50_graph_distances.npz
```

#### Pairwise Distances for Evaluation

The pairwise distances between a set of training graphs `../train_50_graphs` and
a set of test graphs `../test_100_graphs` is computed using the 
`compute_graph_distances_parallel_train_test.py` script as follows:

``` sh
>> python3 compute_graph_distances_parallel_train_test.py ../train_50_graphs ../test_100_graphs ../train_test_graph_distances.npz
```

### Train Siamese Network

The Siamese network is trained using the `train_graph_edit_distance.py` script.
Assuming you want to train the network using the training images in 
`../train_50` and the validation images in `../valid_100` with their respective
graph edit distance files `../train_50_graph_distances.npz` and `../valid_100_graph_edit_distances.npz`, this can be done as follows:

``` sh
>> mkdir log
>> python3 train_graph_edit_distance.py demo ../train_50_graph_distances.npz ../train_50 ../valid_100_graph_edit_distances.npz ../valid_100 ../log
```

Here, `demo` is the name of this run, which will be used when creating names for
log and model files, i.e., `graphtrain_demo.csv` and `graphtrain_demo.h5`. These
files will be stored in the provided log folder `../log`.

### Train Digit Recognizer

The digit or character recognizers in the paper are trained either from scratch
or use the pre-trained Siamese network for training. In the following, scripts
for both tasks are described.

#### Train Digit Recognizer from Scratch

The baseline recognizer is trained from scratch using the 
`train_digit_recognizer.py` script.
Assuming the training images are in `../train_50` and the validation images
are in `../valid_100`, the recognizer is trained as follows:

``` sh
>> python3 tools/train_digit_recognizer.py demo ../train_50 ../valid_100 ../log
```

Here, `demo` is the name of this run, which will be used when creating names for
log and model files, i.e., `demo.csv` and `demo.h5`. These files will be stored
in the provided log folder `../log`.

#### Train Digit Recognizer from Pre-trained Network

The recognizers using the pre-trained Siamese network are trained using the
`train_digit_recognizer_pretrained.py` script. Assuming the pretrained network 
is stored in `../log/graphtrain_demo.h5`, the training images are in 
`../train_50`, the validation images are in `../valid_100`, and the weights of 
the CNN based should be fixed, then the recognizer is trained as follows:

``` sh
>> python3 tools/train_digit_recognizer_pretrained.py demo ../log/graphtrain_demo.h5 ../train_50 ../valid_100 ../log
```

Here, `demo` is the name of this run, which will be used when creating names for
log and model files, i.e., `digit_demo.csv` and `digit_demo.h5`. These files 
will be stored in the provided log folder `../log`.

On the other hand, if all weights should be trained the recognizer is trained as
follows:

``` sh
>> python3 tools/train_digit_recognizer_pretrained.py --trainable demo_trainable ../log/graphtrain_demo.h5 ../train_50 ../valid_100 ../log
```

### Evaluate Approaches

After preparing the data and training the recognizers, the recognition 
performance is evaluated for the graph edit distance based approach, the 
approach based on the feature vectors extracted by the CNN base of the 
Siamese network, and the trained digit recognizers.

#### Graph Edit Distance Based Recognition Performance

The recognition accuracy for using the computed graph edit distance is assessed
using the `evaluate_ged_knn.py` script. Assuming the training images for the 
evaluation are in `../train_50` and the pairwise distances between train and 
test set are in `../train_test_graph_distances.npz`, the evaluation can be run 
as follows:

``` sh
>> python3 tools/evaluate_ged_knn.py demo ../train_50 ../train_test_graph_distances.npz
```

The evaluation result will be printed out to standard out for an accuracy of 77%
as follows:

```
demo,0.77
```

#### Siamese CNN Base Based Recognition Performance

The recognition accuracy for using the CNN base of the Siamese network is 
assessed using the `evaluate_sged_knn.py` script. Assuming the training 
images are in `../train_50`, the test images are in `../test_100`, and
the pre-trained network is stored in `../log/graphtrain_demo.h5`, the evaluation
can be run as follows:

``` sh
>> python3 tools/evaluate_sged_knn.py demo ../train_50 ../test_100 ../log/graphtrain_demo.h5
```

The evaluation result will be printed out to standard out for an accuracy of 77%
as follows:

```
demo,0.77
```

#### Trained Digit Recognizer Recognition Performance

The recognition accuracy of the trained digit recognizers is assessed using the
`evaluate_digit_recognizer.py` script. Assuming the test images are in 
`../test_100` and the trained recognizer model is stored in 
`../log/digit_demo.h5`, the evaluation can be run as follows:

``` sh
>> python3 tools/evaluate_digit_recognizer.py demo ../test_100 ../log/digit_demo.h5
```

The evaluation result will be printed out to standard out for an accuracy of 77%
as follows:

```
demo,0.77
```


