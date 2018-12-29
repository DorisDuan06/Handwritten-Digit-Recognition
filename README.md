# Handwritten-Digit-Recognition
This project builds a 2-layer, feed-forward neural network and trains it using the back-propagation algorithm. The neural network handles a multi-class classification problem for recognizing images of handwritten digits. All inputs to the neural network are numeric. The neural network has one hidden layer. The network is fully connected between consecutive layers, meaning each unit (node) in the input layer is connected to all nodes in the hidden layer, and each node in the hidden layer is connected to all nodes in the output layer. Each node in the hidden layer and the output layer will also have an extra input from a “bias node" that has constant value +1. So, we can consider both the input layer and the hidden layer as containing one additional node called a bias node. All nodes in the hidden layer (except for the bias node) use the ReLU activation function, while all the nodes in the output layer use the Softmax activation function. The initial weights of the network are set randomly based on an input random seed. Assuming that input examples (called instances in the code) have _m_ attributes (hence there are _m_ input nodes, not counting the bias node) and we want _h_ nodes (not counting the bias node) in the hidden layer, and _o_ nodes in the output layer, then the total number of weights in the network is _(m+1)h_ between the input and hidden layers, and _(h+1)o_ connecting the hidden and output layers. The number of nodes to be used in the hidden layer will be given as input.

Important methods in the classes `NNImpl` and `Node`:
```Java
public class Node{
    public void calculateOutput()
    public void calculateDelta()
    public void updateWeight(double learningRate)
}

public class NNImpl{
    public int predict(Instance inst);
    public void train();
    private double loss(Instance inst);
}
```
```Java
void calculateOutput():
```
calculates the output at the current node and stores that value in a member variable called `outputValue`
```Java
void calculateOutput():
```
calculates the delta value, Δ, at the current node and stores that value in a member variable called `delta`
```Java
void updateWeight(double learningRate):
```
updates the weights between parent nodes and the current node using the provided learning rate
```Java
int predict (Instance inst):
```
calculates the output (i.e., the index of the class) from the neural network for a given example
```Java
void train():
```
trains the neural network using a training set, fixed learning rate, and number of epochs (provided as input to the program). This function also prints the total Cross-Entropy loss on all the training examples after each epoch.
```Java
double loss(Instance inst):
```
calculates the Cross-Entropy loss from the neural network for a single instance. This function will be used by `train()`

## Dataset
The dataset we used is called Semeion (https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit). It contains 1,593 binary images of size 16 x 16 that each contain one handwritten digit. This program classifies each example image as one of the three possible digits: 6, 8 or 9. If desired, you can view an image using the supplied python code called `view.py`. Usage is described at the beginning of this file.

Each dataset will begin with a header that describes the dataset: First, there may be several lines starting with `//` that provide a description and comments about the dataset. The line starting with `**` lists the digits. The line starting with `##` lists the number of attributes, i.e., the number of input values in each instance (in our case, the number of pixels). You can assume that the number of classes will _always_ be 3 for this project because this project only considers 3-class classification problems. The first output node should output a large value when the instance is determined to be in class 1 (here meaning it is digit 6). The second output node should output a large value when the instance is in class 2 (i.e., digit 8) and, similarly, the third output node corresponds to class 3 (i.e., digit 9). Following these header lines, there will be one line for each instance, containing the values of each attribute followed by the target/teacher values for each output node. For example, if the last 3 values for an instance are: 0 0 1 then this means the instance is the digit 9.

## Implementation Details 
This project has four assistant classes, called `Instance`, `Node`, `NeuralNetworkImpl` and `NodeWeightPair`. Their data members and methods are commented in the code. An overview of these classes is given next.

The `Instance` class has two data members: `ArrayList<Double> attributes` and `ArrayList<Integer> classValues`. It is used to represent one instance (aka example) as the name suggests. `attributes` is a list of all the attributes (in our case binary pixel values) of that instance (all of them are `double`) and `classValues` is the class (e.g., 1 0 0 for digit 6) for that instance.

The most important data member of the `Node` class is `int type`. It can take the values 0, 1, 2, 3 or 4. Each value represents a particular type of node. The meanings of these values are:

    0: an input node 
    1: a bias node that is connected to all hidden layer nodes 
    2: a hidden layer node 
    3: a bias node that is connected to all output layer nodes 
    4: an output layer node

 `Node` also has a data member `double inputValue` that is only relevant if the type is 0. Its value can be updated using the method `void setInput(double inputValue)`. It also has a data member `ArrayList<NodeWeightPair> parents`, which is a list of all the
nodes that are connected to this node from the previous layer (along with the weight connecting these two nodes). This data member is relevant only for types 2 and 4. The neural network is fully connected, which means that all nodes in the input layer (including the bias node) are connected to each node in the hidden layer and, similarly, all nodes in the hidden layer (including the bias node) are connected to the node in the output layer. The output of each node in the output layer is stored in `double outputValue`. It can be accessed using the method `double getOutput()`. Method  `void calculateOutput()` calculates the output activation value at the node if it’s of type 2 or 4. The calculated output is stored in outputValue. The value is determined by the definition of the activation function (ReLU or Softmax), which depends on the type of node (type 2 means ReLU, type 4 means Softmax). Definitions of the ReLU and Softmax activation functions are given at https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions as well as in the Notes section below.

`NodeWeightPair` has two data members, `Node` and `weight`. These should be self-explanatory. `NNImpl` is the class that maintains the whole neural network. It maintains lists of all input nodes (`ArrayList<Node> inputNodes`), hidden layer nodes (`ArrayList<Node> hiddenNodes`), and output layer nodes (`ArrayList<Node> outputNodes`). The _last_ node in both the input layer and the hidden layer is the bias node for that layer. Its constructor creates the whole network and maintains the links. To train the network, the project implemented the back-propagation algorithm. It updates all the weights in the network after each instance, so it does a form of Stochastic Gradient Descent for training. It uses Cross-Entropy loss as the loss function at the output nodes for the backpropagation algorithm. Details about Cross-Entropy loss are available at http://mlcheatsheet.readthedocs.io/en/latest/loss_functions.html Finally, it changes the input values of each input layer node (except the bias node) when using each new training instance to train the network.

## Classification
Based on the outputs of the output nodes, `int predict(Instance inst)` classifies the instance as the index of the output node with the _maximum_ value. For example, if one instance has outputs [0.1, 0.3, 0.6], this instance will be classified as digit 9.

## Testing
The format of testing commands will be:

    java DigitClassifier <numHidden> <learnRate> <maxEpoch> <randomSeed> <trainFile> <testFile>

where `trainFile`, and `testFile` are the names of training and testing datasets, respectively. `numHidden` specifies the number of nodes in the hidden layer (excluding the bias node at the hidden layer). `learnRate` and `maxEpoch` are the learning rate and the number of epochs that the network will be trained, respectively. `randomSeed` is used to seed the random number generator that initializes the weights and shuffles the training set. To facilitate debugging, there provides you with sample training data and testing data in the files `train1.txt` and `test1.txt`. A sample test command is

    java DigitClassifier 5 0.01 100 1 train1.txt test1.txt

The class `DigitClassifier` will load the data.
