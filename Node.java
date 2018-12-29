import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        }
        else
            this.type = type;

        if (type == 2 || type == 4)
            parents = new ArrayList<>();
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0)   //If input node
            this.inputValue = inputValue;
    }

    /**
     * Calculate the output of a node.
     * Can get this value by using getOutput()
     */
    public void calculateOutput(Instance instance, ArrayList<Node> outputNodes) {
        if (type == 2 || type == 4) {   //Not an input or bias node
        	if (type == 2) {
        		double weightedSum = 0.0;
        		for (int i = 0; i < parents.size(); i++)
            		weightedSum += parents.get(i).node.getOutput() * parents.get(i).weight;
        		outputValue = Math.max(0, weightedSum);
        	}
        	else {
        		double currentWeightedSum = 0.0, denominator = 0.0;
        		double[] outputWeightedSum = new double[outputNodes.size()];
        		for (int i = 0; i < parents.size(); i++)
        			currentWeightedSum += parents.get(i).node.getOutput() * parents.get(i).weight;
        		for (int i = 0; i < outputNodes.size(); i++)
        			outputWeightedSum[i] = 0.0;
        		for (int i = 0; i < outputNodes.size(); i++) {
        			for (int j = 0; j < outputNodes.get(i).parents.size(); j++)
        				outputWeightedSum[i] += outputNodes.get(i).parents.get(j).node.getOutput() * outputNodes.get(i).parents.get(j).weight;
        			denominator += Math.exp(outputWeightedSum[i]);
        		}
        		outputValue = Math.exp(currentWeightedSum) / denominator; 
        	}
        }
    }

    //Gets the output value
    public double getOutput() {

        if (type == 0)    //Input node
            return inputValue;
        else if (type == 1 || type == 3)    //Bias node
            return 1.00;
        else
            return outputValue;
    }

    //Calculate the delta value of a node.
    public void calculateDelta(Instance instance, ArrayList<Node> outputNodes, ArrayList<Node> hiddenNodes, int index) {
        if (type == 2 || type == 4) {
        	for (int i = 0; i < outputNodes.size(); i++) {
        		if (outputNodes.get(i).getOutput() == 0)
        			outputNodes.get(i).outputGradient = 0.0;
        		else
        			outputNodes.get(i).outputGradient = 1;
        	}
        	for (int i = 0; i < hiddenNodes.size(); i++) {
        		if (hiddenNodes.get(i).getOutput() == 0)
        			hiddenNodes.get(i).outputGradient = 0.0;
        		else
        			hiddenNodes.get(i).outputGradient = 1;
        	}
        	if (type == 2) {
        		double sum = 0.0;
        		for (int k = 0; k < outputNodes.size(); k++)
        			sum += outputNodes.get(k).parents.get(index).weight * (instance.classValues.get(k) - outputNodes.get(k).getOutput()) * outputNodes.get(k).outputGradient;
        		delta = outputGradient * sum;
        	}
        	else
        		delta = (instance.classValues.get(index) - getOutput()) * outputGradient; 
        }
    }

    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4)
        	for (int i = 0; i < parents.size(); i++)
        		parents.get(i).weight += learningRate * parents.get(i).node.getOutput() * delta;
    }
}