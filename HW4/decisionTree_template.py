
import treeplot
import numpy as np

def loadDataSet(filepath):
    '''
    Returns
    -----------------
    data: 2-D list
        each row is the feature and label of one instance
    featNames: 1-D list
        feature names
    '''
    data=[]
    featNames = None
    fr = open(filepath)
    for (i,line) in enumerate(fr.readlines()):
        array=line.strip().split(',')
        if i == 0:
            featNames = array[:-1]
        else:
            data.append(array)
    return data, featNames


def splitData(dataSet, axis, value):
    '''
    Split the dataset based on the given axis and feature value

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label
    axis: int 
        index of which feature to split on
    value: string
        the feature value to split on

    Returns
    ------------------
    subset: 2-D list 
        the subset of data by selecting the instances that have the given feature value
        and removing the given feature columns
    '''
    subset = []
    for instance in dataSet:
        if instance[axis] == value:    # if contains the given feature value
            reducedVec = instance[:axis] + instance[axis+1:] # remove the given axis
            subset.append(reducedVec)
    return subset


def chooseBestFeature(dataSet):
    '''
    choose best feature to split based on Gini index
    
    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    bestFeatId: int
        index of the best feature
    '''
    #TODO
    #for feature in range(dataSet.shape[1]-1):
    # dictionary to store the best split
    best_split_index = -1
    max_info_gain = -float("inf")
    #dataSet = np.array(dataSet)
    num_features = len(dataSet[0])
    num_sampels = len(dataSet)
    labels = [instance[-1] for instance in dataSet]
    parent_gini = gini_index(labels)
    # loop over all the features and check the information gain 
    for feature_index in range(num_features-1):
        feature_values = [dataSet[i][feature_index] for i in range(num_sampels)]
        sum_child_gini = 0
        for feature in list(set(feature_values)):
            data_after_split = splitData(dataSet, feature_index, feature)
            feature_gini = gini_index([instance[-1] for instance in data_after_split])
            sum_child_gini += (len(data_after_split)/num_sampels)*feature_gini
        curr_info_gain = parent_gini - sum_child_gini
        #print(f"Gain for {feature_index} is | {parent_gini} - {sum_child_gini} = {curr_info_gain}")
        if curr_info_gain>max_info_gain:
            best_split_index = feature_index
            max_info_gain = curr_info_gain

    # return best split
    return best_split_index
    
def gini_index(y):
    ''' function to compute gini index '''
    class_labels = list(set(y))
    gini = 0
    for cls in class_labels:
        p_cls = y.count(cls) / len(y)
        gini += p_cls**2
    return 1 - gini

def stopCriteria(dataSet):
    '''
    Criteria to stop splitting: 
    1) if all the classe labels are the same, then return the class label;
    2) if there are no more features to split, then return the majority label of the subset.

    Parameters
    -----------------
    dataSet: 2-D list
        [n_sampels, m_features + 1]
        the last column is class label

    Returns
    ------------------
    assignedLabel: string
        if satisfying stop criteria, assignedLabel is the assigned class label;
        else, assignedLabel is None 
    '''
    #assignedLabel = None
    # TODO
    classLabels = [instance[-1] for instance in dataSet]

    # Check if all class labels are the same
    #if classLabels.count(classLabels[0]) == len(classLabels):
    if len(set(classLabels)) == 1:
        assignedLabel = classLabels[0]
    # Check if there are no more features to split
    elif len(dataSet[0]) == 1:
        assignedLabel = majorityLabel(classLabels)
    else:
        assignedLabel = None
    return assignedLabel

def majorityLabel(classLabels):
    labelCounts = {}
    for label in classLabels:
        if label in labelCounts:
            labelCounts[label] += 1
        else:
            labelCounts[label] = 1

    majorityLabel = max(labelCounts, key=labelCounts.get)
    return majorityLabel



def buildTree(dataSet, featNames):
    '''
    Build the decision tree

    Parameters
    -----------------
    dataSet: 2-D list
        [n'_sampels, m'_features + 1]
        the last column is class label

    Returns
    ------------------
        myTree: nested dictionary
    '''
    assignedLabel = stopCriteria(dataSet)
    if assignedLabel:
        return assignedLabel
    
    bestFeatId = chooseBestFeature(dataSet)
    bestFeatName = featNames[bestFeatId]
    myTree = {bestFeatName:{}}
    subFeatName = featNames[:]
    del(subFeatName[bestFeatId])
    featValues = [d[bestFeatId] for d in dataSet]
    uniqueVals = list(set(featValues))
    for value in uniqueVals:
        myTree[bestFeatName][value] = buildTree(splitData(dataSet, bestFeatId, value), subFeatName)
    
    return myTree



if __name__ == "__main__":
    data, featNames = loadDataSet('golf.csv')
    dtTree = buildTree(data, featNames)
    # print (dtTree) 
    treeplot.createPlot(dtTree)
    