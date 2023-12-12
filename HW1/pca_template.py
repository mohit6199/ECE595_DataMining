
# coding: utf-8
from numpy import *
from matplotlib import pyplot as plt
import sys


def loadDataSet(fileName = 'iris_with_cluster.csv'):
    dataMat=[]
    labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArray=line.strip().split(',')
        records = []
        for attr in lineArray[:-1]:
            records.append(float(attr))
        dataMat.append(records)
        labelMat.append(int(lineArray[-1]))
    dataMat = array(dataMat)
    
    labelMat = array(labelMat)
    
    
    return dataMat,labelMat

def pca(dataMat, PC_num=2):
    '''
    Input:
        dataMat: obtained from the loadDataSet function, each row represents an observation
                 and each column represents an attribute
        PC_num:  The number of desired dimensions after applyting PCA. In this project keep it to 2.
    Output:
        lowDDataMat: the 2-d data after PCA transformation
    '''
    Xd = zeros_like(dataMat)
    for i in range(dataMat.shape[1]):
        Xd[:,i] = dataMat[:,i] - mean(dataMat[:,i])

    S = (1/(dataMat.shape[0]-1))*Xd.T@Xd

    eigenvalues, eigenvectors = linalg.eig(S)
    eigenvalues_argsort = flip(argsort(eigenvalues))
    eigenvectors_required = eigenvectors[:,eigenvalues_argsort[0:PC_num]]

    lowDDataMat = Xd@eigenvectors_required

    return array(lowDDataMat)


def plot(lowDDataMat, labelMat, figname:None):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''
    arr = append(lowDDataMat, array([labelMat]).T,axis=1)
    for label in unique(arr[:,2]):
        plt.scatter(arr[arr[:,2]==label][:,0],arr[arr[:,2]==label][:,1])
        
    if figname:
        plt.savefig(figname)
        plt.title(figname[:figname.index(".")]+" Dataset")
    else:
        plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = 'iris_with_cluster.csv'
    figname = filename
    figname = figname.replace('csv','jpg')
    dataMat, labelMat = loadDataSet(filename)
    
    lowDDataMat = pca(dataMat)
    
    plot(lowDDataMat, labelMat, figname)
    

