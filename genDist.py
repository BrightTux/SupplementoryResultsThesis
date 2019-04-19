#!/c/Users/Clarence/Miniconda3/python
import time
#import statements
import numpy as np
from dtw import dtw
from fastdtw import fastdtw
import timeit
from sklearn.metrics.pairwise import euclidean_distances
from math import isinf
import math
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from numpy import array, zeros, full, argmin, inf, ndim
from nltk.metrics.distance import edit_distance
from scipy.spatial.distance import directed_hausdorff
import cv2




#list of trajectories for both gt and query [x,y]

#mirror
q1=[(11, 7),(10, 7),(9, 7),(9, 8),(8, 8),(7, 8),(7, 9),(6, 9),(5, 9),(4, 9),(4, 10),(4, 11),(5, 11),(5, 12),(6, 12),(7, 12),(8, 12),(9, 12),(10, 12)]

#reverse direction
q2=[(10, 12),(11, 12),(12, 12),(13, 12),(13, 13),(14, 13),(15, 13),(16, 13),(17, 13),(17, 12),(17, 11),(17, 10),(16, 10),(16, 9),(16, 8),(16, 7),(16, 6),(15, 6),(15, 5)]

#short query
q3=[(17, 11),(17, 12),(17, 13)]

#similar to gt
q4=[(16, 7),(16, 8),(16, 9),(16, 10),(17, 10),(17, 11),(17, 12),(17, 13),(16, 13),(15, 13),(14, 13),(13, 13),(12, 13),(11, 13)]

#u turn/partial match
q5=[(10, 9),(11, 9),(12, 9),(13, 9),(14, 9),(15, 9),(16, 9),(16, 10),(17, 10),(17, 11),(17, 12),(17, 13),(16, 13),(15, 13),(14, 13),(13, 13),(12, 13),(11, 13),(10, 13)]

#long, very different query
q6=[(1, 17),(2, 17),(3, 17),(4, 17),(4, 18),(5, 18),(6, 18),(7, 18),(8, 18),(9, 18),(10, 18),(11, 18),(12, 18),(13, 18),(14, 18),(15, 18),(16, 18),(17, 18),(18, 18),(18, 17),(18, 16),(18, 15),(17, 15),(17, 14),(16, 14),(15, 13),(14, 13),(13, 13),(12, 13),(11, 13),(11, 12),(10, 12),(9, 12),(8, 12),(7, 12),(6, 12),(5, 12),(5, 11),(4, 11),(4, 10),(5, 10),(5, 9),(6, 9),(7, 9),(8, 9),(9, 9),(10, 9),(11, 9),(12, 9),(13, 9),(13, 10),(14, 10),(15, 10),(16, 10),(16, 9),(16, 8),(15, 8),(14, 8),(13, 8),(12, 8),(11, 8),(11, 7),(12, 7),(13, 7),(13, 6),(14, 6),(15, 6)]

#similar shape
q7=[(15, 6),(15, 7),(16, 7),(16, 8),(16, 9),(16, 10),(15, 10),(14, 10),(13, 10),(13, 9),(12, 9)]

queries = [q1,q2,q3,q4,q5,q6,q7]
# ----------------------------------------------- E N D   O F   Q U E R I E S -------------------------------------------------------------

gt1=[(16, 7),(16, 8),(16, 9),(16, 10),(16, 11),(17, 11),(17, 12),(17, 13),(16, 13),(15, 13),(14, 13),(13, 13),(12, 13),(12, 12),(11, 12),(10, 12)]


groundtruths = [gt1]
# ----------------------------------------------- E N D   O F   G R O U N D T R U T H S -------------------------------------------------------------

def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
    return sig


def emdFunction(gt,q):
    for indgt, valgt in enumerate(gt):
        for indq, valq in enumerate(q):


            x = img_to_sig(np.array(valgt))
            y = img_to_sig(np.array(valq))

            #dist, _, flow = cv2.EMD(x, y, cv2.DIST_L2)
            distl1, _, flow = cv2.EMD(x,y,cv2.DIST_L1)
            distl2, _, flow = cv2.EMD(x,y,cv2.DIST_L2)
            distc, _, flow = cv2.EMD(x,y,cv2.DIST_C)


            print("emd Dist: ", distl1, distl2, distc)

def dtwFunction(gt,q, dist_fun):

    for indgt, valgt in enumerate(gt):
        for indq, valq in enumerate(q):
            w = inf
            s = 1.0

            x = valgt
            y = valq
            x = img_to_sig(np.array(valgt))
            y = img_to_sig(np.array(valq))

            #dist_fun = edit_distance

            dist, cost, acc, path = dtw(x, y, dist_fun, w=w, s=s)

            print("dtw Dist, using edit_distance : ", 1 - dist)

def fastdtwFunction(gt,q):
    for indgt, valgt in enumerate(gt):
        for indq, valq in enumerate(q):

            x = valgt
            y = valq
            x = img_to_sig(np.array(valgt))
            y = img_to_sig(np.array(valq))

            #dist_fun = edit_distance
            distance, path = fastdtw(x, y, dist=euclidean)


            print("fast dtw Dist, using euclidean distance : ", distance)


def hausFunction(gt,q):

    for indgt, valgt in enumerate(gt):
        for indq, valq in enumerate(q):
            x = valgt
            y = valq

            #print("hausdorff distance: ", directed_hausdorff(x, y)[0])
            #print("hausdorff distance: ", directed_hausdorff(y, x)[0])
            #print("---")

            print("hausdorff distance: ", max((directed_hausdorff(x, y)[0]),(directed_hausdorff(y,x)[0])))


def euclideanFunction(gt,q):
    x = []
    y = []
    for indgt, valgt in enumerate(gt):
        for indq, valq in enumerate(q):
            x = valgt
            y = valq

            while (len(x[indgt]) >len(y[indq])):
                padzero = (0,0)
                y.append(padzero)

            while(len(y[indq])>len(x[indgt])):
                padzero=(0,0)
                x.append(padzero)

            #print(x,len(x))
            #print(y,len(y))
            eucl = math.sqrt(pow((x[indgt][0]-y[indq][0]),2) + pow((x[indgt][1]-y[indq][1]),2))


            print("backpadding euclidean distance: ", eucl)



def euclideanFunction2(gt,q):
    x = []
    y = []
    for indgt, valgt in enumerate(gt):
        for indq, valq in enumerate(q):
            x = valgt
            y = valq

            while (len(x[indgt]) >len(y[indq])):
                padzero = (0,0)
                y.insert(0,padzero)

            while(len(y[indq])>len(x[indgt])):
                padzero=(0,0)
                x.insert(0,padzero)

            print(x,len(x))
            print(y,len(y))
            eucl = math.sqrt(pow((x[indgt][0]-y[indq][0]),2) + pow((x[indgt][1]-y[indq][1]),2))


            print("front padding euclidean distance: ", eucl)



def chamferDistance2(gt, q, alpha):

    final_cham=[]

    for indq, val in enumerate(q):

        min_x =[]
        min_y = []
        min_t = []
        tsum=[]
        currentIndexOfMin = 0
        maxDifference = 0


        if(len(gt[0]) >= len(q[indq])):
            outerloop = len(gt[0])
            innerloop = len(q[indq])
            outer = gt[0]
            inner = q[indq]
        else:
            outerloop = len(q[indq])
            innerloop = len(gt[0])
            outer = q[indq]
            inner = gt[0]

        for oidx, o in enumerate(outer):
            for iidx, i in enumerate(inner):

                #print(o, i)

                min_x.append(abs(o[0] - i[0]))
                min_y.append(abs(o[1] - i[1]))
                min_t.append(abs(oidx - iidx))

            for k, val in enumerate(min_x):
                tsum.append(min_x[k] + min_y[k] + min_t[k])

            #indexOfMin = min(xrange(len(tsum)), key=tsum.__getitem__)
            new_tsum=np.array(tsum)
            indexOfMin = np.where(new_tsum==new_tsum.min())

            if(currentIndexOfMin == 0):
                ModifiedMinValue = new_tsum.min()

            maxDifference = maxDifference + ModifiedMinValue

            min_x = []
            min_y = []
            min_t = []
            tsum = []


        cDistance = round(((pow(maxDifference,2))/outerloop)*100.0)/100.0


        curveAdjuster = alpha
        similarityScore = round((1.0-((cDistance)/(cDistance + curveAdjuster)))*100.0)

    #print("similarity score: ", similarityScore)
    #print("cDistance: ", cDistance)
    #time.sleep( 2 )

    #print('***************************************')
       # print("Similarity score: ", similarityScore)
       # print("cDistance:        ", cDistance)
        final_cham.append([similarityScore, cDistance])

    return final_cham

def chamfer_distance_numpy(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (av_dist1+av_dist2)/batch_size

    return dist

def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """

    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1),
                (1, num_point, 1)),
                (-1, num_features))

    distances = LA.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances





chamf = []
eucl = []
earthm = []
hausdrorff = []

#start = timeit.default_timer()
#print("numpy ChamferDistance:", chamfer_distance_numpy(gt1, q1))
#stop = timeit.default_timer()
#print('Time: ', stop - start)
print("300 Similarity Score, ChamferDistance:", chamferDistance2(groundtruths, queries, 300))

dtwFunction(groundtruths, queries, edit_distance)
fastdtwFunction(groundtruths, queries)
hausFunction(groundtruths, queries)
emdFunction(groundtruths, queries)
euclideanFunction(groundtruths, queries)
#euclideanFunction2(groundtruths, queries)
#https://stackoverflow.com/questions/30706079/hausdorff-distance-between-3d-grids





