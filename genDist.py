import time
#import statements
import numpy as np
from dtw import dtw

x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
print(x)

#list of trajectories for both gt and query [x,y]

q1=[(16,7),(16,8),(16,9),(17,10),(17,11),(17,12),(17,13),(16,13),(15,13),(14,13),(13,13),(13,12),(12,12),(11,12),(10,11),(9,11)]

q2=[(17,13),(16,13),(15,13),(14,12),(13,11),(12,12),(11,12),(10,12),(9,12),(8,12),(7,12),(6,12),(5,12),(4,11),(3,11)]

q3=[(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12),(16,13),(15,13),(14,13),(13,13),(13,12),(11,12),(10,12),(10,11),(9,11),(8,11),(7,11),(6,11),(5,11),(5,12),(5,13)]

q4=[(15,4),(15,5),(15,6),(14,6),(13,6),(12,6),(11,6),(11,7),(10,7),(9,7),(9,8),(8,8),(7,8),(7,9),(6,9),(5,10),(4,10),(3,10),(2,10),(1,11),(0,11)]

q5=[(16,6),(16,7),(16,8),(17,9),(17,10),(17,11),(17,12),(17,13),(17,14),(17,15),(17,16),(18,17),(18,18),(16,18),(15,18),(14,18),(13,18),(12,18),(11,18),(10,18),(9,18),(9,17),(8,17),(7,17)]


queries = [q1,q2,q3,q4,q5]
# ----------------------------------------------- E N D   O F   Q U E R I E S -------------------------------------------------------------


gt1=[(15,7),(15,8),(16,9),(16,10),(17,12),(17,13),(16,12),(15,14),(15,13),(14,11),(12,10),(12,12),(11,11),(10,12),(9,11)]


groundtruths = [gt1]
# ----------------------------------------------- E N D   O F   G R O U N D T R U T H S -------------------------------------------------------------

final_cham=[]


def dtwFunction(gt,q):

    x=[(15,7),(15,8),(16,9),(16,10),(17,12),(17,13),(16,12),(15,14),(15,13),(14,11),(12,10),(12,12),(11,11),(10,12),(9,11)]


    y=[(16,6),(16,7),(16,8),(17,9),(17,10),(17,11),(17,12),(17,13),(17,14),(17,15),(17,16),(18,17),(18,18),(16,18),(15,18),(14,18),(13,18),(12,18),(11,18),(10,18),(9,18),(9,17),(8,17),(7,17)]


    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

    print(d)





def chamferDistance2(gt, q):

    min_x =[]
    min_y = []
    min_t = []
    tsum=[]
    currentIndexOfMin = 0
    maxDifference = 0

    print("length of gt: ", len(gt))
    print("length of q : ", len(q))
    print("Query will always be longer than gt. 5 test case, 1 GT")

    for indq, val in enumerate(q):

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


        curveAdjuster = 300
        similarityScore = round((1.0-((cDistance)/(cDistance + curveAdjuster)))*100.0)

    #print("similarity score: ", similarityScore)
    #print("cDistance: ", cDistance)
    #time.sleep( 2 )

    #print('***************************************')
        print("Similarity score: ", similarityScore)
        print("cDistance:        ", cDistance)
        final_cham.append([similarityScore, cDistance])

    return final_cham




chamf = []
eucl = []
earthm = []
hausdrorff = []


print(chamferDistance2(groundtruths, queries))
dtwFunction(groundtruths, queries)
#https://stackoverflow.com/questions/30706079/hausdorff-distance-between-3d-grids
