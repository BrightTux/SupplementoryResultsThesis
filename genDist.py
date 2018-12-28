
#import statements


#list of trajectories for both gt and query [x,y]

q1=[(16,7),(16,8),(16,9),(17,10),(17,11),(17,12),(17,13),(16,13),(15,13),(14,13),(13,13),(13,12),(12,12),(11,12),(10,11),(9,11)]

q2=[(17,13),(16,13),(15,13),(14,12),(13,11),(12,12),(11,12),(10,12),(9,12),(8,12),(7,12),(6,12),(5,12),(4,11),(3,11)]

q3=[(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12),(16,13),(15,13),(14,13),(13,13),(13,12),(11,12),(10,12),(10,11),(9,11),(8,11),(7,11),(6,11),(5,11),(5,12),(5,13)]

q4=[(15,4),(15,5),(15,6),(14,6),(13,6),(12,6),(11,6),(11,7),(10,7),(9,7),(9,8),(8,8),(7,8),(7,9),(6,9),(5,10),(4,10),(3,10),(2,10),(1,11),(0,11)]

q5=[(16,6),(16,7),(16,8),(17,9),(17,10),(17,11),(17,12),(17,13),(17,14),(17,15),(17,16),(18,17),(18,18),(16,18),(15,18),(14,18),(13,18),(12,18),(11,18),(10,18),(9,18),(9,17),(8,17),(7,17)]

q6=[(15,6),(16,7),(16,8),(16,9),(16,10),(17,11),(17,12),(17,13),(18,13),(18,14),(19,14),(19,13)]

q7=[(2,11),(3,11),(3,10),(4,10),(5,10),(5,9),(6,9),(7,9),(7,8),(8,8),(9,8),(9,9),(10,9),(11,9),(12,9),(13,9),(14,9),(15,9),(16,9),(16,10)]

q8=[(15,5),(15,6),(15,7),(15,8),(16,8),(16,9),(16,10),(17,10),(17,11),(17,12),(17,13),(17,14),(17,15)]

q9=[(15,6),(16,6),(16,7),(16,8),(16,9),(16,10),(15,10),(14,10),(13,10),(12,10),(11,10),(10,9),(9,9),(8,9),(8,8)]

q10=[(2,17),(3,17),(4,17),(5,17),(6,17),(8,17),(8,18),(9,18),(10,18),(11,18),(12,18),(13,18),(14,18),(15,18),(15,19)]

q11=[(15,6),(15,7),(16,7),(16,8),(16,9),(16,10),(17,10),(17,11),(17,12),(17,13),(16,13),(15,13),(14,12),(13,12),(12,12),(11,12),(10,12),(9,12),(8,12),(7,12),(7,11),(6,11),(5,11),(8,9)]



queries = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11]
# ----------------------------------------------- E N D   O F   Q U E R I E S -------------------------------------------------------------


gt1=[(15,6),(16,8),(16,9),(17,10),(17,11),(17,12),(17,13),(16,13),(15,13),(14,13),(13,13),(13,12),(12,12),(11,12),(10,11),(9,11)]

gt2=[(17,13),(16,13),(15,13),(14,12),(13,11),(12,12),(11,12),(10,12),(9,12),(8,12),(7,12),(6,12),(5,12),(4,11),(3,11)]

gt3=[(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12),(16,13),(15,13),(14,13),(13,13),(13,12),(11,12),(10,12),(10,11),(9,11),(8,11),(7,11),(6,11),(5,11),(5,12),(5,13)]

gt4=[(15,4),(15,5),(15,6),(14,6),(13,6),(12,6),(11,6),(11,7),(10,7),(9,7),(9,8),(8,8),(7,8),(7,9),(6,9),(5,10),(4,10),(3,10),(2,10),(1,11),(0,11)]

gt5=[(16,6),(16,7),(16,8),(17,9),(17,10),(17,11),(17,12),(17,13),(17,14),(17,15),(17,16),(18,17),(18,18),(16,18),(15,18),(14,18),(13,18),(12,18),(11,18),(10,18),(9,18),(9,17),(8,17),(7,17)]

gt6=[(15,6),(16,7),(16,8),(16,9),(16,10),(17,11),(17,12),(17,13),(18,13),(18,14),(19,14),(19,13)]

gt7=[(2,11),(3,11),(3,10),(4,10),(5,10),(5,9),(6,9),(7,9),(7,8),(8,8),(9,8),(9,9),(10,9),(11,9),(12,9),(13,9),(14,9),(15,9),(16,9),(16,10)]

gt8=[(15,5),(15,6),(15,7),(15,8),(16,8),(16,9),(16,10),(17,10),(17,11),(17,12),(17,13),(17,14),(17,15)]

gt9=[(15,6),(16,6),(16,7),(16,8),(16,9),(16,10),(15,10),(14,10),(13,10),(12,10),(11,10),(10,9),(9,9),(8,9),(8,8)]

gt10=[(2,17),(3,17),(4,17),(5,17),(6,17),(8,17),(8,18),(9,18),(10,18),(11,18),(12,18),(13,18),(14,18),(15,18),(15,19)]

gt11=[(15,6),(15,7),(16,7),(16,8),(16,9),(16,10),(17,10),(17,11),(17,12),(17,13),(16,13),(15,13),(14,12),(13,12),(12,12),(11,12),(10,12),(9,12),(8,12),(7,12),(7,11),(6,11),(5,11),(8,9)]



groundtruths = [gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,gt9,gt10,gt11]
# ----------------------------------------------- E N D   O F   G R O U N D T R U T H S -------------------------------------------------------------


def chamferDistance(gt, q):
    if(len(gt) >= len(q)):
       outerloop = len(gt)
       innerloop = len(q)
       outer = gt
       inner = q
    else:
        outerloop = len(q)
        innerloop = len(gt)
        outer = q
        inner = gt

    for oidx, o in enumerate(outer):
        for iidx, i in enumerate(inner):
            min_x.append(abs(o[0] - i[0]))
            min_y.append(abs(o[1] - i[1]))
            min_t.append(abs(oidx - iidx))

        for k in len(min_x):
            sum.append(min_x[k] + min_y[k] + min_t[k])

        indexOfMin = sum.indexOf(Math.min(...sum))

        if(currentIndexOfMin == 0):
            ModifiedMinValue = Math.min(...sum)
        elif(currentIndexOfMin != 0 && indexOfMin == currentIndexOfMin):
            ModifiedMinValue = Math.min(...sum)
            ModifiedMinValue++

        maxDifference = maxDifference + ModifiedMinValue
        min_x = [], min_y = [] , min_o = [], sum = []
                                                                    }


