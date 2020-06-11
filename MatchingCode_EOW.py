# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:54:18 2020

@author: eugen
"""


import os
import time
import numpy as np
import scipy.io
import scipy.sparse
import random

currPath = os.getcwd()
graphsPath = currPath + '\\Graphs\\'

### LOOP THROUGH GRAPHS HERE

maxGraphTime = 20

filesVec = ['662_bus','b2_ss','bcspwr01','bcspwr10','bcsstk05','can_62','dwt_72',
            'dwt_198','dwt_2680','G15','G17','lp_e226','lshp_406','msc01440',
            'poli','sphere3','mark3jac020sc','bayer04']

bestSizeVec = []
numAugPathsVec = []
numShufflesVec = []
random.seed(33)

for fileInd in range(len(filesVec)):
    
    try:
        File = open(graphsPath+filesVec[fileInd]+'.mtx','r')
        startRow = 14
        if fileInd in [9,10]:
            startRow = 17
        elif fileInd == 11:
            startRow = 66
        li = [i.strip().split() for i in File.readlines()]
        li = li[startRow:]
        E = [[int(i[0]),int(i[1])] for i in li] # edge list
        # remove self-loops
        for e in reversed(E):
            if e[0] == e[1]:
                E.remove(e)
                
        # conduct greedy algorithm first to initialize matching
        V = []
        currM = []
        C = []
        
        for currEdge in E:
            v1 = currEdge[0]
            v2 = currEdge[1]
            if (not v1 in C) and (not v2 in C) and (not v1 == v2):
                C.append(v1)
                C.append(v2)
                currM.append(currEdge)
            if (not v1 in V):
                V.append(v1)
            if (not v2 in V):
                V.append(v2)
        
        V.sort()
        C.sort()
        U = []
        for v in V:
            if not v in C:
                U.append(v)
        U.sort()
        
        # conduct algorithm loop
        currTime = time.time()
        numAugPaths = 0
        numShuffles = 0
        #MAIN OUTER LOOP
        while len(U) > 1 and (time.time()-currTime < maxGraphTime):        
            print('in main outer loop')
            # grab U edges
            Uedges = []
            for e in E:
                if (e[0] in U) or (e[1] in U):
                    Uedges.append(e)
            
            maxEdgeNum = int(np.floor(len(E)*0.3))
            
            # determine which of the U edges and M to keep; use uniform between 5% and 15%
            maxUedges = int(np.floor(len(E)*random.uniform(0.05,0.15)))
            maxMedges = int(np.floor(len(E)*random.uniform(0.05,0.15)))
            
            numShuffles += 1
            # grab random U edges
            currUedges = Uedges.copy()
            if maxUedges < len(Uedges):
                while len(currUedges) > maxUedges:
                    randInd = int(np.floor(random.uniform(0,len(currUedges))))
                    remove = currUedges.pop(randInd)
            # grab random M edges
            currMedges = currM.copy()
            if maxMedges < len(currM):
                while len(currMedges) > maxMedges:
                    randInd = int(np.floor(random.uniform(0,len(currMedges))))
                    remove = currMedges.pop(randInd)
            # use remaining capacity to grab random edges not in currM or incident to U
            maxEdgeNum = maxEdgeNum-len(currMedges)-len(currUedges)
            currEedges = E.copy()
            currEedges = [x for x in currEedges if x not in currUedges]
            currEedges = [x for x in currEedges if x not in currMedges]
            while len(currEedges) > maxEdgeNum:
                randInd = int(np.floor(random.uniform(0,len(currEedges))))
                remove = currEedges.pop(randInd)
            
            # step through each U edge and try to find a path to another U edge
            foundP = False
            while (foundP==False) and (len(currUedges) > 0) and (time.time()-currTime < maxGraphTime):
                print('in cycling through U edges')
                u = currUedges.pop()
                if u[0] in U:
                    leftToExploreNodes = [u[1]]
                    Pstart = u[0]
                else:
                    leftToExploreNodes = [u[0]]
                    Pstart = u[1]
                P = [u] # initialize augmenting path
                exploredNodes = []
                while len(leftToExploreNodes) > 0 and foundP==False and (time.time()-currTime < maxGraphTime):
                    print('in P construction')
                    currEnd = leftToExploreNodes[-1]
                    addedToList = False
                    if np.mod(len(P),2)==1: # P is odd
                        # look in currMedges
                        for mEdge in currMedges:
                            if (currEnd == mEdge[0]) and (not mEdge[1] in exploredNodes) and (not mEdge[1] in leftToExploreNodes) and (addedToList==False):
                                P.append(mEdge)
                                leftToExploreNodes.append(mEdge[1])
                                addedToList = True
                            elif (currEnd == mEdge[1]) and (not mEdge[0] in exploredNodes) and (not mEdge[0] in leftToExploreNodes) and (addedToList==False):
                                P.append(mEdge)
                                leftToExploreNodes.append(mEdge[0])
                                addedToList = True
                        
                    else: # P is even
                        # Look in currUedges first
                        for uEdge in currUedges:
                            if (currEnd == uEdge[0] or currEnd == uEdge[1]) and (not uEdge[0]==Pstart) and (not uEdge[1]==Pstart) and (addedToList==False):
                                # we found our augmenting path
                                P.append(uEdge)
                                addedToList = True
                                foundP = True
                        # then look in Eedges
                        if foundP == False:
                            for eEdge in currEedges:
                                if (currEnd == eEdge[0]) and (not eEdge[1] in exploredNodes) and (not eEdge[1] in leftToExploreNodes) and (addedToList==False):
                                    P.append(eEdge)
                                    leftToExploreNodes.append(eEdge[1])
                                    addedToList = True
                                elif (currEnd == eEdge[1]) and (not eEdge[0] in exploredNodes) and (not eEdge[0] in leftToExploreNodes) and (addedToList==False):
                                    P.append(eEdge)
                                    leftToExploreNodes.append(eEdge[0])
                                    addedToList = True
                    # remove the explored node if nothing added
                    if addedToList == False:
                        removeNode = leftToExploreNodes.pop()
                        removeEdge = P.pop()
                        print('removed ' + str(removeNode))
                        print('removed ' + str(removeEdge))
                        exploredNodes.append(removeNode)
                        
                # END INNER WHILE LOOP
                # we either have an augmenting path or not
                if foundP == True:                    
                    # update U,C
                    if P[0][0] in P[1]:
                        x_0 = P[0][1]
                    else:
                        x_0 = P[0][0]
                        
                    if P[-1][0] in P[-2]:
                        x_t = P[-1][1]
                    else:
                        x_t = P[-1][0]
                    # add x_0,x_t to C
                    U.remove(x_0)
                    U.remove(x_t)
                    C.append(x_0)
                    C.append(x_t)
                    C.sort()
                    numAugPaths += 1
                    # update everything with new edges
                    while len(P) > 0 and (time.time()-currTime < maxGraphTime):
                        print('in P deconstruction')
                        print(P)
                        if np.mod(len(P),2)==1: # currently odd
                            lastEdge = P.pop()
                            currM.append(lastEdge)
                        else: # currently even
                            lastEdge = P.pop()
                            currM.remove(lastEdge)
                        
       
        # END OUTER WHILE LOOP
        numAugPathsVec.append(numAugPaths)
        numShufflesVec.append(numShuffles)
        bestSizeVec.append(len(currM))
        
        print('finished '+filesVec[fileInd])
    except:
        print('hmm')
    
# END FILE FOR LOOP    


# 662_bus: 269/306=0.121; start at 14
# b2_ss: 485/544=0.108; start at 14
# bcspwr01: 14/17=0.176; start at 14
# bcspwr10: 2396/2650(?)=0.096; start at 14
# bcsstk05: 73/76=0.039; start at 14
# can_62: 27/29 = 0.069; start at 14
# dwt_72: 32/32 = 0.000; start at 14
# dwt_198: 96/99 = 0.030; start at 14
# dwt_2680: 1307/1340 = 0.025; start at 14
# G15: 336/400 = 0.160; start at 17
# G17: 334/400 = 0.165; start at 17
# lp_e226: ; start at 66
# lshp_406: 194/203=0.044; start at 14
# msc01440: 706/720=0.019; start at 14
# poli: 685/792=0.135; start at 14
# sphere3: 129/129=0; start at 14
# mark3jac020sc: 4265/4565(?)=0.066; start at 14
# bayer04: 9183/10238(?)=0.106; start at 14


### SCRATCH HERE
#662_bus:	306
#b2_ss:	544
#bcspwr01:	17
#bcspwr10:	2576
#bcsstk05: 	76
#can_62:	29
#dwt_72:	32
#dwr_198 :	99
#dwt_2680:	1340
#G15	400
#G17 :	400
#lp_e226:	200
#lshp_406:	203
#msc01440:	720
#poli:	792
#sphere3:	129
#mark3jac020sc	4554
#bayer04	10238

#arr = [0.121,0.108,0,0.176,0.096,0.039,0.069,0.16,0.165,0,0.03,0.025,0.044,0.066,0.019,0.135,0]
#len(arr)
#np.mean(arr)


