import numpy as np
from constants import *

class Pt():
    """Container for a Point Box on the OMR"""
    def __init__(self, x, y,val):
        self.x=x
        self.y=y
        self.val=val
    # overloaded
    def __init__(self, pt,val):
        self.x=pt[0]
        self.y=pt[1]
        self.val=val

class Q():
    """
    Container for a Question on the OMR
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """
    def __init__(self,qNo,qType, pts,ans=None):
        self.qNo = qNo
        self.qType = qType
        self.pts = pts
        self.ans = ans

def genRect(orig, qNos, gaps, vals, qType, orient):
    """
    Input:
    orig - start point
    qNo  - a qNos tuple
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - values of each alternative for a question

    Returns set of coordinates of a rectangular grid of points
    
        1 2 3 4
        1 2 3 4
        1 2 3 4

        (q1, q2, q3)

        00
        11
        22
        33
        44
        (q1d1,q1d2)

    """
    Qs=[]
    i0, i1 = (0,1) if(orient=='H') else (1,0)
    o=orig[:] # copy list
    for qNo in qNos:
        pt = o[:] #copy pt
        pts=[]
        for v in vals:
            pts.append(Pt(pt,v))
            pt[i0] += gaps[i0]
        o[i1] += gaps[i1]
        Qs.append( Q(qNo,qType, pts))
    return Qs

def genGrid(orig, qNos, bigGaps, gaps, vals, qType, orient='V'):
    """
    Input:
    orig- start point
    qNos - an array of qNos tuples(see below) that align with dimension of the big grid (gridDims extracted from here)
    bigGaps - (bigGapX,bigGapY) are the gaps between blocks
    gaps - (gapX,gapY) are the gaps between rows and cols in a block
    vals - a 1D array of values of each alternative for a question
    orient - The way of arranging the vals (vertical or horizontal)

    Returns an array of Q objects (having their points) arranged in a rectangular grid

                                00    00    00    00
   Q1   1 2 3 4    1 2 3 4      11    11    11    11
   Q2   1 2 3 4    1 2 3 4      22    22    22    22         1234567
   Q3   1 2 3 4    1 2 3 4      33    33    33    33         1234567
                                44    44    44    44
                            ,   55    55    55    55    ,    1234567                       and many more possibilities!
   Q7   1 2 3 4    1 2 3 4      66    66    66    66         1234567
   Q8   1 2 3 4    1 2 3 4      77    77    77    77
   Q9   1 2 3 4    1 2 3 4      88    88    88    88
                                99    99    99    99

    MCQ type (orient='H')-
        [
            [(q1,q2,q3),(q4,q5,q6)]
            [(q7,q8,q9),(q10,q11,q12)]
        ]

    INT type (orient='V')-
        [
            [(q1d1,q1d2),(q2d1,q2d2),(q3d1,q3d2),(q4d1,q4d2)]
        ]
    
    ROLL type-
        [
            [(roll1,roll2,roll3,...,roll10)]
        ]

    """
    qNos=np.array(qNos)
    if(len(qNos.shape)!=3 or qNos.size==0): # product of shape is zero
        print("genGrid: Invalid qNos array given", qNos)
        return []

    # ^ should also validate no overlap of rect points somehow?!
    gridHeight, gridWidth, numDigs = qNos.shape
    numVals = len(vals)

    Qs=[]
    i0, i1 = (0,1) if(orient=='H') else (1,0)
    hGap, vGap = bigGaps[i1], bigGaps[i0]
    if(orient=='H'):
        hGap += (numVals-1)*gaps[i1]
        vGap += (numDigs-1)*gaps[i0]
    else:
        hGap += (numDigs-1)*gaps[i1]
        vGap += (numVals-1)*gaps[i0]
    qStart=orig[:]
    for row in qNos:
        qStart[i1] = orig[i1]
        for qTuple in row:
            Qs += genRect(qStart,qTuple,gaps,vals,qType,orient)
            qStart[i1] += hGap
        qStart[i0] += vGap

    return Qs

# The utility for GUI            
def calcGaps(PointsX,PointsY,numsX,numsY):
    gapsX = ( abs(PointsX[0]-PointsX[1])/(numsX[0]-1),abs(PointsX[2]-PointsX[3]) )
    gapsY = ( abs(PointsY[0]-PointsY[1])/(numsY[0]-1),abs(PointsY[2]-PointsY[3]) )
    return (gapsX,gapsY)


def scalePts(pts,facX,facY):
    for pt in pts:
        pt = (pt[0]*facX,pt[1]*facY)

# Config for Manual fit - 
startRoll=[150,200] if kv else [113,184] 

startIntsH=[ [903,278] ,[1655, 275] ]
bigGapsIntH, gapsIntH = [128,51],[62,46] #51 means no use

# 903+59*3+117*3-12
startIntsJ=[ [903,275] ,[1418, 275] ] 
bigGapsIntJ, gapsIntJ = [115,51], [59,46] #<- diff coz of single digit labels


bigGapsMcq, gapsMcq = [70,50],[70,230]
bigGapsRoll, gapsRoll = [70,250],[70,230]
scalePts([startRoll,bigGapsIntJ,gapsIntJ,bigGapsIntH,gapsIntH,bigGapsMcq,gapsMcq,bigGapsRoll,gapsRoll],omr_templ_scale[0],omr_templ_scale[1])

def maketemplateINT(orig, qNos,bigGaps, gaps):
    vals = range(10)
    qType, orient= QTYPE_INT, 'V'
    return genGrid(orig, qNos, bigGaps,gaps,vals,qType,orient)

def maketemplateMCQ(orig, qNos):
    bigGaps, gaps =  bigGapsMcq, gapsMcq
    vals= [chr(ord('A')+i) for i in range(4)]
    qType, orient= QTYPE_MCQ, 'H'
    return genGrid(orig, qNos, bigGaps,gaps,vals,qType,orient)

def maketemplateRoll(orig):
    bigGaps, gaps =  bigGapsRoll, gapsRoll
    vals= ['Roll'+str(i) for i in range(9)]
    qType, orient= QTYPE_MCQ, 'H'
    return genGrid(orig, qNos, bigGaps,gaps,vals,qType,orient)

TEMPLATES={'J':[],'H':[]}
# intQs = maketemplateINT(orig=start5to9J,qNos=[[('q1', 'q2', 'q3'), ('q4', 'q5', 'q6')], [('q7', 'q8', 'q9'), ('q10', 'q11', 'q12')]])
TEMPLATES['J'] += maketemplateINT(startIntsJ[0],[[('q'+str(i)+'d1','q'+str(i)+'d2') for i in range(5,9)]], bigGapsIntJ,gapsIntJ)
TEMPLATES['J'] += maketemplateINT(startIntsJ[1],[[('q'+str(i)+'d1','q'+str(i)+'d2') for i in range(8,10)]], bigGapsIntJ,gapsIntJ)
TEMPLATES['H'] += maketemplateINT(startIntsH[0],[[('q'+str(i)+'d1','q'+str(i)+'d2') for i in range(9,13)]], bigGapsIntH,gapsIntH)
TEMPLATES['H'] += maketemplateINT(startIntsH[1],[[('q'+str(i)+'d1','q'+str(i)+'d2') for i in range(13,14)]], bigGapsIntH,gapsIntH)
