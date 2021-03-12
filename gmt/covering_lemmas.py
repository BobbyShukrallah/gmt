# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .objects import Ball



def FiveR(Cover):
    
    #Slightly different from the proof, we will arrange our balls in descending order,
    #but below we will work through the list from back to front, i.e. from largest to smallest.
    
    Cover = sorted(Cover,key=lambda x:x.r)
    
    #We make an empty list that will consist of balls that form our subcover.
    
    subcover = list()
    
    #We add the largest element to our subcover first and pop it from the end of our cover.
    
    subcover.append(Cover.pop())
    
    
    while Cover: #While there are still balls in our cover we haven't visited...
        
        #We consider the largest remaining ball.
        
        B = Cover.pop()
        
        #We assume we'll add it to our subcollection until proven otherwise.
        
        add = True
        
        #Now we check whether B intersects any ball already in our subcollection
        
        for C in subcover:
            if B.intersects(C):
                
                #If B intersects a ball C from the subcollection, we decide not to add it to 
                #the subcollection and we end the loop.
                
                add=False
                break
                
        #If 
        if add==True:
            subcover.append(B)
            
    return subcover




#For this algorithm I decided to use deques rather than lists to make the process a bit more efficient.

from collections import deque


def subcover_of_bounded_overlap(Cover):

    
    #We sort the balls in ascending order and convert them to a deque
    
    Cover = deque(sorted(Cover,key=lambda x:x.r))

    #We now create a subcovering of bounded overlap, and we start by adding the last/largest element of our Cover.

    Subcover=deque()

    Subcover.append(Cover.pop())
    

    while Cover: #While there are still balls in the cover we haven't visited.
        
        #We select the next largest ball
        
        next_ball = Cover.pop()
        
        #We assume we will keep the next ball until proven otheriwse
        
        keep_next_ball = True
        
        #If the next ball's center is in any previously selected ball, we discard it
        
        for ball in Subcover:
            if next_ball.x in ball:
                keep_next_ball = False
                break
                
        if keep_next_ball==True:
            
            #We append on the left of our new subcover so that the resulting deque of balls is still from
            #smallest to largest, so when we start popping balls from the deque again later we do so from
            # largest to smallest. 
            
            Subcover.appendleft(next_ball)
            
    return Subcover

            

#Given a covering by balls, returns a finite number of disjoint collections
#whose total union covers the centers of the original collecion.
def Besicovitch(Cover):
    
    
                
    Subcover = subcover_of_bounded_overlap(Cover)
    
    #We now have a Subcover which will have bounded overlap, but now we need to partition this into 
    #disjoint subcovers
    
    #Notice that the balls in Subcover are now in ascending order since we appended smaller balls on the left.
    
    Subcovers = deque([Subcover])
    
    #We iterate through balls from largest to small. 
    
    Disjoint_Collection = deque()
    Disjoint_Collection.append(Subcovers[0].pop())
    Disjoint_Collections = deque([Disjoint_Collection])
    
    
    #Below, we go through each ball of our subcollection from largest to smallest and create
    #new subcollections as follows: we add the last/largest element from Subcover to a new collection.
    #Inductively, if we encounter a ball that doesn't intersect any ball from some subcollection we've created,
    #then we add it to that collection; otherwise, we create a new subcollection. The proof of the Besicovitch covering
    #theorem shows that we at most create 4^d such subcollections. 
    
    while Subcover:
        B=Subcover.pop()
        new_collection = True
        for collection in Disjoint_Collections:
            add_to_this_collection=True
            for b in collection:
                if b.intersects(B):
                    add_to_this_collection=False
                    break
            if add_to_this_collection==True:
                collection.append(B)
                new_collection=False
                break
        if new_collection == True:
            new_collection = deque([B])
            Disjoint_Collections.append(new_collection)
            
    return Disjoint_Collections
     

