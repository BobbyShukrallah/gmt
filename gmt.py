# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge



class Interval:
    #Initiate by passing a tuple/list of two elements
    def __init__(self,L):
        assert len(L)==2, "You must pass a pair of 2 numbers"
        assert L[0]<L[1], "The first term must be greater than the second"
        #This notation assumes I=[a,b], so a is the left endpoint and b the right endpoint of I
        self.a=np.float64(L[0])
        self.b=np.float64(L[1])
        self.len = self.b-self.a
        #The center/midpoint of the interval we will denote x
        self.x = (self.b+self.a)/2
        

    #Some of the operations below takes as input another interval object, so we need a way of checking we have
    #received an interval object.
    def is_interval(self,x):
        assert type(x) is Interval, "Not an interval"
        
    #This operation allows us to test if a number is in an interval, so "x in I" returns true if x is between the endpoints
    def __contains__(self,x):
        return (x<=self.b) and (x>=self.a)
    
    #Two intervals should equal each other if they have the same endpoints
    def __eq__(self,interval):
        if type(interval) is Interval:
            return (self.a==interval.a ) and (self.b==interval.b) 
        else:
            return False
    
    #We want intervals to equal each other even if they have different names. In the same way that if x=2 and y=2 we
    #want x==y to return True, we want this to happen if x and y are equal to intervals with the same endpoint.
    #This also allows for us to take unions of overlapping collections of intervals and the new set of intervals won't
    #count the same interval twice. To do this, we need to hash the intervals.
    def __hash__(self):
        return hash((self.a,self.b))
        
        
    #This gives us a way of checking if one interval is contained in another, e.g. I<=J returns True 
    #if I is contained in J
    def __le__(self,interval):
        
        self.is_interval(interval)
        
        I=self
        J=interval
        
        return (I.a>= J.a) and (I.b<=J.b)
    
    #Similarly, we define I<J to return true of I is properly contained in J. 
    def __lt__(self,interval):
        
        self.is_interval(interval)
        
        return (self<=interval) and (self!=interval)
    
    #The next two define >= and > similarly.
    def __gt__(self,interval):
        return interval< self
    
    def __ge__(self,interval):
        return interval<= self
    
        
    #We define addition to mean translation, so I+x returns an interval whose endpoints have been translated by x.
    def __add__(self,x):
        return Interval([self.a+x,self.b+x])
    
    #We define subtraction similarly
    def __sub__(self,x):
        return Interval([self.a-x,self.b-x])
    
    #When working with intervals, we usually let tI denote the interval with same center as I but t-times the length.
    def __rmul__(self,t):
        
        #Given an interval I=[a,b], the following returns [ta,tb]
        I = Interval([t * self.a,t * self.b])
        
        #Now we need to shift the interval so that its centre is the same as before.
        #We do this by subtracting the new center and adding the old one.
        I = I + (self.x-I.x)
        
        return  I
    
    #This does similar to __rmul__ but fixes the bottom left corner
    def contract(self,delta):
        return Interval([delta*self.a, delta*self.b])
    
    #The representation of the interval.
    def __repr__(self):
        return str([self.a,self.b])
    
    #The following operation computes the intersection of two intervals.
    def intersect(self,I):
        J=self
        if I<=J:
            return I
        elif I>=J:
            return J
        elif I.a in J:
            if I.a<J.b:
                return Interval([I.a,J.b])
            else:
                return None
        elif I.b in J:
            if J.a<I.b:
                
                return Interval([J.a,I.b])
            else:
                return None
        else:
            return None
        
    #This returns the two subintervals of half the size, so if I were a dyadic interval, it would return the two
    #children at the next level of the dyadic grid. If we specify an integer bigger than 3, it will split the interval
    #into d equal intervals
    def children(self,d=2):
        
        assert d>=2, "d must be at least 2"
        
        assert type(d) is int, "d must be an integer"
        
        return [Interval([self.a + j*self.len/d,self.a + (j+1)*self.len/d ]) for j in range(d)]

    def descendants(self,n,d=2):
        assert n>=0 and type(n) is int, "n must be a nonnegative integer"
        if n==0:
            return [self]
        elif n==1:
            return self.children(d)
        else:
            descendants  = list()
            for I in self.children():
                descendants += I.descendants(n-1,d)
            return descendants
        
    
        


        
        
    
#One Third Trick

#The assumption is that I is an interval in the unit interval [0,1]
def one_third_trick_1d(I): #I is an interval
    if I.len > 1/3:
        return Interval([0,1])
    
    #I < 2^{-k}/3 ---> 3I < 2^{-k} ---> log2(3I)< -k ---> [log2(3I)]+1 = -k
    
    k =  -(np.log2(3 * I.len) //  1 + 1) #This way, k is the largest number so that |I|< 2^{-k}/3
    
    n = (2 ** k -1 )//3
    
    a = I.a 
    b = I.b
    
    

    x = ((a * 2**k)// 1) * (2**(-k)) # largest multiple of 2^{-k} less than a.
    
    J = Interval([x,x+2**(-k)])

    if b<=J.b:
        return J
    else:
        return J+(1/3 - 2**(-k)*n)
    



class Cube:
    #Takes a list of Intervals or tuples/lists that define the intervals whose product is the cube
    def __init__(self,L, parent = None):
        
        self.dim = len(L)
        
        if not all([type(I) is Interval for I in L]):
            self.intervals = [Interval(I) for I in L]
        else:
            self.intervals = L
        self.x = np.array([I.x for I in self.intervals])
        self.len = self.intervals[0].len
        
        self.parent = parent
        
        
        
    def check_dim(self,x):
        assert len(x)==self.dim, f"Vector is of dimension {len(x)}, but cube is of dimension {self.dim}"
        
    def __contains__(self,x):
        self.check_dim(x)
        
        intervals =self.intervals
        #checks if the coordinates of x are in each of the intervals defining Q
        return all([x[i] in self.intervals[i] for i in range(self.dim)])
    
    
    def is_cube(self,x):
        assert type(x) is Cube, "Not a cube"
        
    def __eq__(self,cube):
        if type(cube) is Cube:
            return all([I == J for I,J in zip(self.intervals,cube.intervals)])
        else:
            return False
    
    def __hash__(self):
        return hash(tuple((I.a,I.b) for I in self.intervals))
    
    def __le__(self,cube):
        self.is_cube(cube)
        return all([I <= J for I,J in zip(self.intervals,cube.intervals)])
    
    def __lt__(self,cube):
        return (self <= cube) and (self!=cube)
    
    def __gt__(self,cube):
        return cube < self
    
    def __ge__(self,cube):
        return cube <= self
    
        
    def __add__(self,x):
        self.check_dim(x)
        return Cube([I+x[i] for i, I in enumerate(self.intervals)])
    
    def __sub__(self,x):
        self.check_dim(x)
        return Cube([I-x[i] for i, I in enumerate(self.intervals)])
    
    def __rmul__(self,x):
        return Cube([x*I for I in self.intervals])
    
    #This does similar to __rmul__ but doesn't preserve the center
    def contract(self,delta):
        return Cube([I.contract(delta) for I in self.intervals])
    
    def __repr__(self):
        return "Cube" + str([[I.a,I.b] for I in self.intervals])
    
    def __str__(self):
        return "Cube" + str([[I.a,I.b] for I in self.intervals])
    
    def intersect(self,R):
        Q=self
        intervals = [I.intersect(J) for I,J in zip(Q.intervals, R.intervals)]
        
        if all([I for I in intervals]):
            return Cube(intervals)
        else:
            return None
        
    def corner(self):
        return tuple([I.a for I in self.intervals])
    
    def nodes(self,d=2):
        
        numbers = [[i] for i in range(d)]
        A=numbers
        for _ in range(1,self.dim):
            temp = list()
            for a in A:
                for n in numbers:
                    temp.append(a+n)
            A=temp
        return A
    
    def children(self,d=2):
        
        A=self.nodes(d)
        
        
        return [Cube([I.children(d)[a[i]] for i, I in enumerate(self.intervals)], parent = self) for a in A]

    def descendants(self,n,d=2):
        assert n>=0 and type(n) is int, "n must be a nonnegative integer"
        if n==0:
            return [self]
        elif n==1:
            return self.children(d)
        else:
            descendants  = list()
            for I in self.children():
                descendants += I.descendants(n-1,d)
            return descendants
        
    
def unit_d_cube(d):
    return Cube([[0,1] for i in range(d)])


def unit_square():
    return unit_d_cube(2)

        
def one_third_trick(Q):
    return Cube([one_third_trick_1d(I) for I in Q.intervals])



#The following defines a class for a Euclidean ball given a center x and radius r. 
#The dimension of the ambient space is inferred from the dimension of x. 
#When comparing balls from different dimensions, dimensions will be added on to to the ball when the 
#comparison is made

def match_dims(x,y):
    x=x.flatten()
    y=y.flatten()
    d = max(len(x),len(y))
    x = np.pad(x,(0,d-len(x)))
    y = np.pad(y,(0,d-len(y)))
    return x,y

class Ball:
    def __init__(self,x,r):
        
        self.x=np.array(x)
        assert r>0, "Radius must be positive"
        self.r=r
        self.dim = len(self.x.flatten())
        
    
    def __contains__(self,y):
        if type(y) is Ball:
            
            r=y.r
            
            x,y = match_dims(self.x,y.x)
            
            
            
            return np.linalg.norm(x-y)<=self.r-r
        else:
            x,y = match_dims(self.x,y)
    
            return np.linalg.norm(x-y)<=self.r
    
    def __eq__(self,ball):
        
        if type(ball) is Ball:
            x = tuple(self.x)
            y = tuple(ball.x)
            return (x==y) and (ball.r==self.r)
        else:
            return 
        
    def __hash__(self):
        return hash((tuple(self.x), self.r))
    
    #Returns True if the closed balls touch
    def intersects(self,ball):
        x,y = match_dims(self.x,ball.x)
        return np.linalg.norm(x-y) <=  self.r + ball.r
    
    #Translates ball by vector y. Note that it only works if we write Ball + y, not y+Ball
    def __add__(self,y):
        x,y = match_dims(self.x,y)
        return Ball(x+y,self.r)
    
    #Translates ball by vector y. Note that it only works if we write Ball + y, not y+Ball
    def __sub__(self,y):
        x,y = match_dims(self.x,y)
        return Ball(x-y,self.r)
    
    
    #Dilates the ball by factor delta, so 2*B returns double ball, note that B*2 doesn't work.
    def __rmul__(self,delta):
        assert delta>0, "delta must be a positive number"
        return Ball(self.x, delta*self.r)
    
    
        
    def __repr__(self):
        return "Ball" + str((tuple(self.x),self.r))
    
    def __str__(self):
        return "Ball" + str((tuple(self.x),self.r))
    

    
    
    

    

        
"""
Plotting Functions for balls
"""







def plot_centers(collection,ax,color):
    x = [ball.x[0] for ball in collection]
    y = [ball.x[1] for ball in collection]
    ax.scatter(x,y, marker=".", color=color)
    

def plot_ball(ball, ax, **kwargs):
    assert ball.dim==2, "ball must be two dimensional"
    return ax.add_patch(Circle(tuple(ball.x), ball.r, **kwargs))

def plot_circle(ball, ax, **kwargs):
    assert ball.dim==2, "ball must be two dimensional"
    return ax.add_patch(Wedge(tuple(ball.x), ball.r, 0,360, **kwargs))

def plot_balls(collection, ax=None, figsize=(10,10), centers=None,**kwargs):
    if ax==None:
        fig,ax = plt.subplots(figsize=figsize)
        ax.axis("off")
    for ball in collection:
        plot_ball(ball, ax, **kwargs)

    if centers:
        plot_centers(collection,ax,color=centers)
        
    return ax


def plot_circles(collection, color = "black", w=.01, ax=None, figsize=(10,10), centers=None,**kwargs):
    if ax==None:
        fig,ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        
    for ball in collection:
        plot_circle(ball, ax, width= w*ball.r, **kwargs)

    if centers:
        plot_centers(collection,ax,color=centers)
        
    return ax


def plot_cube(cube,ax,**kwargs):
    assert cube.dim==2, "cube must be two dimensional"
    
    return ax.add_patch(Rectangle(cube.corner(),cube.len,cube.len,**kwargs))

def plot_cubes(collection, color = "black", ax=None, figsize=(10,10), centers=None,**kwargs):
    if ax==None:
        fig,ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        
    for cube in collection:
        plot_cube(cube, ax, **kwargs)

    if centers:
        plot_centers(collection,ax,color=centers)
        
    return ax






