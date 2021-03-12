# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge





class Interval:
    #Initiate by passing a tuple/list of two elements
    #If none are give, it returns the unit interval.
    def __init__(self,L=[0,1]):
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
    
    def strictly_contains(self,x):
        return (x<self.b) and (x>self.a)

    
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
    #If none are given, it returns the unit square in R^2 with its SW corner at 0.
    def __init__(self,L = [[0,1],[0,1]], parent = None):
        
        self.dim = len(L)
        
        if not all([type(I) is Interval for I in L]):
            self.intervals = [Interval(I) for I in L]
        else:
            self.intervals = L
        self.x = np.array([I.x for I in self.intervals])
	#There's an issue that sometimes when translating an interval
	#it returns a cube whose sides have different lengths.
	#I'm deliberating on whether to readjust the cubes when 
	#initializing to guarantee the sides have equal length,
	#but for now I'm allowing unequal sides. For this reason,
	#I define the length to be the maximum of the sidelengths
	#(and hence technically this permit rectangular "cubes"...)

        self.len = max([I.len for I in self.intervals])
        
        
        self.parent = parent
        
        
        
    def check_dim(self,x):
        assert len(x)==self.dim, f"Vector is of dimension {len(x)}, but cube is of dimension {self.dim}"
        
    def __contains__(self,x):
        self.check_dim(x)
        
        intervals =self.intervals
        #checks if the coordinates of x are in each of the intervals defining Q
        return all([x[i] in self.intervals[i] for i in range(self.dim)])
    
    def strictly_contains(self,x):
        self.check_dim(x)
        
        intervals =self.intervals
        #checks if the coordinates of x are in each of the intervals defining Q
        return all([self.intervals[i].strictly_contains(x[i]) for i in range(self.dim)])
    
    
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
        
    def corners(self):

      A= self.nodes(d=self.dim)

      return [[[I.a,I.b][a[i]] for i, I in enumerate(self.intervals)] for a in A ]

    def plot(self,ax, **kwargs):
    
      assert self.dim==2, "cube must be two dimensional"
      
      return ax.add_patch(Rectangle(self.corner(),self.len,self.len,**kwargs))

    


        
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

#The Ball class represents a ball in any Euclidean space,
#defined by supplying a center x and a radius r.
#If no center or radius are specified, it returns the 
#unit ball in R^2 centered at the origin. 
class Ball:
    def __init__(self,x=np.array([0,0]),r=1):
        
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
    
 
    
    
    def plot(self,ax,**kwargs):
        
        assert self.dim==2, "ball must be two dimensional"
        
        ax.add_patch(Circle(tuple(self.x), self.r, **kwargs))
        
        return ax
    
    def plot_circle(self, ax, **kwargs):
        
        assert self.dim==2, "ball must be two dimensional"
        
        return ax.add_patch(Wedge(tuple(self.x), self.r, 0,360, **kwargs))
    
    def plot_center(self,ax,**kwargs):
        
        assert self.dim==2, "ball must be two dimensional"
        
        ax.scatter(self.x[0],self.x[1], marker=".", **kwargs)
        
        
    

    
    
    
#Takes a list of the form [[a,b], [c,d]] where [a,b] and [c,d] are the two endpoints. 

norm = lambda x: np.linalg.norm(x)


#Rotation

class Rotation:
    def __init__(self,theta):
        self.M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        
    def __call__(self,x):
        
        return x @ self.M.T
    
right_angle = lambda x : x @ np.array([[0,1],[-1,0]])

#Orthogonal projection in direction v
class Projection:
    def __init__(self,v):

        self.v=np.array(v)

        #If v is not a unit vector, we normalize it.
        self.v = self.v/norm(self.v)
        
        #When we project in the direction v, that means we project
        #onto a line parallel to the vector perpendicular to v, so we
        #record this vector for later. 
        self.perp = np.array([self.v[1],-self.v[0]])
        
    def __call__(self,x,perp=False):
        
        #The orthogonal projection onto a line orthogonal to v is the same
        #as taking v_perp times the dot product with v_perp. First, we 
        #compute the dot product, and do so in a way that we can compute
        #the dot products of v_perp with a whole array of vectors if we'd like.
        y = (x @ self.perp.T) 
        
        #If x is a flattened array, then the above will also return a flattened
        #array.
        #If x is mx2, then the above is 1xm. We reshape it as mx1 so we can 
        #use broadcasting.
        if len(y.shape)>0:
            y=y.reshape(-1,1)
        
        #With y reshaped, the following vector is a list of multiples of self.perp
        proj = y * self.perp

        #if the perp option is set to True, we instead project onto the line
        #orthogonal to our line, which we can do just by subtracting the normal
        #projection from x
        if perp==True:
            return x-proj
        else:
            return proj
    

#This returns a Segment object representing a line segment [a,b] between two vectors
#a and b. We instantiate the class by supplying a list of two vectors M=[a,b]
class Segment:
    def __init__(self,M=[[0,0],[1,0]]):
        
        
        self.M = np.array(M)
        
        n, self.dim = self.M.shape
        
        assert n == 2, "Must give a 2 x d array"
        
        #The endpoints of the segment will be called a and b respectively.
        self.a = self.M[0]
        
        self.b = self.M[1]
        
        #We also record the midpoint
        self.m = (self.a+self.b)/2
        
        #The length of the segment is the distance between the endpoints.
        self.len = np.linalg.norm(self.a-self.b)
        
    def plot(self,ax, **kwargs):
        
        #We provide the option of defining segments in higher dimensions,
        #but for plotting we assert that the segments must be 2-dimensional
        assert self.dim==2, "segment must be in 2 dimensions"
        
        return ax.plot(self.M[:,0], self.M[:,1], **kwargs)
    
    #We can translate a segment S by x by writing S+x
    def __add__(self,x):
        x = np.array(x).reshape(1,-1)
        return Segment(self.M + x)

    #Similarly we can translate a segment S by -x by writing S-x
    def __sub__(self,x):
        x = np.array(x).reshape(1,-1)
        return Segment(self.M - x)
    
    #x*S returns a segment with same midpoint as S but x times the length
    def __rmul__(self,x):
        return Segment(x*(self.M-self.m) + self.m)

    def __repr__(self):
        return f"Segment{list(self.M)}"
    
    #This allows for transforming a segment S under linear transformations A
    #Note that since S is represented by a matrix M whose
    #ROWS are the endpoints, if we want a segment whose endpoints are the image
    #of the endpoings of S under S, we need to write S @ A.T, not S @ A.
    #Also, because of how __matmul__ works, we can't write A @ S. 
    def __matmul__(self,A):
        return Segment(self.M @ A)


    #Rotates segment by angle theta, either about its center (default) or its 
    #left or right endpoints.
    def rotate(self,theta,how="center"):
      
      R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])

      if how == "center":

        translate = self.m

      elif how == "left":

        translate = self.a

      elif how == "right":

        translate = self.b 

      else:
        
        raise "Options are 'center', 'left', and 'right."
        
    
        

      return Segment( (self.M-translate) @ R.T) + translate

    #Divides segment into n equally sized subsegments
    def split(self,n):

      delta = (self.b-self.a)/n

      return [Segment([self.a + j*delta, self.a + (j+1)*delta]) for j in range(n)]
      

        
class Line:
    #When initializing, we give the option of supplying b as either a vector
    #whose direction is which way we want the line to travel, or as a scalar
    #equal to the angle with the x axis. If it is a vector and normal is set to 
    #true, then we give the line normal to the given vector. 

    #Below, b will be the normalized vector in the direction of our line,
    #a will be the point on the line closest to the origin,
    #and n will be the unit normal vector to the line
    def __init__(self,b, a=np.array([0,0]), normal=False):
        
        if normal==True:
          self.n = np.array(b).flatten()
            
          self.n = self.n/norm(self.n)
            
          self.b = right_angle(self.n)
            
        elif type(b) is float:

          self.b = np.array([np.cos(b), np.sin(b)])

          self.n = right_angle(self.b)


        else:

          self.b = np.array(b).flatten()

          self.b = self.b/norm(self.b)

          self.n = right_angle(b)
            
          

            
        
        assert (self.b != 0).any(), "b can't all be zeros"
        
        self.projection = Projection(self.n)
        
        #We project a onto the line perpendicular to our line,
        #which ensures it is the point on our line closest to the origin.
        self.a = self.projection(np.array(a),perp=True)
        
        assert self.b.shape==(2,), "b must be 2 dimensional"
        
        assert self.a.shape==(2,), "a must be 2 dimensional"
        
        #By default, we would like our normal vector n to be pointing upward,
        #and our direction vector b to be pointing to the right.  
        if self.n[1]<0:
            self.n = -self.n
            
        if self.b[0]<0:
            self.b = -self.b
        
    #We can translate a segment S by x by writing S+x
    def __add__(self,x):
        x = np.array(x).reshape(1,-1)
        return Line(self.b,self.a+x)

    #Similarly we can translate a segment S by -x by writing S-x
    def __sub__(self,x):
        x = np.array(x).reshape(1,-1)
        return Line(self.b,self.a-x)
    
    #Returns the line orthogonal to our line passing through the origin    
    def perp(self):
      return Line([self.b[1],-self.b[0]])
        
    #Defined a call function that just gives f(t) = tb+a. This is mostly
    #for the plot function just after.
    def __call__(self,t):
        
        t = np.array([t]).reshape(-1,1)

        return t * self.b.reshape(1,-1) + self.a.reshape(1,-1)
    
    #In order to plot a line without having to specify the endpoints by hand,
    #the following code takes some matplotlib axes object, finds the x and y 
    #limits, and from that figures out how to plot the line so it fits in your
    #frame.
    def plot(self,ax, **kwargs):
        
       
        if self.b[0]==0:
            
            y1,y2 = ax.get_ylim()
            
            t1 = (y1-self.a[1])/self.b[1]
            
            t2 = (y2-self.a[1])/self.b[1]
            
        else:
            
            
            x1, x2 = ax.get_xlim()
            
            t1 = (x1-self.a[0])/self.b[0]
            
            t2 = (x2-self.a[0])/self.b[0]
            
            
        limits = self([t1,t2])
        
        ax.plot(limits[:,0], limits[:,1], **kwargs)
        
    
    #This projects a vetor or list of vectors onto our line.
    def project(self,X):
        
        #X should either be a 1-dim array or a mx2 array. 
        #The following line reshapes whatever vector we have to a mx2 array
        #so we can do some matrix multiplication
        X = np.array([X]).reshape(-1,2)
        
        #We also reshape the vectors self.b and self.a for matrix multiplication
        b = self.b.reshape(-1,2)

        a = self.a.reshape(-1,2)

        #The following returns a mx2 list of projections into our line.
        return (X @ b.T) * b + a
        
    #The next few functions take a list of balls, segments, or squares and 
    #returns the segments equal to their projections into our line.
    def project_balls(self,balls):
        
        centers = [B.x for B in balls]
            
        radii = [B.r for B in balls]
            
        new_centers = list(self.project(centers))
            
        return [Segment([x - self.b*r, x+self.b*r]) for x,r in zip(new_centers,radii)]
        
    def project_segments(self,segments):
        
        
        return [Segment(self.project(S.M)) for S in segments]

            
    def project_squares(self,squares):
            
        t = np.array([2**(-1/2), 2**(-1/2)])

        projections = []

        for Q in X:
            diagonals = [Segment(Q.x + Q.len*t*np.array(1,1), Q.x + Q.len*t*np.array(-1,-1)),
                         Segment(Q.x + Q.len*t*np.array(-1,1), Q.x + Q.len*t*np.array(1,-1))]

            diagonals = project(diagonals,kind="segmments")

            if diagonals[0].len<diagonals[1].len:
                diagonal = diagonals[1]
            else:
                diagonal = diagonals[0]
            projections.append(diagonal)

        return projections
        
        
        
        
    def __repr__(self):
        return f"Line[{list(self.b[0])},{list(self.a[0])}]"
        
    
    





#This defines a Cone object, representing a cone centered at a point x
#(default the origin) with axis parallel to a vector v (default the vector
#pointing up), an aperture (an angle defaulted to pi/4), and finally a radius
#default 1. If two_sided = True, then this will return a two-sided cone. 

#We also give the option of supplying an angle instead of a vector v.

class Cone:
  def __init__(self,x=np.array([0,0]),v = np.array([0,1]),aperture=np.pi/4,r=1,two_sided=False):
    
    self.x = np.array(x).flatten()

    self.v = np.array(v).flatten()

    self.r = r

    self.aperture = aperture

    self.two_sided = two_sided
    
    if self.v.shape == (1,):

      self.theta = v
    
      self.v = np.array([np.cos(self.theta),np.sin(self.theta)])
      
    else:
      
      #If the vector v is vertical, we let theta be either pi/2 or 3pi/2 depending
      #on which direction v is pointing.
      if v[0]==0:

        self.theta = np.pi/2  + np.pi * (1- np.sign(v[1]))/2
      else:
        #Otherwise, we can find theta using arctan.
        
        self.theta = np.arctan(v[1]/v[0])

  #Translates cone by vector y. Note that it only works if we write Cone + y, not y+Cone
  def __add__(self,y):
        
        return Cone(self.x+np.array(y),self.v, self.aperture, self.r, self.two_sided)
    
  #Translates cone by vector -y. Note that it only works if we write Cone - y, not y-Cone
  def __sub__(self,y):
        
    return Cone(self.x-np.array(y),self.v, self.aperture, self.r, self.two_sided)

  def plot(self,ax,**kwargs):

    degrees = lambda x: 360 * x / (2*np.pi)
    theta = degrees(self.theta)

    aperture = degrees(self.aperture)
      
    ax.add_patch(Wedge(tuple(self.x), self.r, theta-aperture,theta+aperture, **kwargs))

    if self.two_sided == True:
      if theta - 180>0:
        ax.add_patch(Wedge(tuple(self.x), self.r, theta-180-aperture,theta-180+aperture, **kwargs))
      else:
        ax.add_patch(Wedge(tuple(self.x), self.r, theta+180-aperture,theta+180+aperture, **kwargs))
      

     
        



def random_curve(segment,height,n, return_segments = False):
    if return_segments==False:
        segments = random_curve(segment,height,n, return_segments=True)

        return np.array([s.a for s in segments])
    else:

        height = height*((-1)**np.random.randint(2))


            

        if n==1:

            return [segment]

        elif n==2:

            segments = segment.split(3)
            s=segments[1]
            v = s.b-s.a
            normal = np.array([v[1],-v[0]])

            raised_segments = s.split(2)

            return [segments[0]]  + [Segment([raised_segments[0].a,raised_segments[0].b + height * normal]),
                    Segment([raised_segments[1].a + height * normal,raised_segments[1].b])] + [segments[2]] 

        else:

            return [t for s in random_curve(segment,height,2, return_segments=True) 
                    for t in random_curve(s,height,n-1, return_segments=True)]
            

        

        
        
        
"""
Plotting Functions for balls
"""







def plot_centers(collection,ax,color):
    x = [ball.x[0] for ball in collection]
    y = [ball.x[1] for ball in collection]
    ax.scatter(x,y, marker=".", color=color)
    

def plot_ball(ball, ax, **kwargs):
    assert self.dim==2, "ball must be two dimensional"
    return ax.add_patch(Circle(tuple(ball.x), ball.r, **kwargs))

def plot_circle(ball, ax, **kwargs):
    assert self.dim==2, "ball must be two dimensional"
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






