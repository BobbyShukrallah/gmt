# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from .objects import Ball, Cube, Segment, Line


"""
I define a convenience function for making new axes.

Since I have been using these for beamer slides where I may want a sequence of 
images that share common objects, I give the option of returning several axes 
objects


"""
def new_plot(n=1,axis="off",xlim = None,ylim = None, **kwargs):
    assert (type(n) is int) and (n>0), "n must be an integer"
    if n==1:
        fig, ax = plt.subplots(**kwargs)
        ax.set_aspect("equal")
        ax.axis(axis)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        return ax
    else:
        return [new_plot(n=1, xlim = xlim,ylim = ylim,**kwargs) for _ in range(n)]



def plot_projections_of_segments(segments,line,ax, **kwargs):
  
  #The projections of the segments into the line
  projected_segments = line.project_segments(segments)

  #Lines showing the shadow cast by the balls

  shadows = [Segment([d.a,s.a]) for (d,s) in zip(segments,projected_segments)]
  shadows += [Segment([d.b,s.b]) for (d,s) in zip(segments,projected_segments)]

  for s in shadows:
    s.plot(ax, **kwargs)


def plot_projections_of_balls(balls,line,ax, **kwargs):


  #The diameters of the balls parallel to theses segments 
  diameters = [Segment([B.x + B.r * line.b, B.x - B.r * line.b]) for B in balls]

  #Lines showing the shadow cast by the balls, done by looking at the
  #shadows of their diameters instead.
  
  plot_projections_of_segments(diameters, line, ax, **kwargs)
  

def plot_projections_of_cubes(cubes,line,ax, **kwargs):
  
  #The following function finds the diagonal segment that has largest
  #projection into the line spanned by a vector, returning it as a Segment
  #object

  def diagonal(Q,vector):
    corners = sorted(Q.corners(), key = lambda x: np.dot(np.array(x),vector))
    return Segment([corners[0],corners[-1]])

  diagonals = [diagonal(Q,line.b) for Q in cubes]

  #Lines showing the shadow cast by the diagonals, done by looking at the
  #shadows of their diameters instead.
  
  plot_projections_of_segments(diagonals, line, ax, **kwargs)



