import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection

import numpy as np
import collections
import functools

class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)
        

class triangle:
    counter = 0
    def __init__(self,x,y,h,th,p,fc,lc):
        triangle.counter += 1
        self.x = x       # origin x
        self.y = y       # origin y
        self.h = h       # hypotenuse
        self.th = th     # inclination hyp vs x axis in radians
        self.p = p       # projection of rect vertex (C) on hyp [0...1]
        self.fc = fc     # face color
        self.lc = lc     # line color
    
    # vertexes
    def A(self):
        """Origin"""
        return [self.x, self.y]
    def B(self): 
        """Not origin, not rect vertex"""
        return [self.x + self.h * np.cos(self.th), self.y + self.h * np.sin(self.th)]
    def C(self): 
        """Rect vertex"""
        return [self.x + self.h * np.sqrt(self.p) * np.cos(np.arccos(np.sqrt(self.p)) + self.th), 
                self.y + self.h * np.sqrt(self.p) * np.sin(np.arccos(np.sqrt(self.p)) + self.th)]
    def verts(self):
        """All vertexes as list"""
        return [self.A(), self.B(), self.C()]
    def min_x(self):
        """Lowest x point"""
        return min([x[0] for x in self.verts()])
    def min_y(self):
        """Lowest y point"""
        return min([x[1] for x in self.verts()])
    def max_x(self):
        """Highest x point"""
        return max([x[0] for x in self.verts()])
    def max_y(self):
        """Highest y point"""
        return max([x[0] for x in self.verts()])
    
    # angles
    def alpha(self):
        """Returns angle between cat and hyp at vertex A in radians"""
        return np.arccos(np.sqrt(self.p))
    def beta(self):
        """Returns angle between cat and hyp at vertex B in radians"""
        return np.pi/2 - self.alpha()
    
    # drawing
    def to_poly(self):
        """Returns a plt.Polygon of a rect triangle"""
        return plt.Polygon(self.verts(), fc = self.fc, edgecolor = self.lc)

class square:
    counter = 0
    def __init__(self,x,y,a,th,fc,lc):
        square.counter += 1
        self.x = x       # origin x
        self.y = y       # origin y
        self.a = a       # side
        self.th = th     # inclination hyp vs x axis in radians
        self.fc = fc     # face color
        self.lc = lc     # line color
    
    # vertexes
    def A(self):
        """Origin"""
        return [self.x, self.y]
    def B(self): 
        """south east"""
        return [self.x + self.a * np.cos(self.th), 
                self.y + self.a * np.sin(self.th)]
    def C(self): 
        """north east"""
        return [self.B()[0] + self.a * np.cos(self.th + np.pi/2), 
                self.B()[1] + self.a * np.sin(self.th + np.pi/2)]
    def D(self):
        """north west"""
        return [self.x + self.a * np.cos(self.th + np.pi/2), 
                self.y + self.a * np.sin(self.th + np.pi/2)]
    
    def verts(self):
        """All vertexes as list"""
        return [self.A(), self.B(), self.C(), self.D()]
    def min_x(self):
        """Lowest x point"""
        return min([x[0] for x in self.verts()])
    def min_y(self):
        """Lowest y point"""
        return min([x[1] for x in self.verts()])
    def max_x(self):
        """Highest x point"""
        return max([x[0] for x in self.verts()])
    def max_y(self):
        """Highest y point"""
        return max([x[1] for x in self.verts()])
    
    # drawing
    def to_poly(self):
        """Returns a plt.Polygon of a square"""
        return plt.Polygon(self.verts(), fc = self.fc, edgecolor = self.lc)

class branch:
    counter = 0
    def __init__(self,x,y,a,th,p,s_fc,s_lc,t_fc,t_lc):
        branch.counter += 1
        self.x = x       # origin x
        self.y = y       # origin y
        self.a = a       # side
        self.th = th     # inclination hyp vs x axis in radians
        self.p = p       # projection of rect vertex (C) on hyp [0...1]
        self.s_fc = s_fc   # square face color
        self.s_lc = s_lc   # square line color
        self.t_fc = t_fc   # triangle face color
        self.t_lc = t_lc   # triangle line color
    def __str__(self):
        return "<Branch at (%s,%s)>"%(self.x, self.y)
    
    # square and tri
    def s(self):
        """Returns the square as an object"""
        return square(self.x,self.y,self.a,self.th,self.s_fc,self.s_lc)
    def t(self):
        """Returns the triangle as an object"""
        return triangle(self.s().D()[0],self.s().D()[1],self.a,self.th,self.p,self.t_fc,self.t_lc)
    
    # vertexes
    def A(self):
        """Origin"""
        return [self.x, self.y]
    def B(self): 
        """south east"""
        return self.s().B()
    def C(self): 
        """north east"""
        return self.s().C()
    def D(self):
        """north center"""
        return self.t().C()
    def E(self):
        """north west"""
        return self.s().D()
    
    def verts(self):
        """All vertexes as list"""
        return [self.A(), self.B(), self.C(), self.D(), self.E()]
    def min_x(self):
        """Lowest x point"""
        return min([x[0] for x in self.verts()])
    def min_y(self):
        """Lowest y point"""
        return min([x[1] for x in self.verts()])
    def max_x(self):
        """Highest x point"""
        return max([x[0] for x in self.verts()])
    def max_y(self):
        """Highest y point"""
        return max([x[1] for x in self.verts()])
    
    # drawing
    def to_poly(self):
        """Returns a plt.Polygons of the branch"""
        fc=self.s_fc
        lc=self.s_lc
        return plt.Polygon(self.verts(), fc = fc, edgecolor = lc)

    def to_poly_st(self,s=True,t=True):
        """Returns a list with plt.Polygons of the square and the rect triangle"""
        square = plt.Polygon(self.s().verts(), fc = self.s_fc, edgecolor = self.s_lc)
        triangle = plt.Polygon(self.t().verts(), fc = self.t_fc, edgecolor = self.t_lc)
        r = []
        if s == False and t == False:
            r = None
        elif s == False and t == True:
            r.append(triangle)
        elif s == True and t == False: 
            r.append(square)
        else: 
            r.append(square)
            r.append(triangle)
        return r
    
    # growing
    def grow(self,left=True,right=True):
        """Returns the left/rigth branch, if left/right is True"""
        r = []
        if left == True:
            r.append(
                branch(
                     self.E()[0],
                     self.E()[1], 
                     self.a * np.cos(self.t().alpha()), 
                     self.t().alpha() + self.th, 
                     self.p, self.s_fc, self.s_lc, self.t_fc, self.t_lc
                ))
        if right == True: 
            r.append(
                branch(
                    self.D()[0],
                    self.D()[1], 
                    self.a * np.cos(self.t().beta()), 
                    -self.t().beta() + self.th, 
                    self.p, self.s_fc, self.s_lc, self.t_fc, self.t_lc
                ))
        return r

    def grow_c(self,fc,lc,left=True,right=True):
        """Returns the left/rigth branch, if left/right is True, specifying the colors"""
        r = []
        if left == True:
            r.append(
                branch(
                     self.E()[0],
                     self.E()[1], 
                     self.a * np.cos(self.t().alpha()), 
                     self.t().alpha() + self.th, 
                     self.p, fc, lc, fc, lc
                ))
        if right == True: 
            r.append(
                branch(
                    self.D()[0],
                    self.D()[1], 
                    self.a * np.cos(self.t().beta()), 
                    -self.t().beta() + self.th, 
                    self.p, fc, lc, fc, lc
                ))
        return r

        
    @memoized
    def nth_branches(self,n):
        """Returns a list with the nth-order branches growing off the main branch"""
        if n == 0: 
            r = [self]
        else:
            l = []
            for b in self.nth_branches(n-1):
                for item in b.grow():
                    l.append(item)
            r = l
        return r
        
    def tree(self,depth):
        """Returns a dict with all nth-order branches up to depth, growing from main branch"""
        d = {}
        for i in range(depth):
            l = []
            for b in self.nth_branches(i):
                l.append(b)
            d[i]=l
        return d

    def tree_as_coll(self,depth,cmap_name):
        """Returns a tree as a mpl patch collection for plotting, using colormap cmap_name"""
        cmap = plt.get_cmap(cmap_name)
        patches = []
        colors = []
        for i in range(depth):
            for b in self.tree(depth)[i]:
                patches.append(b.to_poly())
                colors.append(cmap(i/float(depth)))
        coll = PatchCollection(patches)
        coll.set_color(colors)
        return coll

    def draw_tree(self,depth,cmap_name):
        fig, ax = plt.subplots()
        cmap = plt.get_cmap(cmap_name)

        patches = []
        colors = []
        for i in range(depth):
            for b in self.tree(depth)[i]:
                patches.append(b.to_poly())
                colors.append(cmap(i/float(depth)))

        coll = PatchCollection(patches)
        ax.add_collection(coll)
        coll.set_color(colors)
        
        nt = collections.namedtuple('results', ['fig','ax','treedict','patchcoll'])
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.autoscale()
        plt.close(fig)
        
        return nt(fig,ax,self.tree(depth),coll)


        
class trunk(branch):
    """A shortcut to make a branch with origin at 0,0 and th = 0"""
    def __init__(self,a,p,fc,lc):
        branch.__init__(self,0,0,a,0,p,fc,lc,fc,lc)
        self.a = a
        self.p = p
        self.fc = fc 
        self.lc = lc    