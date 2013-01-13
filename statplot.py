# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import numpy
import __builtin__
all = __builtin__.all
any = __builtin__.any
sum = numpy.sum

# <codecell>

from matplotlib import cm,colors
cm_bkr=colors.LinearSegmentedColormap.from_list('mycm',[(0,'b'),(0.5,'k'),(1,'r')])

# <codecell>

def multiget(dictionary,keylist,default=None):
    """
    returns the value in the dict from multiple, equivalent keys (that shouldn't be duplicated)

    >>> a = dict(a = 1, b = 2)
    >>> multiget(a,['a'],0)
    1
    >>> multiget(a,['c'],0)
    0
    >>> multiget(a,['a','b'],0)
    Traceback (most recent call last):
    ...
    AttributeError: double definition for keylist: a, b
    """
    res = None
    for key in keylist:
        if key in dictionary:
            if res is None:
                res = dictionary[key]
            else:
                raise AttributeError('double definition for keylist: '+", ".join(keylist))
    res = default if res is None else res
    return res

# <markdowncell>

# #Violin plot and boxplot

# <codecell>


from scipy.stats import gaussian_kde,sem
import pylab as plt

def half_violin_plot(data, pos, left=False, **kwargs):
    #http://pyinsci.blogspot.it/2009/09/violin-plot-with-matplotlib.html
    #get the value of the parameters
    amplitude = kwargs.pop('amplitude',0.33)
    ax = kwargs.pop('ax',plt.gca())
    #evaluate the violin plot
    x = np.linspace(min(data),max(data),101) # support for violin
    v = gaussian_kde(data).evaluate(x) #violin profile (density curve)
    v = v/v.max()*amplitude * (1 if left else -1) #set the lenght of the profile
    kwargs.setdefault('facecolor','r')
    kwargs.setdefault('alpha',0.33)
    return ax.fill_betweenx(x,pos,pos+v,**kwargs)

def violin_plot(data1,classes=None,data2=None,**kwargs):
    ax = kwargs.get('ax',plt.gca())
    positions=range(len(data1))
    data2 = data2 if data2 is not None else data1
    classes = classes if classes is not None else positions
    assert len(classes)==len(data1) and len(classes)==len(data2)
    for pos,key in zip(positions,classes):
        try:
            d1,d2=data1[key],data2[key]
        except TypeError:
            d1,d2=data1[pos],data2[pos]
        color1=kwargs.pop('color1','b')
        color2=kwargs.pop('color2','b' if data1 is data2 else 'r')
        half_violin_plot(d1,pos,False,facecolor=color1)
        half_violin_plot(d2,pos,True,facecolor=color2)
        #division line between the two half
        plt.plot([pos]*2,[min(min(d1),min(d2)),max(max(d1),max(d2))],'k-')
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in classes])

if __name__=='__main__':
    ax=plt.figure().add_subplot(2,1,1)
    n=100
    data=[normal(size=n)+i for i in range(4)]
    violin_plot(data,ax=ax)
    
    figure()
    pos=['dog','cat','horse','mouse']
    data=[normal(size=n) for i in range(len(pos))]
    violin_plot(data,pos)
    
    figure()
    pos=['dog','cat','horse','mouse']
    data1={i:normal(size=n) for i in pos}
    data2={i:normal(size=n) for i in pos}
    violin_plot(data1,pos,data2)

# <codecell>

from itertools import cycle
def fillboxplot(ax, data, **keywords):
    vert = keywords.get('vert',1)
    if keywords.get('vert',1):
        ax.tickNames = plt.setp(ax, xticklabels=keywords.pop('names',[]) )
    else:
        ax.tickNames = plt.setp(ax, yticklabels=keywords.pop('names',[]) )
    colors = keywords.pop('colors',['0.95'])
    
    bp = ax.boxplot(data, patch_artist=True, **keywords)
    for r,c in zip(bp['boxes'],cycle(colors)):
        r.set_facecolor(c)
    pylab.setp(bp['boxes'], edgecolor='k')
    pylab.setp(bp['whiskers'], color='black', linestyle = 'solid')
    pylab.setp(bp['fliers'], color='black', alpha = 0.9, marker= 'o', markersize = 3)
    pylab.setp(bp['medians'], color='black')

    return bp

if __name__=='__main__':
    import scipy.stats
    data = [scipy.stats.norm.rvs(size = 100), scipy.stats.norm.rvs(size = 100), scipy.stats.norm.rvs(size = 100)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.legend()
    fillboxplot(ax, data, names = ("One", "Two", "Three"), colors = ('white', 'cyan'),vert=0);

# <codecell>

from itertools import cycle
from matplotlib.colors import hex2color
from  matplotlib.colors import colorConverter as cc
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

single_rgb_to_hsv=lambda rgb: rgb_to_hsv( array(rgb).reshape(1,1,3) ).reshape(3)
single_hsv_to_rgb=lambda hsv: hsv_to_rgb( array(hsv).reshape(1,1,3) ).reshape(3)

def desaturate(color):
    hsv = single_rgb_to_hsv(color)
    hsv[1] = 0.5
    hsv[2] = 0.7
    return single_hsv_to_rgb(hsv)


def desaturize(ax=None):
    if ax is None: ax=plt.gca()
    ax.set_axisbelow(True)
    ax.set_axis_bgcolor([0.8]*3) 
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('outward',10))
    ax.spines['left'].set_position(('outward',10))
    ax.spines['left'].set_edgecolor('gray')
    ax.spines['bottom'].set_edgecolor('gray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #ax.spines['bottom'].set_smart_bounds(True)
    #ax.spines['left'].set_smart_bounds(True)
    ax.grid(True,color='w',linestyle='-',linewidth=2)
    for line in ax.lines:
        col = line.get_color()
        line.set_color(desaturate(cc.to_rgb(col)))
    for patch in ax.patches:
        col = patch.get_facecolor()
        patch.set_facecolor(desaturate(cc.to_rgb(col)))
        patch.set_edgecolor(patch.get_facecolor())
    #ax.invert_xaxis()
    return ax

if __name__=='__main__':
    fig,(ax1,ax2) = subplots(1,2,figsize=(9,4))

    ax1.plot([1,2,1,4],linewidth=2,color='r')
    ax1.bar(arange(3)-0.40,[1,2,3],[0.8,0.8,0.8])
    
    ax2.plot([1,2,1,4],linewidth=2,color='r')
    ax2.bar(arange(3)-0.40,[1,2,3],[0.8,0.8,0.8])

    desaturize(ax2)

# <codecell>

import pylab as plt
import numpy as np
from collections import Counter

def explore(data,**kwargs):
    ax = plt.gca()
    res=Counter(data)
    key=sorted(res.keys())
    #nel caso siano degl interi riempe i numeri vuoti
    if isinstance(key[0],int):
        key=range(min(key),max(key)+1)
    val=[res[i] for i in key]
    kwargs.update({'align':'center'})
    rects = ax.bar(range(len(val)), val,**kwargs)
    #gestione delle label x
    ax.set_xticks(range(len(val)))
    ax.set_xlim(-0.5,len(val)-0.5)
    ax.set_xticklabels(key)
    #gestione delle label y
    ax.set_ylabel('Counts')
    ax.set_yticks([ int(i) for i in ax.get_yticks() if i==int(i) ])
    ax.set_ylim(0.,ax.get_ylim()[1]*1.05)
    return rects
  

if __name__=='__main__':
    explore([1,1,1,2,2,3,5,5,5,5,5]+[10]*20)
    figure()
    esamina('pippo')
    #figure()
    #esamina(['male','female','male','female','male'], facecolor='#777777', ecolor='black')

# <markdowncell>

# #Plotting lambdas function

# <codecell>

def plotline(grad, inter=0,*args,**kwargs):
    """plot a regression line on the plot
    Parameter:
        grad: float
            the slope of the line
        inter: float
            the intercept of the line
    
    it will plot the given regression line on the current axis, with the formula
    
    y =  inter + grad * x
    
    Return:
        None

    Examples:
    >>> from scipy.stats import linregress
    >>> x = rand(10)
    >>> y = 0.1 * x + rand(10)
    >>> plot(x,y,'.')
    >>> plotline(*linregress(x, y),color='r')
    """
    ax = gca()
    x0,x1 = ax.get_xlim()
    #x1 = x1 - 0.01 * (x1-x0)
    yo0,yo1 = ax.get_ylim()
    y0 = inter + grad * x0
    y1 = inter + grad * x1
    ax.plot([x0,x1],[y0,y1],**kwargs)
    ax.set_ylim(yo0,yo1)
    ax.set_xlim(x0,x1)

if __name__=='__main__':
    from scipy.stats import linregress
    x = rand(10)
    y = 0.1 * x + rand(10)
    plot(x,y,'.')
    #grad, inter, r, p, std_err = linregress(x, y)
    #plotline(grad,inter,color='r',linewidth=4)
    plotline(*linregress(x, y),color='r')

# <codecell>

#plot a single parameter function
def plotfunc(func, step = 100, *args,**kwargs):
    ax = kwargs.pop('ax',plt.gca())
    xmin,xmax = kwargs.pop('xlim',ax.get_xlim())
    ymin,ymax = kwargs.pop('ylim',ax.get_ylim())
    x = linspace(xmin,xmax,step)
    y_base = array(np.vectorize(func)(x))
    y = where((y_base>ymin) & (y_base<ymax), y_base, np.nan)
    ax.plot(x,y,*args,**kwargs)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    
if __name__=='__main__':
    x = rand(30)*2
    y = x + rand(len(x)) -1
    plot(x,y,'.')
    plotfunc(lambda x: x+x**2-x**3, xlim=(0.,2.), ylim=(-1,1))

# <codecell>

import inspect
def plot_func_1to1(function,domain=None,N=100.,*args,**kwargs):
    if domain is None:
        domain = linspace(-1.,1.,num=N+1)
    y = function(domain)
    gca().plot(domain, y, *args,**kwargs)

def apply_func_2(function,domain=None,N=100.):
    if domain is None:
        x = linspace(-1.,1.,num=N+1)
        y = linspace(-1.,1.,num=N+1)
        domain=meshgrid(x,y)
    xt,yt=domain
    dx=np.max(xt)-np.min(xt)
    dy=np.max(yt)-np.min(yt)
    z = function(*domain)
    return z, dx,dy
    
def plot_func_2to1(function,z,domain, dx,dy,N=100.,*args,**kwargs):
    z,dx,dy = apply_func_2(function,domain=domain,N=N)
    gca().imshow( z, *args,**kwargs)
    gca().yaxis.set_major_formatter(FuncFormatter(lambda x,pos: dx*x/N-1))
    gca().xaxis.set_major_formatter(FuncFormatter(lambda y,pos: dy*y/N-1))

    
def plot_func_2to2(function,z,domain,dx,dy,N=100.,*args,**kwargs):
    z,dx,dy = apply_func_2(function,domain=domain,N=N)
    U,V = z
    gca().quiver( U,V, *args,**kwargs)
    gca().yaxis.set_major_formatter(FuncFormatter(lambda x,pos: dx*x/N-1))
    gca().xaxis.set_major_formatter(FuncFormatter(lambda y,pos: dy*y/N-1))
    
def plot_lambda(function,domain=None,N=100,*args,**kwargs):
    """
    print a function over a domain. it inspect the function to infer
    wich kind of function it is and plot by consequence
    """
    if len(inspect.getargspec(function).args)<=1:
        #se parte da una dimensione e restituisce un valore ne faccio il grafico
        plot_func_1to1(function,domain=domain,N=N,*args,**kwargs)
    elif len(inspect.getargspec(function).args)==2:
        #testo per vedere se restituisce una funzione a uno o due valori
        z,dx,dy = apply_func_2(function,domain=array([[0,],[0,]]),N=N)
        if len(z)==2:
            z,dx,dy = apply_func_2(function,domain=domain,N=25)
            plot_func_2to2(function,z,domain,dx,dy,N=25,*args,**kwargs)
        else:
            z,dx,dy = apply_func_2(function,domain=domain,N=N)
            plot_func_2to1(function,z,domain,dx,dy,N,*args,**kwargs)
        
if __name__=='__main__':
    f = lambda x: x**2-x**3
    plot_lambda(f)
    figure()
    g = lambda x,y: (x**2+y**2)*cos(x)
    plot_lambda(g)
    figure()
    h = lambda x,y: (x+y,x-y)
    plot_lambda(h)

# <markdowncell>

# #Applying gradients to set of patches

# <codecell>

def _repatch(rect,cmin,cmax,cbot=0.,ctop=1.,cmap=cm.jet,n=10):
    ax = rect.axes
    ax.set_autoscale_on(False)
    base = np.repeat(np.linspace(cmin,cmax,n).reshape(1,-1),2,axis=0)
    rect.remove()
    x,y,w,h = rect.get_bbox().bounds
    im = ax.imshow(base,extent=(x,x+w,y,y+h), cmap = cmap, vmin = cbot, vmax= ctop, aspect='auto')
    ax.set_autoscale_on(True)
    return im

def repatch_set(rects,cmap=cm.jet):
    images = []
    cmin = min( rect.get_bbox().bounds[0] for rect in rects )
    cmax = max( rect.get_bbox().bounds[0]+rect.get_bbox().bounds[2] for rect in rects )
    for rect in rects:
        x,y,w,h = rect.get_bbox().bounds
        images.append(_repatch(rect,x,x+w,cmin,cmax,cmap))
    return images

if __name__=='__main__':
    fig, ax = subplots(1,figsize=(4,4))
    rects = ax.bar(range(11),range(1,6)+[6]+range(1,6)[::-1],[1.]*11)
    imgs = repatch_set(rects,cm.jet)

    fig, ax = subplots(1,2,figsize=(8,4))
    img = randn(30,30)
    cmap = cm.winter
    ax[0].imshow(img,interpolation='nearest',cmap=cmap)
    _,_,rects = hist(img.flat)
    imgs = repatch_set(rects,cmap)

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import scipy

def patch_gradient(patch,  direction = lambda x,y: x, **kwargs):
    """
    take a patch and apply a gradient to it.

    :patch: the patch to be decorated
    :direction: if a number, indicates the direction of the linear gradient, otherway it should be a callable

    the function take several optional arguments:
    
    colormap: the colormap used for the gradient [any valid colormap, default cm.jet]
    
    BUG: if the clipping patch is not a rectangle the alpha value get lost
    """
    
    ax = plt.gca()
    #loading of the default keywords
    colormap = multiget(kwargs,['colormap','cmap','cm'],cm.jet)
    colormap = plt.get_cmap(colormap)
    
    resolution = multiget(kwargs,['resolution','res'],101j)
    alpha = multiget(kwargs,['alpha'],1)
    x_min = multiget(kwargs,['x_min','xmin'],-1)
    x_max = multiget(kwargs,['x_max','xmax'],1)
    y_min = multiget(kwargs,['y_min','ymin'],-1)
    y_max = multiget(kwargs,['y_max','ymax'],1)
    c_min = multiget(kwargs,['c_min','cmin'],None)
    c_max = multiget(kwargs,['c_max','cmax'],None)
    edgecolor = multiget(kwargs,['edgecolor','ec'],None)
    linestyle = multiget(kwargs,['linestyle','ls'],None)
    linewidth = multiget(kwargs,['linewidth','lw'],None)
    #set the function of the gradient
    try:
        dir2rad = scipy.deg2rad(1.*direction)
        xmean = (x_max+x_min)/2.
        ymean = (y_max+y_min)/2.
        xampl = (x_max-x_min)/2.
        yampl = (y_max-y_min)/2.
        dir_func = lambda x,y: ((x-xmean)/xampl)*np.cos(dir2rad) + ((y-ymean)/yampl)*np.sin(dir2rad)
    except TypeError:
        dir_func = direction        
    #get the extent of the patch
    extent = patch.get_extents().transformed(ax.transData.inverted()).extents
    extent[1],extent[2] = extent[2],extent[1]
    #create the grid on which the function will be evaluated
    yy,xx = np.ogrid[y_min:y_max:resolution,x_min:x_max:resolution]
    data = dir_func(xx,yy)
    #temporally disable the autoscale to avoid problem with the imshow
    autoscale = ax.get_autoscale_on()
    ax.set_autoscale_on(False)
    #create the image on the patch
    props = dict(extent=extent,origin='lower',cmap=colormap,alpha=alpha,aspect='auto')
    if c_min is not None: 
        props.update(vmin=c_min)
    if c_max is not None: 
        props.update(vmax=c_max)
    im = ax.imshow(data,**props)
    im.set_alpha(alpha)
    #remove the foreground from the patch and set the line properties
    patch.set_fc('none')
    patch.set_alpha(alpha)
    if edgecolor is not None: patch.set_edgecolor(edgecolor)
    if linestyle is not None: patch.set_linestyle(linestyle)
    if linewidth is not None: patch.set_linewidth(linewidth)
    #apply the clipping and restore the original autoscale setting
    im.set_clip_path(patch)
    ax.set_autoscale_on(autoscale)
    return im

if __name__=='__main__':
    fig, ax = subplots(1,figsize=(8,8))
    border = patches.Circle((.6,.6),radius=.3)
    ax.add_patch(border)
    patch_gradient(border,direction=0,alpha=0.5, cmap='Paired')

    border = patches.Rectangle((0,0),.4,.4,fc='none')
    ax.add_patch(border)
    patch_gradient(border,lambda x,y: cos(exp(4*x**2+4*y**2)), cm=cm.summer, res=1001j, alpha=1 )

    border = patches.RegularPolygon((0.2,0.8),5,radius=0.3)
    ax.add_patch(border)
    im = patch_gradient(border,direction=-40,alpha=0.6)

# <codecell>

def gradient_patchset(patchset,**kwargs):
    ax = patchset[0].axes
    extremes = [ rect.get_extents().transformed(ax.transData.inverted()).extents for rect in patchset]
    xmin_etr = min( i[0] for i in extremes )
    xmax_etr = max( i[2] for i in extremes )
    ymin_etr = min( i[1] for i in extremes )
    ymax_etr = max( i[3] for i in extremes )
    direction = multiget(kwargs,['direction','dir'],0)
    try:
        dir2rad = scipy.deg2rad(1.*direction)
        xmean = (xmax_etr+xmin_etr)/2.
        ymean = (ymax_etr+ymin_etr)/2.
        xampl = (xmax_etr-xmin_etr)/2.
        yampl = (ymax_etr-ymin_etr)/2.
        dir_func = lambda x,y: ((x-xmean)/xampl)*np.cos(dir2rad) + ((y-ymean)/yampl)*np.sin(dir2rad)
    except TypeError:
        dir_func = direction 
    yy,xx = np.ogrid[ymin_etr:ymax_etr:101j,xmin_etr:xmax_etr:101j]
    z = dir_func(xx,yy)
    cmin,cmax = np.min(z),np.max(z)
    kwargs.update(direction=dir_func,cmax=cmax,cmin=cmin)
    imgs = []
    for (xmin,ymin,xmax,ymax),rect in zip(extremes,patchset):
        im = patch_gradient(rect,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,**kwargs)
        imgs.append(im)
    return imgs

if __name__=='__main__':
    pg = patch_gradient
    fig, ax = subplots(1,2,figsize=(13,6))
    #numpy.random.seed(0)
    data = rand(100,100)
    ax[0].imshow(data)
    _,_,rects = ax[1].hist(data.flat)
    gradient_patchset(rects);
    
    pg = patch_gradient
    fig, ax = subplots(1,figsize=(6,6))
    rects = [ Rectangle((i,i),0.2,0.2) for i in [0.,0.2,0.4,0.6,0.8] ]
    for rect in rects:
        ax.add_patch(rect)
    gradient_patchset(rects,direction=45);

# <markdowncell>

# ----

# <markdowncell>

# #Mosaic plot

# <codecell>



from numpy import iterable,r_,cumsum
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from collections import Counter, OrderedDict

single_rgb_to_hsv=lambda rgb: rgb_to_hsv( np.array(rgb).reshape(1,1,3) ).reshape(3)
single_hsv_to_rgb=lambda hsv: hsv_to_rgb( np.array(hsv).reshape(1,1,3) ).reshape(3)

def split_rect(point,width,height,proportion,direction='horizontal',gap=0.05):
    """
    divide un rettangolo in n pezzi secondo una proporzione data
    """
    x,y = point
    direction = direction[0]
    proportion = proportion if iterable(proportion) else array([proportion,1.-proportion])
    if sum(proportion)<1:
        proportion = r_[proportion,1.-sum(proportion)]
    left = r_[0,cumsum(proportion)]
    left /= left[-1]*1.
    L = len(left)
    gap_w = gap#*width
    gap_h = gap#*height
    size = 1. + gap*(L-2)
    #size=1.
    if  direction == 'h':
        #return [ ((x,y+height*left[idx]+gap_h*(0<idx<L-1)),width,height*proportion[idx]-gap_h-gap_h*(0<idx<L-2)) for idx in range(L-1)]
        sol = []
        for idx in range(L-1):
            new_y = y+(height*left[idx]+gap_h*idx)/size
            new_h = height*proportion[idx]/size
            sol.append(((x,new_y),width,new_h))
        return sol
        #return [ ((x,(y+height*left[idx]+gap_h*idx)/size),width,height*proportion[idx]) for idx in range(L-1)]
    elif direction == 'v':
        #return [ ((x+width*left[idx]+gap_w*(0<idx<L-1),y),width*proportion[idx]-gap_w-gap_w*(0<idx<L-2),height) for idx in range(L-1)]
        sol = []
        for idx in range(L-1):
            new_x = x+(width*left[idx]+gap_w*idx)/size
            new_w = width*proportion[idx]/size
            sol.append(((new_x,y),new_w,height))
        return sol
        #return [ (((x+width*left[idx]+gap_w*idx)/size,y),width*proportion[idx],height) for idx in range(L-1)]
    else:
        raise ValueError("direction of division should be 'vertical' or 'horizontal'")
        


def MosaicDivision(counted,direction='v',gap=0.005):
    """
    given a dictionary of counting for each category, it return the Rectangles
    Bounding boxes and the relative axis ticks
    """
    #preparazione dei valori da utilizzare
    ticks_tot = []
    rects2 = { ('total',):((0,0),1,1) }
    
    #categories = [ list(OrderedSet(i)) for i in zip(*(counted.keys())) ]
    #uso l'orderedDict come un orderedSet
    categories = [ list(OrderedDict([(j,None) for j in i])) for i in zip(*(counted.keys())) ]
    
    #inizio il ciclo per le varie categorie
    #divido ricorsivamente i vari rettangoli

    def recursive_split(rect_key,rect_coords,category_idx,split_dir,gap):
        """
        given a key of the boxes and the data to analyze,
        split the key into several keys stratificated by the given
        category in the assigned direction
        """
        ticks = []
        category = categories[category_idx]
        chiave=rect_key
        divisione = OrderedDict()
        for tipo in category:
            divisione[tipo]=0.
            for k,v in counted.items():
                if k[len(rect_key)-1]!=tipo:
                    continue 
                if not all( k[k1]==v1 for k1,v1 in enumerate(rect_key[1:])):
                    continue
                divisione[tipo]+=v
        totali = 1.*sum(divisione.values())
        if totali: #check for empty categories
            divisione = OrderedDict( (k,v/totali) for k,v in divisione.items() )
        else:
            divisione = OrderedDict( (k,0.) for k,v in divisione.items() )
        prop = divisione.values()
        div_keys = divisione.keys()
        new_rects = split_rect(*rect_coords,proportion=prop,direction=split_dir,gap=gap)
        divisi = OrderedDict( (chiave+(k,),v) for k,v in zip(div_keys,new_rects))
        d = (split_dir == 'h')
        ticks = [ (k,O[d]+0.5*[h,w][d]) for k,(O,h,w) in zip(div_keys,new_rects) ]
        return divisi,zip(*ticks)
   
    for cat in range(len(categories)):
        tipi = categories[cat]
        chiavi = rects2.keys()
        res = OrderedDict()
        #per ogni categoria pesco le chiavi dal dizionario dei rettangoli
        #le divido in base alle categorie presenti e le reinserisco
        # in un nuovo dizionario
        temp_ticks = []
        for chiave,coords in rects2.items():
            
            partial,ticks = recursive_split(chiave,coords,cat,direction,gap/2.**cat)
            res.update(partial)
            temp_ticks.append(ticks)
            #if len(ticks_tot)<=cat:
            #    ticks_tot.append(ticks)
        ticks_tot.append(temp_ticks[0 if cat<2 else -1])
        rects2=res
        direction = 'h' if direction=='v' else 'v'
        #level+=1
        
    rects2 = { k[1:]:v for k,v in rects2.items() }
    return rects2,ticks_tot,categories



def MosaicPlot(data,ax=None,direction='v',gap=0.005,decorator=None):
    """
    it create the actual plot:
        takes the set of boxes of the division with the ticks
        use the decorator to generate the patches
        draw the patches
        draw the appropriate ticks on the plot
    """
    if ax is None:
        ax=plt.gca()
    data = OrderedDict( (k,v) for k,v in sorted(data.items()) )
    rects,ticks,categories = MosaicDivision(data,direction=direction,gap=gap)

    if decorator is None:
        L = [1.*len(cat) for cat in categories]
        props = [ np.linspace(0,1,l+2)[1:-1] for l in L ]
        if len(L)==4:
            props[3]=[ '', 'x', '/', '\\', '|', '-', '+',  'o', 'O', '.', '*' ]
        def dec(cat):
            prop = [ props[k][categories[k].index(cat[k])] for k in range(len(cat)) ]
            hsv = [0., 0.4, 0.7]
            for idx,i in enumerate(prop[:3]):
                hsv[idx]=i
            hatch = prop[3] if len(prop)==4 else ''
            return dict( color=single_hsv_to_rgb(hsv), hatch=hatch )
        decorator = dec
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    for k,r in rects.items():
        ax.add_patch(plt.Rectangle(*r,**(decorator(k))))
    
    for idx,t in enumerate(ticks):
        for (lab,pos) in zip(*t):
            s = 0.02
            border= -s if idx<2 else 1+s
            valign= 'top' if idx<2 else 'baseline'
            halign= 'right' if idx<2 else 'left'
            x,y,v,h = (border,pos,'center',halign) if (direction =='v')!=(not idx%2) else (pos,border,valign,'center')
            size = ['xx-large','x-large','large','large','medium','medium','small','x-small'][idx]
            ax.text(x,y,lab,horizontalalignment = h, verticalalignment = v,size=size,rotation=0)

# <codecell>

import numpy as np
class WRGnumpy(object):
    def __init__(self, ensemble, weights=1):
        try:
            weights=list(weights)
        except TypeError:
            weights=[weights]*len(ensemble)
        assert len(weights)==len(ensemble)
        self.totals = cumsum(weights)
        self.ensemble = np.array(list(ensemble))

    def __call__(self, n, shape=(-1,), rand = np.random.rand, bisect = np.searchsorted):
        rnd = rand(n) * self.totals[-1]
        idx = bisect(self.totals, rnd)
        return self.ensemble[idx].reshape(shape)

# <codecell>

if __name__=='__main__':
    from random import choice
    from collections import Counter
    import pylab as plt
    L=500
    males = WRGnumpy(['male','female'],[2,1])(L)
    working  = WRGnumpy(['employment','educations','training','neet'])(L)
    married  = WRGnumpy(['married','coupled','single'])(L)
    health = WRGnumpy(['healthy','ill'],[2,1])(L)
    data = zip(males,working,married,health)
    for k,v in Counter( (d[0],d[1]) for d in data ).items():
        if 'neet' in k:
            print k,v

# <codecell>

if __name__=='__main__':
    def dec(cat):
        if  'neet' in cat:
            if 'female' in cat:
                return dict(color='r')
            else:
                return dict(color='g')
        else:
            return dict(color='b')

    f,ax = subplots(1,figsize=(6,6))
    MosaicPlot(Counter( (d[0],d[1]) for d in data ),ax=ax,gap=0.02,direction='h',decorator=dec)
    

# <codecell>

if __name__=='__main__':
    f,ax = subplots(1,figsize=(7,7))
    gap = 0.03
    MosaicPlot(Counter( d for d in data ),ax=ax,gap=gap,direction='v')

# <codecell>


# <codecell>

import matplotlib
def axes_subaxes(bounds,**kwargs):
    ax = kwargs.pop('ax',plt.gca())
    fig = ax.figure
    Bbox = matplotlib.transforms.Bbox.from_bounds(*bounds) 
    trans = ax.transAxes + fig.transFigure.inverted() 
    new_bounds = matplotlib.transforms.TransformedBbox(Bbox, trans).bounds 
    axins = fig.add_axes(new_bounds,**kwargs) 
    return axins

if __name__=='__main__':
    fig,ax = pylab.subplots(1,figsize=(4,4))
    inax = axes_subaxes([0.2, 0.45, .5, .5],sharex=ax,sharey=ax)
    ax.plot([1,2,3],[1,4,9])
    inax.plot([1,2,3],[1,8,27])

# <codecell>


