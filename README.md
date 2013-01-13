This is a closet repository to keep track of the utility function that I've written for myself for data plotting.
It's still full of non clean function, but it's already rich of utilities.

The source ipython notebook can bee seen at the address:

http://nbviewer.ipython.org/urls/raw.github.com/EnricoGiampieri/dataplot/master/statplot.ipynb

or the python script can be imported and used.

a brief list of the most relevant functions include:
  
  * **violin_plot**: to generate violin plots up to two different dataset (for example, males Vs females for variuos category)
  * **fillboxplot**: an enhanced version of boxplot instead of the basic matplotlib one
  * **desaturize**: change the color of lines, patches and background of a pylab axes to be more web friendly
  * **repatch_set**: apply a colormap gradient to a whole set of patches
  * **patch_gradient**: same as before, but with an arbitrary function
  * **gradient_patchset**: as before, but applyied to a whole set of patches
  * **plotline**: similar to R abline, plot the regression line of a dataset
  * **plotfunc**: as above, but plot a generic function given by a lambda
  * **plot_lambda**: the most generic one, infere the number of dimensions of input and output of a lambda and create the (more or less) correct plot
  * **MosaicPlot**: create a mosaic plot given a Counter object of a dataset
  * **axes_subaxes**: create a subaxes in a pylab axes in a more natural way
  