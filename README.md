
# K2 Self-flat-fielding, a la Vanderburg et al.

This code is modelled after the description of the K2 self-flat-fielding (K2SFF) 
algorithm described in (Vanderburg & Johnson 2014)[http://iopscience.iop.org/article/10.1086/678764] 
and (Vanderburg et al. 2016)[http://iopscience.iop.org/article/10.3847/0067-0049/222/1/14/meta].
Brightness of stars is recorded as a function of time in a light curve, and 
for the (K2 mission)[https://www.nasa.gov/mission_pages/kepler/main/index.html],
the combination of a rolling spacecraft and differing pixel sensitivities
caused systematic variations in the lightcurve.  As Vanderburg & Johnson 
realized, since these variations are due to the position of a star on the
instrument's detector, it is possible to correlate these variations with
detector position and thus remove them.

