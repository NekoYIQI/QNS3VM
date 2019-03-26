############################################################################################
# QN-S3VM BFGS optimizer for semi-supervised support vector machines. 
#
# This implementation provides both a L-BFGS optimization scheme
# for semi-supvised support vector machines. Details can be found in:
#
#   F. Gieseke, A. Airola, T. Pahikkala, O. Kramer, Sparse quasi-
#   Newton optimization for semi-supervised support vector ma-
#   chines, in: Proc. of the 1st Int. Conf. on Pattern Recognition
#   Applications and Methods, 2012, pp. 45-54.
#
# Version: 0.1 (September, 2012)
#
# Bugs: Please send any bugs to "f DOT gieseke AT uni-oldenburg.de"
#
#
# Copyright (C) 2012  Fabian Gieseke, Antti Airola, Tapio Pahikkala, Oliver Kramer
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# 
# INSTALLATION and DEPENDENCIES
#
# The module should work out of the box, given Python and Numpy (http://numpy.scipy.org/)
# and Scipy (http://scipy.org/) installed correctly. 
# 
# We have tested the code on Ubuntu 12.04 (32 Bit) with Python 2.7.3, Numpy 1.6.1, 
# and Scipy 0.9.0. Installing these packages on a Ubuntu- or Debian-based systems 
# can be done via "sudo apt-get install python python-numpy python-scipy".
#
#
# RUNNING THE EXAMPLES
# 
# For a description of the data sets, see the paper mentioned above and the references 
# therein. Running the command "python qns3vm.py" should yield an output similar to:
# 
# Sparse text data set instance
# Number of labeled patterns:  48
# Number of unlabeled patterns:  924
# Number of test patterns:  974
# Time needed to compute the model:  0.775886058807  seconds
# Classification error of QN-S3VM:  0.0667351129363
#
# Dense gaussian data set instance
# Number of labeled patterns:  25
# Number of unlabeled patterns:  225
# Number of test patterns:  250
# Time needed to compute the model:  0.464584112167  seconds
# Classification error of QN-S3VM:  0.012
#
# Dense moons data set instance
# Number of labeled patterns:  5
# Number of unlabeled patterns:  495
# Number of test patterns:  500
# Time needed to compute the model:  0.69714307785  seconds
# Classification error of QN-S3VM:  0.0


