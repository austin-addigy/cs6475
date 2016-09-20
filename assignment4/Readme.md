# Assignment 4: Gradients & Edges

## Synopsis

This assignment explores treating images as functions, and the connection between image gradients and edges. In the first part of this assignment you will write several functions to compute image gradients and apply filter kernels. (View lectures 2-05, 2-06 and 2-07 for a refresher on this). 

In the second part of the assignment, you will use the ideas presented in lecture to perform edge detection on an image of your own. You will then describe the different things you attempted to find edges in the image, and include your output in the PDF.

Detailed [instructions](https://drive.google.com/open?id=1NyLqV16nX-AFmYsPmeTmVMvx0Jr06yx2Z6GqBy-JLmE) and the [report template](https://drive.google.com/open?id=1JzeL4cHk_gn9oUTNRrp2QYJ7_bjqQw3dok71fWKepp4) are available on Google drive.

## Note about vectorized operations and array slicing

Numpy, Scipy, OpenCV, and other libaries includes a wealth of functions that broadcast operations on arrays (commonly called vectorization). Using them makes it possible to get these assignments to work with no real understanding of what's being performed, and that understanding is crucial for this class. The docstring notes where functions have been disallowed for this assignment, and the autograder will return a score of zero for any function that uses disabled library functions. If the autograder accepts your code, then you haven't used any banned function calls.

However, these restrictions do not apply to basic array slicing, which can often perform row/column/array at a time operations.  If you are not familiar with slicing, it is not critical for this assignment, as you can implement everything with nested for loops.  But if you have the time, it will pay dividends later to get comfortable with it.
