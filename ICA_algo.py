#"Indepdent Component Analysis Code"
#(for CSE 5350)
#Written by Corey Warren II And Pablo (...)

#Group Roster:
#Corey, Pablo, Arevalo, Ericka

#Objective of Program:
#Generate multiple different types of wave signals
#Combine them into one signal
#Put them into the Independent Component Analyzer (Our chosen Project Type)
#The Independent Component Analyzer will separate them into different signals again
#Show proof of the independent signals being separated

#Expected Difficulty Level: 3/5 stars

#ICA Algorithm Steps:
#1) Center x by subtracting the mean
#2) Whiten x with a whitening function
#3) Choose some random initial value for the de-mixing matrix W
    #-start_loop-#
    #4) Calculate the new value for W
    #5) Normalize W
    #6) Check whether the algorithm has converged. If it hasn't, loop, starting from step 4
    #--end_loop--#
#7) Take the dot product of W and X to get the independent source signals

import numpy as np
np.random.seed(0)
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import seaborn as sns
sns.set( rc={'figure.figsize':(11.7,8.27)} )

# G is our main function

def g(x):
    return np.tanh(x)

# G_der is G' or "G prime"
# G' is the derivative of G

def g_der(x):
    return 1 - g(x) * g(x)
    

# "center" function
# centers the input by subracting the mean from X
# X is our array of values

def center(X):
    X = np.array(X)
    
    mean = X.mean(axis = 1, keepdims = True)
    
    return X - mean
    
# whitening function
# the purpose of this is to transform matrix x into matrix x_whiten
# x_whiten is transformed such that x_whiten has identity covariance matrix.

def whitening(X):
    cov = np.cov(X)
    
    #return to d, E these values:
    #d will receive the eigenvalues in ascending order, each repeated according to its multiplicity
    #multiplicity of an eigenvalue is the number of times it appears as a root of
    #the characteristic polynomial (ex.: the polynomial whose roots are the eigenvalues of a matrix)
    #polynomial being an expression such as: P(x) = 2x^2 - 6x + 12
    
    #meanwhile, E receives the normalized eigenvectors corresponding to the eigenvalue
    # d[i], for each column of E, which can be expressed as E[:,i].
    
    d, E = np.linalg.eigh(cov)
    
    #D shall be a diagonal matrix of eigenvalues (every lambda is an eigenvalue of the covariance matrix)
    D = np.diag(d)
    
    D_inv = np.sqrt(np.linalg.inv(D))
    
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    
    return X_whiten
    
# update the de-mixing matrix W

def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    
    w_new /= np.sqrt((w_new ** 2).sum() )
    
    return w_new
    
# Main method
# calls the preprocessing functions
# initializes w to some random set of values and updates w iteratively
# convergence of the algorithm is judged by whenever W multiplied by its transpose is approximately equal to 1.
# after computing the optimal value of W for each component, 
# take the dot product of the new matrix and the signal x
# this will give you the independent sources

def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    
    X = whitening(X)
        
    components_nr = X.shape[0]

    #W is an empty matrix that has a pre-defined shape.
    #It is shaped so that it is ready to take in the new 
    #data that w will have in this little loop.
    #It's just a way to get the data out from the loop.
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)

    for i in range(components_nr):
        
        w = np.random.rand(components_nr)
        
        for j in range(iterations):
            
            #keep calculating a new de-mixing matrix W until the algorithm is converged
            #or until the max number of iterations has been reached.
            w_new = calculate_new_w(w, X)
            
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            
            #np.abs(...) means take the absolute value of "...".
            #(w * w_new).sum() returns the sum of those matrix elements, along some given axis
            #since we specify no axis, it adds all of the elements in the given matrices.
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            
            #update w
            w = w_new
            
            #if distance between old w and new w is less than 1e-5
            #we can confidently stop our ICA knowing that
            #our results are good enough...
            if distance < tolerance:
                break
                
        W[i, :] = w
        
    S = np.dot(W, X)
    
    return S

# Data visualization and plotting
# This will display the original, mixed, and predicted signals
# X is the Mixture
# original_sources is the collection of original sources (s1, s2, and s3)
# S is the dot product of W and x, and is a collection of the
#   obtained source signals

def plot_mixture_sources_predictions(X, original_sources, S):
    fig = plt.figure()
    
    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("mixtures")
    
    plt.subplot(3, 1, 2)
    for s in original_sources:
        plt.plot(s)
    plt.title("real sources")
    
    plt.subplot(3,1,3)
    for s in S:
        plt.plot(s)
    plt.title("predicted sources")
    
    fig.tight_layout()
    plt.show()
    
# Artificially mix the different source signals

def mix_sources(mixtures, apply_noise=False):
    for i in range(len(mixtures)):
    
        max_val = np.max(mixtures[i])
        
        if max_val > 1 or np.min(mixtures[i]) < 1:
        
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5
            
    X = np.c_[[mix for mix in mixtures]]
    
    if apply_noise:
    
        X += 0.02 * np.random.normal(size = X.shape)
        
    return x

    
    
def main():
    print("Running... Please wait...")
    print()
    
    #Adding more samples makes the resulting wave signals smoother
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    
    #Create the wave signals
    s1 = np.sin(2 * time) #sinusoidal
    s2 = np.sign(np.sin(3 * time)) # square signal
    s3 = signal.sawtooth(2 * np.pi * time) #saw tooth signal

    print("s1 = \n",s1)
    print()
    print("s2 = \n",s2)
    print()
    print("s3 = \n",s3)
    print()
    # Compute dot product of matrix A and the signals
    # this will give us a combination of all three
    # Then, use ICA to separate the mixed signal into the three
    # original source signals

    #np.c_ is a special type of np.r_
    #np.r_ Translates slice objects to concatenation along the first axis.
    #np.c_ is a variation of that adds column vectors to each other, in first-to-last order.
    #therefore, np.c_ [s1, s2, s3] means add the signals as columns to the matrix X!
    X = np.c_[s1, s2, s3]
    
    print("X = ", X)
    print()
    
    #this A matrix is mostly arbitrary. Its only purpose is to mix up the X matrix data.
    A = np.array(([[0.1, 0.1, 0.1], [1, 2, 1.0], [1, 1, 2.0]]))
    
    #np.dot(X, A.T) means take the dot product of X and A.T (the transpose of A)
    X = np.dot (X, A.T)
    print("X*A.T = ", X)
    print()
    
    #X.T means get the transpose matrix of X
    X = X.T
    print("X = X.T => ", X)
    print()
    
    
    #ica is our custom function for this problem
    #1000 iterations is more than enough to de-mix the columns
    
    S = ica(X, iterations = 1000)
    #s1, s2, and s3, in that order, are predicted to be...
    #s1 = sin wave, s2 = square wave, s3 = sawtooth wave.
    #Please read the report for what actually happened when we ran the code!
    #The results surprised us!
    
    #plot all of our data (all-in-one function)
    plot_mixture_sources_predictions(X, [s1, s2, s3], S)


#run the "main" function (simple Python jargon)
if __name__=="__main__":
    main()