import argparse
import time
import numpy as np 

def d(x,y):
    '''
    Given x and y where each is an np arrays of size (dim,1), compute L2 distance between them
    '''
    
    ## ADD CODE HERE ##
    return np.linalg.norm(x-y)**2
    # pass


def pairwise_similarity_looped(X):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given
    '''
    ## STEP 1 - Initialize K to zeros  - ADD CODE TO COMPUTE n1, n2 ##
    
    n = X.shape[0]
    K = np.zeros((n,n))

    ## STEP 2 - Loop and compute  -- COMPLETE THE LOOP BELOW ##

    for i in range(n):
        for j in range (n):
            ## ADD CODE HERE ##
            K[i,j] = d(X[i],X[j])

    return K 


def pairwise_similarity_vec(X):
    '''
    Given X, Y where each is an np array of size (num_points_1,dim) and (num_points_2,dim), 
    return K, an array having size (num_points_1,num_points_2) according to the problem given

    This problem can be simplified in the following way - 
    Each entry in K has three terms (as seen in problem 2.1 (a)).
    Hence, first  computethe norm for all points, reshape it suitably,
    then compute the dot product.
    All of these can be done by using on the *, @, sum(), and transpose operators.
    '''


    ## STEP 1 - COMPUTE THE PR
    ## ADD CODE TO COMPUTE K ##
    norm_x = (X*X).sum(axis=1).reshape((-1,1))
    K = norm_x + norm_x.T - 2 * np.dot(X,X.T)

    return K
    # pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num', type=int, default=1000,
                    help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                    help='Seed for random generator')
    parser.add_argument('--dim', type=float, default=2,
                    help='Lambda parameter for the distribution')

    args = parser.parse_args()

    np.random.seed(args.seed)

    X = np.random.normal(0.,1.,size=(args.num,args.dim))
    # Y = np.random.normal(1.,1.,size=(args.num,args.dim))

    t1 = time.time()
    K_loop = pairwise_similarity_looped(X)
    t2 = time.time()
    K_vec  = pairwise_similarity_vec(X)
    t3 = time.time()

    print(K_loop)
    print(K_vec)
    assert np.allclose(K_loop,K_vec)   # Checking that the two computations match

    np.savetxt("problem_2_loop.txt",K_loop)
    np.savetxt("problem_2_vec.txt",K_vec)
    print("Vectorized time : {}, Loop time : {}".format(t3-t2,t2-t1))
    