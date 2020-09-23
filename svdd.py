import numpy as np
import xpress as xp

class SVDD:
    def __init__(self, **params):
        """
            The class implement the Support Vector Data Description method for novelty detection as in paper https://www.researchgate.net/publication/226109293_Support_Vector_Data_Description.
            The class make use of the solver FICO Express and it is composed by the following method:
                - solve: to model the svdd problem
                - select_support_vectors: method that randomply pick a point that can be considered as a good support vector candidate
                - outliers: returns the outliers of the dataset X
                - count_outliers: returns the number of outiers in X
                - kernel: define the kernel in the dual problem
        """
        
        # DUAL PROBLEM
        self.p = xp.problem()
        
        self.C = params['C']
        self.params = params
        #self.l = params['l']
        
        self.opt = []
        self.c0 = []
        self.support_vectors = []

        
    def solve(self, X):
        """
            Implementation of the dual of SVDD problem
            
            Params:
                - X, the dataset without labels
            Returns:
                None
        """
        
        m = X.shape[0]
        
        # Define Variables of the problem
        alphas = [xp.var(lb=0, ub=self.C) for i in range(m)]
        self.p.addVariable(alphas)

        # Define Constraints
        # constr1 = [alpha <= C for alpha in alphas] 
        # p.addConstraint(constr1)
        # already defined in the definition of the variable as a upper bound

        # Define objective function
        obj1 = xp.Sum(alphas[i]*np.sum(self.kernel(X[i].reshape(1, -1), X[i].reshape(1, -1))[0][0]) for i in range(m))
        obj2 = xp.Sum(alphas[i]*alphas[j]*np.sum(self.kernel(X[i].reshape(1, -1), X[j].reshape(1, -1))[0][0]) for i in range(m) for j in range(m))

        obj = obj1 - obj2
        self.p.setObjective(obj, sense=xp.maximize) #set maximization problem
        # solve the problem
        self.p.solve()
        
        self.opt = np.array(self.p.getSolution(alphas)) #save the values of alphas
        self.c0 = np.sum(self.opt.reshape(-1, 1)*X, axis=0)
        
        indices_support_vectors = self.select_support_vectors() #choose a point as support vector
        self.support_vectors = X[indices_support_vectors]
        
    def select_support_vectors(self, eps=None):
        """
            The following function selects randomply a point as a support vector.
            The condition to be a support vector is that his coefficient is lower that C and greater than 0.
            
            Params:
                - eps, a small constant
            Returns:
                - The index of the support vector in X 
        """
        if eps is None:
            eps = 0.01*self.C

        isv = np.where((self.opt<(self.C-eps))&(self.opt>eps))[0][0]
        return isv

    def outliers(self, eps=None):
        """
            The function select which points has to be considered as outliers.
            If the point is an outlier than the value returned is equal to -1, if not it corresponds to 1.
            
            Params:
                - eps, a small constant
            Returns:
                - The list of outliers
        """
        if eps is None:
            eps = 0.01*self.C

        outliers = []
        for alpha in self.opt:
            if alpha >= self.C - eps:
                outliers.append(-1)
            else:
                outliers.append(1)

        return np.array(outliers)

    def count_outliers(self, eps=None):
        """
            The following function counts the number of outliers that are present in X.
            
            Params:
                - eps, a small constant
            Returns:
                - The number of outliers in X
        """
        if eps is None:
            eps = 0.01*self.C
        n_outliers = 0

        for alpha in self.opt:
            if alpha >= self.C - eps:
                n_outliers+=1
        return n_outliers

    
    def kernel(self, x1, x2):
        """
            The following function returns the kernel choosen by the user.
            
            Params:
                - x1, x2, the data points to project
                - k, the kernel type among 'linear', 'rbf', 'poly'
            Returns:
                - The number of outliers in X
        """
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.metrics.pairwise import polynomial_kernel
        
        k = self.params['k']
        if k == 'rbf':
            rbf = RBF(length_scale=self.params['l'])
            return rbf(x1, x2)
        
        if k == 'linear':
            return np.dot(x1, x2.T)
        
        if k == 'poly':
            return polynomial_kernel(x1, x2, degree=self.params['degree'])