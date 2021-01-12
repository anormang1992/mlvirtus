import numpy as np

class LinearRegression(): 
	def __init__( self, alpha=.01, iters=500 ): 
		self.alpha = alpha 
		self.iters = iters

	def Feature_Normalize(self,X):
		self.X = X
		self.mu = np.mean(self.X,axis=0)
		self.sigma = np.std(self.X,axis=0)
		self.X = (self.X-self.mu)/self.sigma

		return self.mu, self.sigma, self.X

	def Linear_Regression_Cost(self,X,y, theta):
		self.X = X
		self.y = y
		self.theta = theta

		self.m = len(self.y)

		#calculate linear regression cost using the cost function
		self.h =self.X.dot(self.theta)
		self.error = (self.h-self.y)**2
		self.cost = (1/(2*self.m)) * np.sum(self.error)

		return self.cost

	def Gradient_Descent(self, X, y, theta):
		self.X = X
		self.y = y
		self.theta = theta

		self.m = len(self.y)
		self.cost_history = []

		for i in range(self.iters):
			self.h =self.X.dot(self.theta)
			self.error = np.dot(X.transpose(),(self.h-self.y))
			self.theta = self.theta - (self.alpha*(1/self.m)*self.error)
			self.cost_history.append(self.Linear_Regression_Cost(self.X,self.y,self.theta))

		return self.theta, self.cost_history

	def Prediction(self,X,theta):
		self.X = X
		self.theta = theta

		self.predictions = self.theta.T.dot(self.X)

		return self.predictions[0]













class LogisticRegression():
	def __init__(self, alpha=.01, Lambda=1): 
		self.alpha = alpha 
		self.Lambda = Lambda

	def Feature_Normalize(self,X):
		self.X = X
		self.mu = np.mean(self.X,axis=0)
		self.sigma = np.std(self.X,axis=0)
		self.X = (self.X-self.mu)/self.sigma

		return self.mu, self.sigma, self.X

	def Feature_Mapping(self,x1,x2,degree):
		self.x1 = x1
		self.x2= x2
		self.degree = degree

		#add x intercept to matrix containing the newly mapped features
		self.output = np.ones((x1.shape[0], 1))
		for i in range(1,self.degree+1):
			for j in range(0,i+1):
				self.feature1 = self.x1 ** (i-j)
				self.feature2 = self.x2 ** (j)
				self.newFeatures  = (self.feature1 * self.feature2).reshape( self.feature1.shape[0], 1 ) 
				self.output   = np.hstack(( self.output, self.newFeatures ))

		return self.output

	def sigmoid(self, z):
		self.z = z
		return 1/(1+np.exp(-z))

	def Logistic_Regression_Cost(self,X,y, theta):
		self.X = X
		self.y = y
		self.theta = theta

		self.m = len(self.y)
	
		#calculate logistic regression cost using the cost function
		self.h = self.sigmoid(self.X @ self.theta)
		self.error = (-self.y * np.log(self.h)) - ((1-self.y)*np.log(1-self.h))
		self.cost = 1/self.m * np.sum(self.error)
		self.reg = self.Lambda/(2*self.m) * np.sum(self.theta[1:]**2)
		self.regCost = (self.cost+self.reg)

		#calculate gradients
		self.grad0= 1/self.m * (self.X.T @ (self.h - self.y))[0]
		self.grad1 = 1/self.m * (self.X.T @ (self.h - self.y))[1:] + (self.Lambda/self.m)* self.theta[1:]
		self.gradients = np.vstack((self.grad0[:,np.newaxis],self.grad1))		


		return self.regCost, self.gradients


	def Gradient_Descent(self,X,y,theta,iters=1000):
		self.X = X
		self.y = y
		self.theta = theta
		self.iters = iters


		self.cost_history=[]

		for i in range(self.iters):
			self.cost, self.gradients = self.Logistic_Regression_Cost(self.X,self.y,self.theta)
			self.theta = self.theta - (self.alpha * self.gradients)
			self.cost_history.append(self.cost)

		return self.theta, self.cost_history

