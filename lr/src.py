import math

class LogisticRegression:
	
	def sigmoid(self, x):
		return 1.0 / (1.0 + math.exp(-x))

	def hypothesis(self, W, X):
		y = sum([W[i]*X[i] for i in xrange(len(W))])
		return self.sigmoid(y)

	def cost_function(self, X, Y, W):
		err, m = 0, len(Y)
		for i in xrange(m):
			hi = self.hypothesis(W, X[i])
			err += Y[i] * math.log(hi) + (1-Y[i]) * math.log(1-hi)
		return -err/m

	def cost_function_derivative(self, X, Y, W, j, alpha):
		err, m = 0, len(Y)
		for i in xrange(m):
			hi = self.hypothesis(W, X[i])
			err += (hi - Y[i]) * X[i][j]
		return err*alpha/m

	def gradient_descent(self, X, Y, W, alpha):
		_W, m = [], len(Y)
		for j in xrange(len(W)):
			d = self.cost_function_derivative(X, Y, W, j, alpha)
			_W.append(W[j] - d)
		return _W

	def fit(self, X, Y, W, alpha, n_iters, n_print = 10):
		m, n = len(Y), n_iters/n_print
		for i in xrange(n_iters):
			W = self.gradient_descent(X, Y, W, alpha)
			if i%n == 0 or i == n_iters-1:
				print 'iter #', i
				print 'weight is', W
				print 'cost is', self.cost_function(X, Y, W)
		return W

	def predict(self, X, W):
		return round(self.hypothesis(X, W))

	def score(self, X, Y, W):
		score, m = 0, len(Y)
		for i in xrange(m):
			label = self.predict(X[i], W)
			if label == Y[i]:
				score += 1
		return float(score)/m
