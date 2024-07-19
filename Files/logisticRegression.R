# This is a program that executes a simple Logistic Regression algorithm.
#
# Functions
#  standardize(v)
#    Returns standardized vector
#
#  computeProb(obs, beta)
#    Returns probability of positive class assignment
#
#  negLogLik(beta)
#    Returns negative log-likelihood of data given features


# loading dependencies
library(pROC)


# clear memory
rm(list = ls())


# reproducibility
set.seed(1)


# import training data
train = read.table(
  file = "train.txt",
  header = TRUE,
  sep = "\t"
)


# import test data
test = read.table(
  file = "test.txt",
  header = TRUE,
  sep = "\t"
)


standardize = function(v) {
  "
  Returns standardized vector
  
  Parameters:
    v: vector of values
    
  Return:
    vScaled: scaled vector of values
    
  Assumptions:
    v is a vector of 1 or more elements
  "
  if (length(v) == 0) {
    print("You passed an empty vector...")
    return()
  }
  
  avg = mean(v)
  sd = sqrt(var(v))
  
  vScaled = c()
  for (i in 1:length(v)) {
    vScaled[i] = (v[i] - avg) / sd
  }
  
  return(vScaled)
}


computeProb = function(obs, beta) {
  "
  Returns probability of positive class assignment
  
  Parameters:
    obs: vector of feature values for observation
    beta: vector of maximum likelihood estimates for model parameters
    
  Return:
    pi: probability of positive class assignment given observation
    
  Assumptions:
    obs is a vector of 1 or more elements
    length of obs is one less than that of beta
  "
  if (length(obs) == 0 || length(beta) == 0) {
    print("You passed an empty vector")
  }
  
  sum = beta[1]
  for (i in 1:length(obs)) {
    sum = sum + (obs[i] * beta[i + 1])
  }
  
  pi = 1 / (1 + exp(-sum))
  
  return(pi)
}


negLogLik = function(beta) {
  "
  Returns negative log-likelihood of data given features
  
  Parameters:
    beta: vector of maximum likelihood estimates for model parameters
    
  Return:
    negLogLik: negative log-likelihood of data given features
    
  Assumptions:
    beta is vector of 1 or more elements
    length of beta is one more than number of features in data
  "
  if (length(beta) == 0) {
    print("You passed an empty vector...")
    return()
  }
  
  negLogLik = 0
  for (i in 1:nrow(train)) {
    obs = as.vector(unlist(train[i,]))
    class = obs[1]
    negLogLik = 
      negLogLik + (class * log(computeProb(obs[2:length(obs)], beta))) +
      ((1 - class) * log((1 - computeProb(obs[2:length(obs)], beta))))
  }
  
  return(-negLogLik)
}


# feature subsetting
setOne = c(1, sample(x = 2:19, size = 10))
setTwo = c(1, sample(x = 2:19, size = 10))
setThree = c(1, sample(x = 2:19, size = 10))
setFour = c(1, sample(x = 2:19, size = 10))
setFive = c(1, sample(x = 2:19, size = 10))
setSix = 1:19


subset = 100


# drawing random sample
trainSubset = sample(x = seq(from = 1, to = 10000), size = subset)
testSubset = sample(x = seq(from = 1, to = 10000), size = subset)


features = setSix

train = train[trainSubset, features]
test = test[testSubset, features]
numBeta = length(features) + 1


# standardizing data
for (i in 2:ncol(train)) {
  train[,i] = standardize(train[,i])
  test[,i] = standardize(test[,i])
}


# double check Tuesday with Majoros and others if Nelder-Mead is fine instead
# of BFGS
# training model
beta = rep(x = 1, times = numBeta)
optimizedBeta = optim(beta, negLogLik, method = "Nelder-Mead")$par


# testing model
probabilities = c()
for (i in 1:subset){
  obs = as.vector(unlist(test[i,]))
  prob = computeProb(obs[2:length(obs)], optimizedBeta)
  probabilities[i] = prob
}

numCorrectAcc = 0
truePositive = 0
falseNegative = 0
falsePositive = 0

for (i in 1:nrow(test)) {
  obs = as.vector(unlist(test[i,]))
  prob = probabilities[i]
  cutoff = 0.51
  
  if ((prob >= cutoff && obs[1] == 1) || (prob < cutoff && obs[1] == 0)) {
    numCorrectAcc = numCorrectAcc + 1
  }
  
  if (prob >= cutoff && obs[1] == 1) {
    truePositive = truePositive + 1
  }
  
  if (prob < cutoff && obs[1] == 1) {
    falseNegative = falseNegative + 1
  }
  
  if (prob >= cutoff && obs[1] == 0) {
    falsePositive = falsePositive + 1
  }
}

accuracy = numCorrectAcc / nrow(test)
sensitivity = truePositive / (truePositive + falseNegative)
specificity = truePositive / (truePositive + falsePositive)


# plotting ROC curve and calculating AUC
pROC = roc(
  test[,1],
  probabilities
)

plot(
  tpr ~ fpr, 
  coords(
    pROC,
    "all",
    ret=c("tpr","fpr"),
    transpose=FALSE
  ), 
  type="l",
  xlab = "FPR",
  ylab = "TPR"
)

auc(pROC)