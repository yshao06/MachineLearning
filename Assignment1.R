rm(list=ls())

#####################
#### QUESTION 1 ####
#####################

### Predicting House Prices: The Regression Setting ###

### Install Package for further processing ###
installIfAbsentAndLoad <- function(neededVector) {
    for(thispackage in neededVector) {
        if( ! require(thispackage, character.only = T) )
        { install.packages(thispackage)}
        require(thispackage, character.only = T)
    }
}
# load required packages
needed  <-  c("class", "FNN")  # class contains the knn() function
installIfAbsentAndLoad(needed)

# create a dataframe from "HomePrices.txt"
hp <- read.csv(file = "HomePrices.txt", sep = "\t", stringsAsFactors = F)
n1 = nrow(hp)

# calculate the MSE of medv and print out
sum((hp$medv - mean(hp$medv))^2) / n1
mean((hp$med - mean(hp$med))^2)
# calculate variance of medv
var(hp$medv) * (n1-1) / n1
# scale and center the numeric and integer predictors for "nearest-neighbor" models
hp_adj <- data.frame(scale(hp[-13], center = T, scale = T), hp[13])
head(hp_adj)

### Calculate MSE for test set ###
# set the random seed to 5072
set.seed(5072)
trainprop1 <- 0.75
validateprop1 <- 0.15
testprop1 <- 0.1

### Create three random subsets ###
# create a vector of random integers of training size from the vector 1:n
train1 <- sample(n1, n1*trainprop1)
# create a vector of random integers of validate size
# that is different from the training vector 
validate1 <- sample(setdiff(1:n1, train1), n1*validateprop1)
# create a vector of random integers of test size
# that is different from both training and validate vector aove
test1 <- setdiff(setdiff(1:n1, train1), validate1)
# create the data frames using the indices created in the three vectors above
trainset1 <- hp_adj[train1, ]
validateset1 <- hp_adj[validate1, ]
testset1 <- hp_adj[test1, ]
# display the first row of each data frame
head(trainset1, 1)
head(validateset1, 1)
head(testset1, 1)

### Run the KNN Regression Models ###
# create the 6 data frames for KNN models
train1.x <- trainset1[-13]
validate1.x <- validateset1[-13]
test1.x <- testset1[-13]
train1.y <- trainset1$medv
validate1.y <- validateset1$medv
test1.y <- testset1$medv
# use knn.reg() function to predict median value of homes
k_list <- seq(1, 19, by = 2)
train1.MSE <- rep(0, length(k_list))
validate1.MSE <- rep(0, length(k_list))
for (i in k_list) {
    knn1.pred <- knn.reg(train1.x, validate1.x, train1.y, k = i)$pred
    validate1.MSE[(i+1)/2] <- sum((knn1.pred - validate1.y)^2)/length(knn1.pred)
    knn1.pred <- knn.reg(train1.x, train1.x, train1.y, k=i)$pred
    train1.MSE[(i+1)/2] <- sum((knn1.pred - train1.y)^2)/length(knn1.pred)
}
# plot MSE as function of flexibility for KNN regression
plot(NULL, NULL, type='n', xlim=c(length(k_list)*2-1, 1), ylim=c(0,max(c(validate1.MSE, train1.MSE))), xlab='Increasing Flexibility (Decreasing k)', ylab='Mean Squared Errors', main='MSEs as a Function of \n Flexibility for KNN Regression')
lines(seq(length(k_list)*2-1, 1, by = -2), validate1.MSE[length(validate1.MSE):1], type='b', col=2, pch=16)
lines(seq(length(k_list)*2-1, 1, by = -2), train1.MSE[length(train1.MSE):1], type='b', col=1, pch=16)
legend("topright", legend = c("Validation MSEs", "Training MSEs"), col=c(2, 1), cex=.75, pch=16, lty=1)
# print minimum validate/training MSE and corresponding k's
print(paste("Minimum validate set MSE occurred at k=", which.min(validate1.MSE)*2-1))
print(paste("Minimum validate MSE was", validate1.MSE[which.min(validate1.MSE)]))
print(paste("Minimum training set MSE occurred at k=", which.min(train1.MSE)*2-1))
print(paste("Minimum training MSE was", train1.MSE[which.min(train1.MSE)]))
# predict medv and calculate MSE for test set
knn1.pred <- knn.reg(train1.x, test1.x, train1.y, k = which.min(validate1.MSE)*2-1)$pred
test1.MSE <- sum((knn1.pred - test1.y)^2)/length(knn1.pred)
test1.MSE



#####################
#### QUESTION 2 ####
#####################

### Predicting Loan Repayment: The Classification Setting ###

# create a dataframe from "LoadData.csv"
ld <- read.csv(file = "LoanData.csv", sep = ",", stringsAsFactors = F)
# calculate the error rate if always predicting Yes
allYes <- table(ld$loan.repaid, rep("Yes", nrow(ld)))
print(paste("Test set error rate was ", allYes["No", ] / sum(allYes)))
# scale and center the numeric and integer predictors
ld_adj <- data.frame(scale(ld[-8], center = T, scale = T), ld[8])
n2 = nrow(ld)
head(ld)

# set the random seed to 5072
set.seed(5072)
trainprop2 = 0.75
validateprop2 = 0.15
testprop2 = 0.1

### Create three random subsets ###
# create a vector of random integers of training size from the vector 1:n
train2 <- sample(n2, n2*trainprop2)
# create a vector of random integers of validate size
# that is different from the training vector 
validate2 <- sample(setdiff(1:n2, train2), n2*validateprop2)
# create a vector of random integers of test size
# that is different from both training and validate vector aove
test2 <- setdiff(setdiff(1:n2, train2), validate2)
# create the data frames using the indices created in the three vectors above
trainset2 <- ld_adj[train2, ]
validateset2 <- ld_adj[validate2, ]
testset2 <- ld_adj[test2, ]
# display the first row of each data frame
head(trainset2, 1)
head(validateset2, 1)
head(testset2, 1)

### Run the KNN Classification ###
# create the 6 data frames for KNN
train2.x <- trainset2[-8]
validate2.x <- validateset2[-8]
test2.x <- testset2[-8]
train2.y <- trainset2$loan.repaid
validate2.y <- validateset2$loan.repaid
test2.y <- testset2$loan.repaid

# use knn() function to predict loan repayment
# k_list2 <- seq(1, 19, by = 2)
train2.error <- rep(0, length(k_list))
validate2.error <- rep(0, length(k_list))
for (i in k_list) {
    knn2.pred <- knn(train2.x, validate2.x, train2.y, k = i)
    validate2.error[(i+1)/2] <- mean(validate2.y != knn2.pred)
    knn2.pred <- knn(train2.x, train2.x, train2.y, k=i)
    train2.error[(i+1)/2] <- mean(train2.y != knn2.pred)
}
# plot error rate as function of flexibility for KNN classification
plot(NULL, NULL, type='n', xlim=c(length(k_list)*2-1, 1), ylim=c(0,max(c(validate2.error, train2.error))), xlab='Increasing Flexibility (Decreasing k)', ylab='Error Rates', main='Error Rates as a Function of \n Flexibility for KNN Classification')
lines(seq(length(k_list)*2-1, 1, by = -2), validate2.error[length(validate2.error):1], type='b', col=2, pch=16)
lines(seq(length(k_list)*2-1, 1, by = -2), train2.error[length(train2.error):1], type='b', col=1, pch=16)
legend("topleft", legend = c("Validation Error Rate", "Training Error Rate"), col=c(2, 1), cex=.75, pch=16, lty=1)
# print minimum validate/training error rate and corresponding k's
print(paste("Minimum validate set error rate occurred at k=", which.min(validate2.error)*2-1))
print(paste("Minimum validate error rate was", validate2.error[which.min(validate2.error)]))
print(paste("Minimum training set error rate occurred at k=", which.min(train2.error)*2-1))
print(paste("Minimum training error rate was", train2.error[which.min(train2.error)]))
# predict loan repayment and calculate error rate for test set
knn2.pred <- knn(train2.x, test2.x, train2.y, k = which.min(validate2.error)*2-1)
test2.error <- mean(test2.y != knn2.pred)
test2.error



#####################
#### QUESTION 3 ####
#####################

### Investigating the Variance of the knn.reg Model on the Home Pricing Dataset Used in Part 1 ###

set.seed(5072)
### Create three random subsets ###
# create a vector of random integers of training size from the vector 1:n
best.validate.MSE <- rep(0, 50)
best.test.MSE <- rep(0, 50)
for (j in 1:50) {
    # produce train/validate/test set for testing
    train3 <- sample(n1, n1*trainprop1)
    validate3 <- sample(setdiff(1:n1, train3), n1*validateprop1)
    test3 <- setdiff(setdiff(1:n1, train3), validate3)
    trainset3 <- hp_adj[train3, ]
    validateset3 <- hp_adj[validate3, ]
    testset3 <- hp_adj[test3, ]
    train3.x <- trainset3[-13]
    validate3.x <- validateset3[-13]
    test3.x <- testset3[-13]
    train3.y <- trainset3[13]
    validate3.y <- validateset3[13]
    test3.y <- testset3[13]
    # display the first row of each data frame
    # head(trainset3, 1)
    # head(validateset3, 1)
    # head(testset3, 1)
    
    # use knn.reg() function to predict median value of homes
    # k_list <- seq(1, 19, by = 2)
    validate3.MSE <- rep(0, length(k_list))
    for (i in k_list) {
        knn3.pred <- knn.reg(train3.x, validate3.x, train3.y, k = i)$pred
        validate3.MSE[(i+1)/2] <- sum((knn3.pred - validate3.y)^2)/length(knn3.pred)
    }
    best.validate.MSE[j] <- validate3.MSE[which.min(validate3.MSE)]
    knn3.pred <- knn.reg(train3.x, test3.x, train3.y, k = which.min(validate1.MSE)*2-1)$pred
    best.test.MSE[j] <- sum((knn3.pred - test3.y)^2)/length(knn1.pred)
}

mean(best.validate.MSE)
sd(best.validate.MSE)
mean(best.test.MSE)
sd(best.test.MSE)
### By repeating the randomization for 50 times, the mean of best validate MSE vs corresponding test MSE
### indicates that the variance of prediction in test dataset is significantly larger than that of validate dataset
### Therefore, conclude that the searching of best k valude through training & validate dataset
### does not leads to an accurate prediction in dataset

    
# plot Test and Best Validation MSEs for Many Prtitionings of the Data
plot(NULL, NULL, type='n', xlim=c(1, 50), ylim=c(0,max(c(best.validate.MSE, best.test.MSE))), xlab='Replication)', ylab='MSEs', main='Test and Best Validation MSEs for Many Partitionings of the Data')
lines(seq(1, 50), best.validate.MSE[1:50], type='b', col=2, pch=16)
abline(h=mean(best.validate.MSE), col = 2, lty = 2)
lines(seq(1, 50), best.test.MSE[1:50], type='b', col=1, pch=16)
abline(h=mean(best.test.MSE), col = 1, lty = 2)
legend("topright", legend = c("Validation MSEs", "Validation MSE mean", "Test MSEs", "Test MSE mean"), col=c(2, 2, 1, 1), cex=.75, pch=c(16, NA, 16, NA), lty=c(1, 2, 1, 2))



#####################
#### QUESTION 4 ####
#####################

### Predicting College Applications ###

app <- read.csv(file = "applications.train.csv", sep = ",", stringsAsFactors = F)
n4 = nrow(app)
head(app)

### Calculate MSE for test set ###
# set the random seed to 5072
set.seed(5072)
# trainprop4 <- 0.75
# validateprop4 <- 0.15
# testprop4 <- 0.1


### Create three random subsets ###
# create a vector of random integers of training size from the vector 1:n
# train4 <- sample(n4, n4*trainprop4)
# create a vector of random integers of validate size
# that is different from the training vector 
# validate4 <- sample(setdiff(1:n4, train4), n4*validateprop4)
# create a vector of random integers of test size
# that is different from both training and validate vector aove
# test4 <- setdiff(setdiff(1:n4, train4), validate4)
# create the data frames using the indices created in the three vectors above
# app_adj <- app[-8, -15]
# trainset4 <- app_adj[train4, ]
# validateset4 <- app_adj[validate4, ]
# testset4 <- app_adj[test4, ]
# trainset4 <- app[train4, ]
# validateset4 <- app[validate4, ]
# testset4 <- app[test4, ]

### Run the KNN Regression Models ###
# create the 6 data frames for KNN models
# train4.x <- trainset4[-1]
# validate4.x <- validateset4[-1]
# test4.x <- testset4[-1]
# train4.y <- trainset4[1]
# validate4.y <- validateset4[1]
# test4.y <- testset4[1]


trainprop4 <- 0.9
validateprop4 <- 0.1
train4 <- sample(n4, n4*trainprop4)
validate4 <- setdiff(1:n4, train4)
trainset4 <- app[train4, ]
validateset4 <- app[validate4, ]
train4.x <- trainset4[-1]
validate4.x <- validateset4[-1]
train4.y <- trainset4[1]
validate4.y <- validateset4[1]



# use knn.reg() function to predict median value of homes
k_list4 <- seq(1, 19, by = 2)
train4.MSE <- rep(0, length(k_list4))
validate4.MSE <- rep(0, length(k_list4))
for (i in k_list4) {
    knn4.pred <- knn.reg(train4.x, validate4.x, train4.y, k = i)$pred
    validate4.MSE[(i+1)/2] <- sum((knn4.pred - validate4.y)^2)/length(knn4.pred)
    knn4.pred <- knn.reg(train4.x, train4.x, train4.y, k=i)$pred
    train4.MSE[(i+1)/2] <- sum((knn4.pred - train4.y)^2)/length(knn4.pred)
}
# plot MSE as function of flexibility for KNN regression
plot(NULL, NULL, type='n', xlim=c(length(k_list)*2-1, 1), ylim=c(0,max(c(validate4.MSE, train4.MSE))), xlab='Increasing Flexibility (Decreasing k)', ylab='Mean Squared Errors', main='MSEs as a Function of \n Flexibility for KNN Regression')
lines(seq(length(k_list)*2-1, 1, by = -2), validate4.MSE[length(validate4.MSE):1], type='b', col=2, pch=16)
lines(seq(length(k_list)*2-1, 1, by = -2), train4.MSE[length(train4.MSE):1], type='b', col=1, pch=16)
legend("topright", legend = c("Validation MSEs", "Training MSEs"), col=c(2, 1), cex=.75, pch=16, lty=1)
# print minimum validate/training MSE and corresponding k's
print(paste("Minimum validate set MSE occurred at k=", which.min(validate4.MSE)*2-1))
print(paste("Minimum validate MSE was", validate4.MSE[which.min(validate4.MSE)]))
print(paste("Minimum training set MSE occurred at k=", which.min(train4.MSE)*2-1))
print(paste("Minimum training MSE was", train4.MSE[which.min(train4.MSE)]))
# predict medv and calculate MSE for test set
# knn4.pred <- knn.reg(train4.x, test4.x, train4.y, k = which.min(validate1.MSE)*2-1)$pred
# test4.MSE <- mean((knn4.pred - test4.y)^2)
# test4.MSE
