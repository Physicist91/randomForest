### Fitting Regression Trees, Bagging, Random Forest, and Gradient Boosting Machine
### on Boston housing data

# the Boston housing data
library(MASS)
?Boston
summary(Boston)
Boston$chas <- as.factor(Boston$chas)

# separate training and test
set.seed(2345)
train.index <- sample(1:nrow(Boston), nrow(Boston)/2)
test.index <- -train.index
x.test <- Boston[test.index, -14]
y.test <- Boston[test.index, "medv"]


## Tree as a benchmark
library("tree")
# grow a tree
boston.tree <- tree(medv ~ ., Boston, subset=train.index)
summary(boston.tree)
plot(boston.tree)
text(boston.tree)
# CV
set.seed(12345)
boston.tree.cv <- cv.tree(boston.tree)
plot(boston.tree.cv, type="b")
# prune
boston.tree.pruned <- prune.tree(boston.tree, best=7)
plot(boston.tree.pruned)
text(boston.tree.pruned)
# predict
yhat.tree <- predict(boston.tree, newdata=x.test)
plot(yhat.tree, y.test)
abline(0,1)
# MSE
mse.tree <- mean((yhat.tree - y.test)^2)


## Bagging
library("randomForest")
set.seed(12)
# fit a bagging model (randomForest when mtry == p)
boston.bag <- randomForest(medv ~ ., data=Boston, subset=train.index, mtry=13)
boston.bag
plot(boston.bag)
# predict
yhat.bag <- predict(boston.bag, newdata=x.test)
plot(yhat.bag, y.test)
abline(0,1)
mse.bag <- mean((yhat.bag - y.test)^2)


## Random Forests
# fit a random forest model
set.seed(12)
boston.rf <- randomForest(medv ~ ., data=Boston, subset=train.index)
boston.rf
plot(boston.rf)
# predict
yhat.rf <- predict(boston.rf, newdata=x.test)
plot(yhat.rf, y.test)
abline(0,1)
mse.rf <- mean((yhat.rf - y.test)^2)

# variable importance
importance(boston.rf)
varImpPlot(boston.rf)
# more variable importance
boston.rf3 <- randomForest(medv ~ ., data=Boston, subset=train.index, importance=TRUE)
boston.rf3
importance(boston.rf3)
varImpPlot(boston.rf3)

# partial plot in RF
partialPlot(boston.rf, Boston[train.index, ], x.var="rm")
partialPlot(boston.rf, Boston[train.index, ], x.var="rad")
partialPlot(boston.rf, Boston[train.index, ], x.var="chas")

# tune random forest (mtry) by tuneRF (highly variable)
set.seed(12)
tuneRF(x=Boston[train.index, -14], y=Boston[train.index, 14], mtryStart=4, ntreeTry=500, stepFactor=1.5)

# tune random forest (mtry) manually
mse.rfs <- rep(0, 13)
for(m in 1:13){
    set.seed(12)
    rf <- randomForest(medv ~ ., data=Boston, subset=train.index, mtry=m)
    mse.rfs[m] <- rf$mse[500]
}
plot(1:13, mse.rfs, type="b", xlab="mtry", ylab="OOB Error")


# predict on test data directly
set.seed(12)
boston.bag2 <- randomForest(medv ~ ., data=Boston, subset=train.index, mtry=13, xtest=x.test, ytest=y.test)
boston.bag2
set.seed(12)
boston.rf2 <- randomForest(medv ~ ., data=Boston, subset=train.index, mtry=7, xtest=x.test, ytest=y.test)
boston.rf2
str(boston.rf2)
boston.rf2$test
mse.bag2 <- boston.bag2$test$mse[500]
mse.rf2 <- boston.rf2$test$mse[500]

# plots
plot(boston.bag2$mse, type="l", col=2, ylim=c(10, 30), xlab="Number of Trees", ylab="MSE")
lines(boston.bag2$test$mse, lwd=3, col=2)
abline(h=mse.tree, lty=2)
legend(350, 25, c("Bagging - OOB error", "Bagging - Test error"), col=c(2, 2), lty=c(1, 1), lwd=c(1, 3))

plot(boston.rf2$mse, type="l", col=4, ylim=c(10, 30), xlab="Number of Trees", ylab="MSE")
lines(boston.rf2$test$mse, lwd=3, col=4)
lines(boston.bag2$mse, col=2)
lines(boston.bag2$test$mse, lwd=3, col=2)
abline(h=mse.tree, lty=2)
legend(350, 25, c("Bagging - OOB error", "Bagging - Test error", "RF - OOB error", "RF - Test error"), col=c(2, 2, 4, 4), lty=c(1, 1, 1, 1), lwd=c(1, 3, 1, 3))



## Boosting
library("gbm")

# fit a boosting model
set.seed(321)
boston.gbm <- gbm(medv ~ ., data=Boston[train.index, ], distribution="gaussian", n.trees=5000, interaction.depth=4)
boston.gbm
#str(boston.gbm)

# inspect a particular tree
pretty.gbm.tree(boston.gbm, i.tree=2)

# plot training error
gbm.perf(boston.gbm)
# plot oob error
gbm.perf(boston.gbm, oobag.curve=TRUE)
hist(boston.gbm$oobag.improve[3000:5000])

# predict
yhat.gbm <- predict(boston.gbm, newdata=x.test, n.trees=5000)
mse.gbm <- mean((yhat.gbm - y.test)^2)


# tune gbm by CV
set.seed(321)
boston.gbm2 <- gbm(medv ~ ., data=Boston[train.index, ], distribution="gaussian", n.trees=20000, interaction.depth=4, shrinkage=0.001, cv.folds=10)
gbm.perf(boston.gbm2)
plot(boston.gbm2$cv.error, type="l")

yhat.gbm2 <- predict(boston.gbm2, newdata=x.test, n.trees=20000)
mse.gbm2 <- mean((yhat.gbm2 - y.test)^2)

# best tuned tree
set.seed(321)
boston.gbm3 <- gbm(medv ~ ., data=Boston[train.index, ], distribution="gaussian", n.trees=2000, interaction.depth=6, shrinkage=0.008)
boston.gbm3
gbm.perf(boston.gbm3)

yhat.gbm3 <- predict(boston.gbm3, newdata=x.test, n.trees=2000)
mse.gbm3 <- mean((yhat.gbm3 - y.test)^2)


# variable importance
summary(boston.gbm3)
summary(boston.gbm3, n.trees=1)

# partial plot in gbm
plot(boston.gbm3, i="rm")
plot(boston.gbm3, i="rad")
plot(boston.gbm3, i="chas")
plot(boston.gbm3, i=c("rm", "crim"))

# plot cv and test errors
test.gbm2 <- colMeans((predict(boston.gbm2, newdata=x.test, n.trees=seq(1, 20000, 100)) - y.test)^2)
gbm.perf(boston.gbm2)
lines(seq(1, 20000, 100), test.gbm2, type="l", col=4, lwd=3)
legend(7000, 80, c("GBM - Training error", "GBM - CV error", "GBM - Test error"), col=c(1, 3, 4), lwd=c(1, 1, 3))


## Ensemble of the models
yhat.ensemble1 <- (0.2 * boston.bag2$test$predicted + 0.8 * yhat.gbm3) 
mse.ensemble1 <- mean((yhat.ensemble1 - y.test)^2)
