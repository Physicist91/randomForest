##########################################################
# DSC5103 Statistics
# Session 10. Demo of RandomForest classification on the mixture.example dataset
# 2015.10
#
# -- based on the mixture.example dataset and documentation in the "ElemStatLearn" package
##########################################################


#############################
### Loading data
#############################
# load the dataset "mixture.example" in package "ElemStatLearn")
library("ElemStatLearn")  # run install.packages("ElemStatLearn") if you haven't


#############################
### Exloration
#############################

# copy important ones out
x <- mixture.example$x
y <- mixture.example$y
prob.true <- mixture.example$prob
px1 <- mixture.example$px1
px2 <- mixture.example$px2

summary(x)
summary(y)
summary(prob.true)

# make dataframe for x and y (for ggplot use)
df.training <- data.frame(x1=x[ , 1], x2=x[ , 2], y=y)
summary(df.training)
df.training$y <- as.factor(df.training$y)

# dataframe for plotting the boundary
df.grid <- expand.grid(x1=px1, x2=px2)
df.grid$prob.true <- prob.true
summary(df.grid)


# plot X and Y
library("ggplot2")
p0 <- ggplot() + geom_point(data=df.training, aes(x=x1, y=x2, color=y), size=4) + scale_color_manual(values=c("green", "red")) + theme_bw()
p0

# add the true boundary into the plot
p.true <- p0 + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.true), breaks=c(0.5))
p.true


#############################
### Test Data
#############################

# generate test data:
# The data do not contain a test sample, so we make one,
# using the description of the oracle page 17 of the book: The centers 
# is in the means component of mixture.example, with green(0) first, 
# so red(1).  For a test sample of size 10000 we simulate
# 5000 observations of each class.

# DO NOT WORRY about the secret algorithm for generating test data
library("MASS")
set.seed(123)
centers <- c(sample(1:10, 5000, replace=TRUE), 
             sample(11:20, 5000, replace=TRUE))
means <- mixture.example$means
means <- means[centers, ]
x.test <- mvrnorm(10000, c(0, 0), 0.2 * diag(2))
x.test <- x.test + means
y.test <- c(rep(0, 5000), rep(1, 5000))
df.test <- data.frame(x1=x.test[, 1], x2=x.test[, 2], y=as.factor(y.test))
summary(df.test)

# best possible misclassification rate
bayes.error <- sum(mixture.example$marginal * (prob.true * I(prob.true < 0.5) + (1 - prob.true) * I(prob.true >= 0.5)))


#############################
### knn classification
#############################
library("class")
knn7  <- knn(x, x.test, y, k=7, prob=TRUE)
prob <- attr(knn7, "prob")
prob <- ifelse(knn7 == "1", prob, 1 - prob)
df.test$prob.knn7 <- prob
df.test$pred.knn7 <- knn7
head(df.test)
table(df.test$pred.knn7, df.test$y)

# plot the boundary
knn7b <- knn(x, df.grid[, 1:2], y, k=7, prob=TRUE)
prob7b <- attr(knn7b, "prob")
prob7b <- ifelse(knn7b == "1", prob7b, 1 - prob7b)
df.grid$prob.knn7 <- prob7b
p.knn7 <- p0 + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.knn7), breaks=c(0.5)) 
p.knn7


#############################
### Logistic Regression classification
#############################

## fit the Logistic Regression model with 6th-order polynomial
lr6 <- glm(y ~ poly(x1, 6) + poly(x2, 6), data=df.training, family=binomial())
summary(lr6)

df.grid$prob.lr6 <- predict(lr6, newdata=df.grid, type="response")
head(df.grid)

# plot the decision boundary of Logistic Regression model lr6
p.lr6 <- p0 + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.lr6), breaks=c(0.5)) 
p.lr6

## predict with the Logistic Regression models
df.test$prob.lr6 <- predict(lr6, newdata=df.test, type="response")
df.test$pred.lr6 <- as.factor(ifelse(df.test$prob.lr6 > 0.5, 1, 0))
head(df.test)
table(df.test$pred.lr6, df.test$y)


#############################
### Tree
#############################
library("tree")

# grow a tree
tree1 <- tree(y ~ ., data=df.training)
tree1
summary(tree1)
# plot the fitted tree
plot(tree1)
text(tree1)

# pruning by cross-validation
set.seed(123)
tree1.cv <- cv.tree(tree1, method="misclass")
tree1.cv
plot(tree1.cv)

# optimal tree size obtained by CV
optimal <- which.min(tree1.cv$dev)
optimal.size <- tree1.cv$size[optimal]

# pruned tree
tree1.pruned <- prune.tree(tree1, best=optimal.size, method="misclass")
tree1.pruned
plot(tree1.pruned)
text(tree1.pruned)

# plot the partition
plot(df.training$x1, df.training$x2, col=ifelse(df.training$y==1, 2, 3), pch=20, cex=2)
partition.tree(tree1.pruned, ordvars=c("x1", "x2"), add=TRUE)

# plot the boundary
df.grid$prob.tree6 <- predict(tree1.pruned, newdata=df.grid, type="vector")[, 2]
head(df.grid)
p.tree6 <- p0 + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.tree6), breaks=c(0.5)) 
p.tree6

# prediction on test data
df.test$prob.tree6 <- predict(tree1.pruned, newdata=df.test, type="vector")[, 2]
df.test$pred.tree6 <- predict(tree1.pruned, newdata=df.test, type="class")
head(df.test)
table(df.test$pred.tree6, df.test$y)


#############################
### Random Forest
#############################
library("randomForest")

# fit bagging and random forest
set.seed(12)
bag <- randomForest(y ~ ., data=df.training, mtry=2, xtest=df.test[, 1:2], ytest=df.test[, 3], keep.forest=TRUE)
plot(bag)
set.seed(12)
rf <- randomForest(y ~ ., data=df.training, mtry=1, ntree=500, xtest=df.test[, 1:2], ytest=df.test[, 3], keep.forest=TRUE)
plot(rf)

# plot the RF boundary
df.grid$prob.rf <- predict(rf, newdata=df.grid, type="prob")[, 2]
head(df.grid)
p.rf <- p0 + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.rf), breaks=c(0.5)) 
p.rf

# partial plot in RF
partialPlot(rf, df.training, x.var="x1", which.class="1")
partialPlot(rf, df.training, x.var="x2", which.class="1")


# prediction
df.test$prob.bag <- bag$test$votes[, 2]
df.test$prob.rf <- rf$test$votes[, 2]
df.test$pred.rf <- rf$test$predicted
head(df.test)
table(df.test$pred.rf, df.test$y)


#############################
### GBM
#############################
library("gbm")

y.train <- as.numeric(df.training$y) - 1
# fit a boosting model
set.seed(321)
boost <- gbm(y.train ~ x1 + x2, data=df.training, distribution="bernoulli", n.trees=5000, shrinkage=0.001, interaction.depth=4)
gbm.perf(boost)

# partial effect
plot(boost, i=1, type="response")
plot(boost, i=2, type="response")

# plot the boundary
df.grid$prob.gbm <- predict(boost, newdata=df.grid, n.trees=5000, type="response")
p.gbm <- p0 + stat_contour(data=df.grid, aes(x=x1, y=x2, z=prob.gbm), breaks=c(0.5)) 
p.gbm

# prediction
df.test$prob.gbm <- predict(boost, newdata=df.test, n.trees=5000, type="response")
df.test$pred.gbm <- as.factor(ifelse(df.test$prob.gbm > 0.5, 1, 0))
head(df.test)
table(df.test$pred.gbm, df.test$y)


#############################
## compare ROC
#############################
library("ROCR")

# construct ROCR prediction objects
knn7.pred <- prediction(df.test$prob.knn7, df.test$y)
lr6.pred <- prediction(df.test$prob.lr6, df.test$y)
tree6.pred <- prediction(df.test$prob.tree6, df.test$y)
bag.pred <- prediction(df.test$prob.bag, df.test$y)
rf.pred <- prediction(df.test$prob.rf, df.test$y)
gbm.pred <- prediction(df.test$prob.gbm, df.test$y)


# misclassification rate
knn7.err <- performance(knn7.pred, measure = "err")
lr6.err <- performance(lr6.pred, measure = "err")
tree6.err <- performance(tree6.pred, measure = "err")
bag.err <- performance(bag.pred, measure = "err")
rf.err <- performance(rf.pred, measure = "err")
gbm.err <- performance(gbm.pred, measure = "err")

plot(knn7.err, ylim=c(0.2, 0.5), col="yellow")
plot(lr6.err, add=TRUE, col="purple")
plot(tree6.err, add=TRUE, col="green")
plot(bag.err, add=TRUE, col="blue", lwd=2)
plot(rf.err, add=TRUE, col="red", lwd=2)
plot(gbm.err, add=TRUE, col="grey", lwd=2)
abline(h=bayes.error, lty=2)


# ROC plot
knn7.ROC <- performance(knn7.pred, measure = "tpr", x.measure = "fpr")
lr6.ROC <- performance(lr6.pred, measure = "tpr", x.measure = "fpr")
tree6.ROC <- performance(tree6.pred, measure = "tpr", x.measure = "fpr")
bag.ROC <- performance(bag.pred, measure = "tpr", x.measure = "fpr")
rf.ROC <- performance(rf.pred, measure = "tpr", x.measure = "fpr")
gbm.ROC <- performance(gbm.pred, measure = "tpr", x.measure = "fpr")

plot(knn7.ROC, col="yellow")
abline(a=0, b=1, lty=2) # diagonal line
plot(lr6.ROC, add=TRUE, col="purple")
plot(tree6.ROC, add=TRUE, col="green")
plot(bag.ROC, add=TRUE, col="blue", lwd=2)
plot(rf.ROC, add=TRUE, col="red", lwd=2)
plot(gbm.ROC, add=TRUE, col="red", lwd=2)


# AUC
as.numeric(performance(knn7.pred, "auc")@y.values)
as.numeric(performance(lr6.pred, "auc")@y.values)
as.numeric(performance(tree6.pred, "auc")@y.values)
as.numeric(performance(bag.pred, "auc")@y.values)
as.numeric(performance(rf.pred, "auc")@y.values)
as.numeric(performance(gbm.pred, "auc")@y.values)



#############################
## plot the boundaries together
#############################
library("grid")
library("gridExtra")
grid.arrange(p.true, p.knn7, p.lr6, p.tree6, p.rf, p.gbm, ncol=3)
