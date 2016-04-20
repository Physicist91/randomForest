### Linear Model Selection with Cross-Validation

library("ISLR")

names(Hitters)
dim(Hitters)

# the Salary column has NA's
sum(is.na(Hitters$Salary))
# remove NA's in Salary
Hitters <- na.omit(Hitters)
sum(is.na(Hitters))
dim(Hitters)
summary(Hitters)


## full subsets selection
library("leaps")
regfit.full <- regsubsets(Salary ~ ., Hitters)
summary(regfit.full)

regfit.full <- regsubsets(Salary ~ ., data=Hitters, nvmax=19)
summary(regfit.full)
reg.summary <- summary(regfit.full)
names(reg.summary)

reg.summary$rsq
plot(reg.summary$rsq, xlab="Number of Variables", ylab="rsq", type="l")

reg.summary$adjr2
plot(reg.summary$adjr2, xlab="Number of Variables", ylab="Adjusted RSq", type="l")
which.max(reg.summary$adjr2)
points(11, reg.summary$adjr2[11], col="red", cex=2, pch=20)

reg.summary$cp
plot(reg.summary$cp, xlab="Number of Variables", ylab="Cp", type='l')
which.min(reg.summary$cp)
points(10, reg.summary$cp[10], col="red", cex=2, pch=20)

reg.summary$bic
plot(reg.summary$bic, xlab="Number of Variables", ylab="BIC", type='l')
which.min(reg.summary$bic)
points(6, reg.summary$bic[6], col="red", cex=2, pch=20)

coef(regfit.full,6)


## forward selection
regfit.fwd <- regsubsets(Salary ~ ., data=Hitters, nvmax=19, method="forward")
summary(regfit.fwd)

## backward selection
regfit.bwd <- regsubsets(Salary ~ ., data=Hitters, nvmax=19, method="backward")
summary(regfit.bwd)


# the three selections may not give the same model
coef(regfit.full,7)
coef(regfit.fwd,7)
coef(regfit.bwd,7)



## Choosing Among Models using Cross Validation

# first create a predict() function for regsubsets()
predict.regsubsets <- function(object, newdata, id, ...){
    form <- as.formula(object$call[[2]])
    mat <- model.matrix(form, newdata)
    coefi <- coef(object, id=id)
    xvars <- names(coefi)
    mat[,xvars] %*% coefi
}


k <- 10
set.seed(1)
folds <- sample(1:k, nrow(Hitters), replace=TRUE)
predictions <- matrix(NA, nrow(Hitters), 19, dimnames=list(NULL, paste(1:19)))
for(j in 1:k){
    model.j <- regsubsets(Salary ~ ., data=Hitters[folds != j, ], nvmax=19)
    for(i in 1:19){
        pred <- predict(model.j, Hitters[folds == j, ], id=i)
        predictions[folds == j, i] <- pred
    }
}
head(predictions)
cv.errors <- colMeans((Hitters$Salary - predictions)^2)
cv.errors
plot(cv.errors, type="b", ylim=c(90000, 160000), col="red")
lines(reg.summary$rss / nrow(Hitters), type="b")


cv.rsq <- cor(Hitters$Salary, predictions)^2
cv.rsq
plot(1:19, cv.rsq, type="b", ylim=c(0.25, 0.6), col="red")
lines(1:19, reg.summary$rsq, type="b")


regfit.best <- regsubsets(Salary ~ ., data=Hitters, nvmax=19)
coef(regfit.best, 10)
str(regfit.best)
