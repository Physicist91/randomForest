##########################################################
# DSC5103 Statistics
# Session 6. Demo of the Bootstrap on Logistic Regression with rare events
# 2015.9
#
##########################################################
library("plyr")
library("ggplot2")


#############################
## simulate data
#############################
N <- 500  # sample size
set.seed(12345)
x <- rnorm(N)  # one-dimensional predictor
beta0.true <- seq(0, -5)  # coefficients to test on
beta1.true <- 1
K <- length(beta0.true)
data <- data.frame(x=x)
for (k in 1:K) {
    eta <- beta0.true[k] + beta1.true * x
    prob.true <- 1 / (1 + exp(-eta))
    y <- (runif(N) < prob.true)
    data[, paste0("y_", k-1)] <- y
}
summary(data)



#############################
### Resample from population (as a benchmark)
#############################

RUN <- 1000
beta.est <- expand.grid(run=1:RUN, k=1:K)
beta.est$beta0.pop <- 0
beta.est$beta1.pop <- 0
head(beta.est)
for (run in 1:RUN) {
    for (k in 1:K) {
        # generate new sample from populatin
        eta <- beta0.true[k] + beta1.true * x
        prob.true <- 1 / (1 + exp(-eta))
        y.new <- runif(N) < prob.true
        
        # obtain estimates
        glm.fit <- glm(y.new ~ x, family=binomial())
        beta.est[beta.est$run == run & beta.est$k == k, c("beta0.pop", "beta1.pop")] <- coef(glm.fit)
    }
}
summary(beta.est)

# find mean and sd of the estimators
ddply(beta.est, .(k), summarize, mean0=mean(beta0.pop), sd0=sd(beta0.pop), mean1=mean(beta1.pop), sd1=sd(beta1.pop))

# plot the distribution of the estimators
ggplot(data=beta.est) + geom_density(aes(color=factor(k), x=beta0.pop)) + xlim(c(-8, 0.5)) + theme_bw()
ggplot(data=beta.est) + geom_density(aes(color=factor(k), x=beta1.pop)) + xlim(c(-1, 3)) + theme_bw()



#############################
### The Bootstrap by boot()
#############################
library("boot")

# boot.fn is the function to obtain estimators
boot.fn <- function(data, index){
    return(coef(glm(y ~ x, data=data, family=binomial(), subset=index)))
}
# test boot.fn on the whole dataset
boot.fn(data.frame(x=data[, 1], y=data[, 3]), 1:N)

## run the bootstrap on the k-th dataset
beta.est$beta0.bs <- 0
beta.est$beta1.bs <- 0
head(beta.est)
set.seed(119)
for (k in 1:K) {
    # obtain bootstrap estimates
    boot.out <- boot(data=data.frame(x=data[, 1], y=data[, k+1]), statistic=boot.fn, R=RUN)
    beta.est[beta.est$k == k, c("beta0.bs", "beta1.bs")] <- boot.out$t
}
summary(beta.est)

# find mean and sd of the estimators
ddply(beta.est, .(k), summarize, mean0=mean(beta0.bs), sd0=sd(beta0.bs), mean1=mean(beta1.bs), sd1=sd(beta1.bs))

# plot the distribution of the estimators
ggplot(data=beta.est) + geom_density(aes(color=factor(k), x=beta0.bs)) + xlim(c(-8, 0.5)) + theme_bw()
ggplot(data=beta.est) + geom_density(aes(color=factor(k), x=beta1.bs)) + xlim(c(-1, 3)) + theme_bw()



#############################
### The Bootstrap manually
#############################

beta.est$beta0.bs2 <- 0
beta.est$beta1.bs2 <- 0
head(beta.est)
set.seed(119)
for (run in 1:RUN) {
    # index for the bootstrap sample
    index <- sample(N, N, replace=TRUE)

    for (k in 1:K) {
        # obtain estimates
        glm.fit <- glm(data[index, k+1] ~ x[index], family=binomial())
        beta.est[beta.est$run == run & beta.est$k == k, c("beta0.bs2", "beta1.bs2")] <- coef(glm.fit)
    }
}
summary(beta.est)

# find mean and sd of the estimators
ddply(beta.est, .(k), summarize, mean0=mean(beta0.bs2), sd0=sd(beta0.bs2), mean1=mean(beta1.bs2), sd1=sd(beta1.bs2))

# plot the distribution of the estimators
ggplot(data=beta.est) + geom_density(aes(color=factor(k), x=beta0.bs2)) + xlim(c(-8, 0.5)) + theme_bw()
ggplot(data=beta.est) + geom_density(aes(color=factor(k), x=beta1.bs2)) + xlim(c(-1, 3)) + theme_bw()




## put everything together
ggplot(data=subset(beta.est, k==2)) + geom_density(aes(x=beta0.pop))  + geom_density(aes(x=beta0.bs), color="red")  + geom_density(aes(x=beta0.bs2), color="blue") 
ggplot(data=subset(beta.est, k==2)) + geom_density(aes(x=beta1.pop))  + geom_density(aes(x=beta1.bs), color="red")  + geom_density(aes(x=beta1.bs2), color="blue") 

# compare the sd of estimators
ddply(beta.est, .(k), summarize, sd.pop=sd(beta0.pop), sd.bs=sd(beta0.bs), sd.bs2=sd(beta0.bs2))
ddply(beta.est, .(k), summarize, sd.pop=sd(beta1.pop), sd.bs=sd(beta1.bs), sd.bs2=sd(beta1.bs2))

