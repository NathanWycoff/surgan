#!/usr/bin/Rscript
#  R/quantile_fun.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.19.2019

#Can we come up with a way of estimating quantiles that doesn't involve sorting the data?

################## Take 1
N <- 100000
x <- rnorm(N)
alpha <- 0.8
qnorm(alpha)

mu <- 0

eta <- 1.0
mus <- rep(NA, N)
for (n in 1:N) {
    eta <- 1
    if (x[n] > mu) {
        mu <- mu + (eta * alpha)
    } else {
        mu <- mu + (eta * (1-alpha))
    }
}

################## Take 2
