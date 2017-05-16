tab<-read.table("out/estimates.csv",sep=',',header=TRUE)
plot(tab$rate,tab$rate_estimate,pch=20)
cor(tab$rate,tab$rate_estimate)

tab1<-tab[tab$rate!=tab$rate_estimate,]
plot(tab1$rate,tab1$rate_estimate,pch=20)
cor(tab1$rate,tab1$rate_estimate)
