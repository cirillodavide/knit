library('igraph')

tab <- read.table("out/item-similarity.csv",sep=',',header=TRUE, row.names=1)
links <- as.matrix(tab)
print(dim(links))

net <- graph_from_incidence_matrix(links)
net <- simplify(net, remove.multiple = F, remove.loops = T) 

pdf('graphs/similarity.clust.pdf')
plot(net)
dev.off()
