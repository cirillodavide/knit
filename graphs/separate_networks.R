library(igraph)

links <- read.csv("out/coexpr_user-similarity.csv", header=T, row.names=1)
links <- as.matrix(links)
net <- graph_from_incidence_matrix(links) 
net <- simplify(net, remove.multiple = F, remove.loops = T)

E(net)$width <- 0.01
V(net)$size <- degree(net)/max(degree(net))
V(net)$frame.color <- NA
V(net)$label <- V(net)$name
V(net)$label.cex <- 0.2
E(net)$arrow.size <- 0

layout <- layout_nicely

pdf("graphs/separate_networks.pdf")
	par(mar=c(1,1,1,1))
	plot(net,
		layout=layout,
		vertex.label.cex=.2)
dev.off()
