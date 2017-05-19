library(igraph)

args <- commandArgs(trailingOnly = TRUE)
tag <- args[1]

links <- read.csv(paste("out/",tag,"_knitout.csv",sep=""), header=T, as.is=T)
#links <- links[links$gene%in%c("BRCA1","RAD51"),]
#links <- links[,-5]
#links$weight <- 1
#links <- unique(links)

net <- graph_from_data_frame(d=links, directed=T) 
net <- simplify(net, remove.multiple = F, remove.loops = T)

#net <- delete.edges(net, which(E(net)$weight <= 0.99))
#net <- delete.vertices(net, which(degree(net) < 4))

E(net)$color <- c("red","blue")[as.factor(E(net)$type)]
E(net)$width <- 0.01
V(net)$size <- degree(net)/max(degree(net))
V(net)$frame.color <- NA
V(net)$label <- V(net)$name
V(net)$label.cex <- 0.2
E(net)$arrow.size <- 0

layout <- layout_nicely

pdf(paste("out/",tag,"_knitter.pdf",sep=""))
	par(mar=c(1,1,1,1))
	plot(net,
		layout=layout,
		vertex.label.cex=.2)
dev.off()
