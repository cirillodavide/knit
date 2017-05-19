library(igraph)

file <- ""
outpdf <- ""

links <- read.csv(file, header=T, sep='\t')

# matrix

# pair-wise list
net <- graph_from_data_frame(d=links, directed=F) 

net <- simplify(net, remove.multiple = F, remove.loops = T)
V(net)$frame.cex <- 0.2

V(net)$label.cex <- 0.2
V(net)$frame.color <- NA
V(net)$label <- NA


pdf(outpdf)
layout <- layout_nicely
par(mar=c(1,1,1,1))
plot(net,
    layout=layout,
    vertex.label.cex=.2)
dev.off()


pdf('erase.pdf')
col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = as.matrix(res), col = col, symm = TRUE)
dev.off()

library(reshape2)
df <- read.table('data/E.coli/coexpression.ecoli.511145',sep='\t')
colnames(df) <- c('user','item','rate')
m <- acast(df, user ~ item)
image(m)
corrplot(m, type='upper', order='hclust',tl.col='black',tl.srt=45)

