library(igraph)

#tab<-read.table('data/E.coli/ppi_files/ppi.ecoli.511145_rajagopala',sep='\t')
#tab<-tab[tab[,3]==1,]
tab<-read.table('data/E.coli/coexpression.ecoli.511145',sep='\t')
tab <- lapply(tab, function(x) {gsub("511145.", "", x)})

net<-graph_from_data_frame(tab)
d <- distances(net)
d[is.infinite(d)] <- 0

#write.table(d,'out/ppi.ecoli.511145_rajagopala_dist.csv',sep=',',quote=FALSE)
write.table(d,'out/coexpression.ecoli.511145_dist.csv',sep=',',quote=FALSE)
