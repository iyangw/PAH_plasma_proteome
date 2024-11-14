library(VennDiagram) 

set1 = c(1:539)
set2 = c(46:619)

venn.diagram(x=list(set1,set2),
             scaled = F, 
             alpha= 0.5, 
             lty='blank', 
             label.col ='#000000' ,
             cex = 2, 
             fontface = "bold", 
             fill=c('#F4EC31','#64B2E7'), 
          #  fill=c('#FFFFCC','#CCFFFF'),
             category.names = c("2022", "2023") , 
             cat.dist = 0.02, 
             cat.pos = -180, 
             cat.cex = 2, 
             cat.fontface = "bold",
             cat.col='#000000' ,
             cat.default.pos = "outer",  
             output=TRUE,
             filename='E:/PAH/Venn.tiff',
             imagetype="tiff",
             resolution = 400,  
             compression = "lzw"
)
