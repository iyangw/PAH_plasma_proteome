library(clusterProfiler)
library(org.Hs.eg.db)

setwd('E:/PAH/Enrichment')
gene <- read.delim('PAH.txt', header = TRUE, stringsAsFactors = FALSE)[[1]]
geneid <- bitr(gene ,fromType="SYMBOL",toType=c("ENTREZID"),OrgDb = org.Hs.eg.db)
geneinput = geneid$ENTREZID

kegg <- enrichKEGG(
  gene = geneinput, 
  keyType = 'kegg', 
  organism = 'hsa', 
  pAdjustMethod = 'BH', 
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.25) 

kegg_top5 <- kegg[c(3:5,7),]
kegg_top5$pathway <- factor(kegg_top5$Description, levels = rev(kegg_top5$Description))
kegg_top5$pathway = sub("(.)","\\U\\1",kegg_top5$pathway,perl=TRUE)
colnames(kegg_top5)[11] <- "adjustedP"


library(ggplot2)
mytheme <- theme(axis.title = element_text(size = 13),
                 axis.text = element_text(size = 11, face = "bold"), 
                 plot.title = element_text(size = 14, hjust = 0.5, face = "bold"), 
                 legend.title = element_text(size = 13), 
                 legend.text = element_text(size = 11)) 

p_KEGG <- ggplot(data = kegg_top5, 
               aes(x = RichFactor, y = reorder(pathway,-adjustedP))) + 
  geom_point(aes(size = Count, color = -log10(adjustedP))) + 
  scale_size(breaks = c(4, 5, 6), range = c(4, 7)) +
  scale_color_distiller(palette = "Spectral",direction = -1) +
  labs(x = "Rich Factor", 
       y = "",
       title = "Dotplot of Enriched KEGG Pathways",
       size = "Count") + 
  theme_bw() + mytheme
p_KEGG
