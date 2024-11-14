#The enrichment analysis was based on 96 altered proteins

library(clusterProfiler)
library(enrichplot)
library(org.Hs.eg.db)
gene <- read.delim('PAH.txt', header = TRUE, stringsAsFactors = FALSE)[[1]]

GO_BP <- enrichGO(gene = gene,  
                  OrgDb = "org.Hs.eg.db",  
                  keyType = 'SYMBOL',  
                  ont = 'BP', 
                  pAdjustMethod = 'BH',  
                  pvalueCutoff = 0.05,  
                  qvalueCutoff = 0.25,  
                  readable = FALSE)
GO_MF <- enrichGO(gene = gene,  
                  OrgDb = "org.Hs.eg.db",  
                  keyType = 'SYMBOL',  
                  ont = 'MF', 
                  pAdjustMethod = 'BH',  
                  pvalueCutoff = 0.05,  
                  qvalueCutoff = 0.25,  
                  readable = FALSE)

GO_BP_top10 <- GO_BP[c(1,2,5,7:10,14:16),]
GO_MF_top10 <- GO_MF[c(1:7,9,11,14),]


GO_BP_top10$pathway <- factor(GO_BP_top10$Description, levels = rev(GO_BP_top10$Description))
GO_MF_top10$pathway <- factor(GO_MF_top10$Description, levels = rev(GO_MF_top10$Description))

GO_BP_top10$pathway = sub("(.)","\\U\\1",GO_BP_top10$pathway,perl=TRUE)
GO_MF_top10$pathway = sub("(.)","\\U\\1",GO_MF_top10$pathway,perl=TRUE)

colnames(GO_BP_top10)[9] <- "adjustedP"
colnames(GO_MF_top10)[9] <- "adjustedP"


library(ggplot2)
mytheme <- theme(axis.title = element_text(size = 13),
                 axis.text = element_text(size = 11, face = "bold"), 
                 plot.title = element_text(size = 14, hjust = 0.5, face = "bold"), 
                 legend.title = element_text(size = 13), 
                 legend.text = element_text(size = 11)) 

p_BP <- ggplot(data = GO_BP_top10, 
             aes(x = RichFactor, y = reorder(pathway,-adjustedP))) + 
  geom_point(aes(size = Count, color = -log10(adjustedP))) + 
  scale_size(range = c(4, 7)) +
  scale_color_distiller(palette = "Spectral",direction = -1) +
  labs(x = "Rich Factor", 
       y = "",
       title = "Dotplot of Enriched GO BP Pathways",
       size = "Count") + 
  theme_bw() + mytheme
p_BP


p_MF <- ggplot(data = GO_MF_top10, 
               aes(x = RichFactor, y = reorder(pathway,-adjustedP))) + 
  geom_point(aes(size = Count, color = -log10(adjustedP))) + 
  scale_size(range = c(4, 7)) +
  scale_color_distiller(palette = "Spectral",direction = -1) +
  labs(x = "Rich Factor", 
       y = "",
       title = "Dotplot of Enriched GO MF Pathways",
       size = "Count") + 
  theme_bw() + mytheme
p_MF