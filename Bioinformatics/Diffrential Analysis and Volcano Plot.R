#Differential analysis was based on Supplementary Table S6

library(readxl)
pah = read_excel("PAH_vertical.xlsx")
list <- c(rep("PAH", 36), rep("CTR",30),rep("PAH", 10)) %>% factor(., levels = c("PAH", "CTR"), ordered = F)
list <- model.matrix(~factor(list)+0)  
colnames(list) <- c("PAH", "CTR")

library(limma)
df.pah <- lmFit(pah, list) 
df.matrix <- makeContrasts(PAH - CTR , levels = list)
fit <- contrasts.fit(df.pah, df.matrix)
fit <- eBayes(fit)
tempOutput <- topTable(fit,n = Inf, adjust.method="BH")

library(ggVolcano)
tempOutput_new <- add_regulate(tempOutput, log2FC_name = "logFC",
                           fdr_name = "adj.P.Val",log2FC = 0.58, fdr = 0.05)
gradual_volcano(tempOutput, x = "logFC", y = "adj.P.Val",log2FC_cut = 0.58,
                label = "Lead.Gene.Name", label_number = 10, output = FALSE)
ggvolcano(tempOutput_new, x = "log2FoldChange", y = "padj", log2FC_cut = 0.58,
          label = "Lead.Gene.Name", label_number = 10, output = FALSE)
