library(umap)
library(readxl)
pah <- read_excel("PAH.xlsx")

library(dplyr)
pah <- pah %>% mutate_at('Group', as.factor)
pah.label = pah$Group
pah.data = pah[,-1]


config.params = umap.defaults
config.params$n_neighbors =10
config.params$min_dist =0.05

pah.umap = umap::umap(pah.data, config = config.params)

# 使用plot函数可视化UMAP的结果
plot(pah.umap$layout,col=pah.label,pch=16,asp = 1,
     xlab = "UMAP_1",ylab = "UMAP_2",
     main = "UMAP of the corrected dataset")
# 添加分隔线
abline(h=0,v=0,lty=2,col="gray")
# 添加图例
legend("topright",title = "Condition",inset = 0.01,
       legend = unique(pah.label),pch=16,
       col = unique(pah.label),
       bty = "n")