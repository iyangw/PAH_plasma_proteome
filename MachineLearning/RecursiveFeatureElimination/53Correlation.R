library(ComplexHeatmap)
library(circlize)

library(readxl)
data <- read_excel("13proteins.xlsx")
data = data[,-1]
data = as.matrix(data)

cor_matrix <- cor(data) 

custom_colors <- colorRamp2(c(-1, -0.5, 0, 0.5, 1), c("#053061", "#6CA4CC", "#FFFFFF", "#F4A582", "#67001F"))

png("high_res_heatmap.png", width = 6000, height = 6000, res = 600)
# 绘制热图
Heatmap(
  cor_matrix,
  name = "Correlation", 
  col = custom_colors,        
  heatmap_legend_param = list(
    title = "Correlation",
    title_gp = gpar(fontsize = 12),
    labels_gp = gpar(fontsize = 10),
    legend_height = unit(10, "cm")   
  ),
  cell_fun = function(j, i, x, y, width, height, fill) {
    text_color <- ifelse(cor_matrix[i, j] == 1, "white", "black")
    grid.text(sprintf("%.2f", cor_matrix[i, j]), x, y, gp = gpar(fontsize = 8, col = text_color))
  }
)
dev.off()

