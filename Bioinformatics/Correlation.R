library(ppcor)
library(readxl)
pah <- read_excel("PAHcorrelation.xlsx")
pah_clean <- subset(pah, !is.na(mPAP))

X <- pah_clean$mPAP
Sex <- pah_clean$Sex
Sex <- as.factor(Sex)
Age <- pah_clean$Age
BMI <- pah_clean$BMI
Subtype <- pah_clean$Subtype
Subtype <- as.factor(Subtype)
Meditation <- pah_clean$Meditation


result <- data.frame(Protein = colnames(pah_clean),
                     Partial_Correlation = numeric(505),
                     P_value = numeric(505))


for (i in 12:505) {  
  Y <- pah_clean[, i]  

  pcor_result <- pcor.test(X, Y, c(Sex, Age, BMI, Subtype, Meditation), method = "spearman")  
  
  result$Partial_Correlation[i] <- pcor_result$estimate
  result$P_value[i] <- pcor_result$p.value
}


result=result[c(12:505),]

result$adjusted_p_values <- p.adjust(result$P_value, method = "BH")

print(result)

library(writexl)
write_xlsx(result, path = "mPAP.xlsx")