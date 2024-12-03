from matplotlib_venn import venn2, venn2_circles
import matplotlib.pyplot as plt

Group1 = range(1, 540)
Group2 = range(46, 620)

plt.rcParams['font.family'] = ["Arial"]
fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
vee2 = venn2((set(Group1), set(Group2)), set_labels=("2022", "2023"),
             set_colors=("#0073C2FF", "#EFC000FF"), alpha=0.8, ax=ax)
venn2_circles((set(Group1), set(Group2)), linestyle="--", linewidth=2, color="black", ax=ax)

for text in vee2.set_labels:
    text.set_fontsize(24)
    text.set_fontweight('bold')
    # 修改位置，您可以根据需要调整x和y的数值
    if text.get_text() == "2022":
        text.set_position((-0.5, 0.5))  # 2022标签的位置
    elif text.get_text() == "2023":
        text.set_position((0.5, 0.5))  # 2023标签的位置
for text in vee2.subset_labels:
    text.set_fontsize(20)
    text.set_fontweight("bold")

plt.show()
