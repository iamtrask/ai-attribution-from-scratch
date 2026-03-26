"""Generate the "Who gets credit?" plot for Lecture 1."""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

x_common = np.random.randn(30) * 0.6 + 2.5
y_common = 0.5 * x_common + np.random.randn(30) * 0.2 + 1.0
x_unique = np.array([6.5])
y_unique = np.array([5.0])

x_all = np.concatenate([x_common, x_unique])
y_all = np.concatenate([y_common, y_unique])
c1 = np.polyfit(x_all, y_all, 1)
c2 = np.polyfit(x_common, y_common, 1)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.scatter(x_common, y_common, c='tab:blue', s=20, alpha=0.5)
ax.scatter(x_unique, y_unique, c='tab:red', s=80, marker='*')
x_l = np.linspace(0, 8, 100)
ax.plot(x_l, np.polyval(c1, x_l), 'k-', lw=1.5)
ax.plot(x_l, np.polyval(c2, x_l), 'b--', lw=1, alpha=0.4)
ax.annotate('', xy=(6.5, np.polyval(c1,6.5)), xytext=(6.5, np.polyval(c2,6.5)),
            arrowprops=dict(arrowstyle='<->', color='tab:red', lw=1.5))
ax.set_xlim(0, 8); ax.set_ylim(0, 6)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('credit_plot.png', dpi=150)
print("Saved credit_plot.png")
