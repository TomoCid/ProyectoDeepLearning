import matplotlib.pyplot as plt
import numpy as np

# Datos de los modelos
data_sizes = [1000, 5000, 10000, 50000, 100000]

# ML (XGBoost) - Binario
ml_accuracy = [0.92, 0.952, 0.9615, 0.9404, 0.94695]
ml_f1_class0 = [0.95, 0.97, 0.98, 0.97, 0.97]
ml_f1_class1 = [0.65, 0.74, 0.78, 0.69, 0.72]

# MLP - Binario
mlp_accuracy = [0.93, 0.92, 0.91, 0.92, 0.92]
mlp_f1_class0 = [0.96, 0.94, 0.94, 0.95, 0.95]
mlp_f1_class1 = [0.67, 0.60, 0.60, 0.63, 0.64]

# TabNet
tabnet_accuracy = [0.9400, 0.9360, 0.9330, 0.9267, 0.9262]
tabnet_f1_class0 = [0.97, 0.96, 0.96, 0.96, 0.96]
tabnet_f1_class1 = [0.60, 0.69, 0.68, 0.66, 0.67]


# FIGURA 1: Accuracy
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, ml_accuracy, 'o-', label='XGBoost', linewidth=3, markersize=10, color='#1f77b4')
plt.plot(data_sizes, mlp_accuracy, 's-', label='MLP', linewidth=3, markersize=10, color='#ff7f0e')
plt.plot(data_sizes, tabnet_accuracy, '^-', label='TabNet', linewidth=3, markersize=10, color='#2ca02c')
plt.title('Accuracy por Tamaño de Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de Datos', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(data_sizes, [f'{x//1000}k' if x >= 1000 else str(x) for x in data_sizes])
plt.ylim(0.88, 0.98)
plt.tight_layout()
plt.show()

# FIGURA 2: F1-Score Clase 0 (Sin diabetes)
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, ml_f1_class0, 'o-', label='XGBoost', linewidth=3, markersize=10, color='#1f77b4')
plt.plot(data_sizes, mlp_f1_class0, 's-', label='MLP', linewidth=3, markersize=10, color='#ff7f0e')
plt.plot(data_sizes, tabnet_f1_class0, '^-', label='TabNet', linewidth=3, markersize=10, color='#2ca02c')
plt.title('F1-Score Clase 0 (Sin Diabetes)', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de Datos', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(data_sizes, [f'{x//1000}k' if x >= 1000 else str(x) for x in data_sizes])
plt.ylim(0.92, 1.0)
plt.tight_layout()
plt.show()

# FIGURA 3: F1-Score Clase 1 (Con diabetes)
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, ml_f1_class1, 'o-', label='XGBoost', linewidth=3, markersize=10, color='#1f77b4')
plt.plot(data_sizes, mlp_f1_class1, 's-', label='MLP', linewidth=3, markersize=10, color='#ff7f0e')
plt.plot(data_sizes, tabnet_f1_class1, '^-', label='TabNet', linewidth=3, markersize=10, color='#2ca02c')
plt.title('F1-Score Clase 1 (Con Diabetes)', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de Datos', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(data_sizes, [f'{x//1000}k' if x >= 1000 else str(x) for x in data_sizes])
plt.ylim(0.40, 0.85)
plt.tight_layout()
plt.show()

# FIGURA 4: Gráfico de barras - Accuracy en 100k datos
plt.figure(figsize=(10, 6))
models = ['XGBoost', 'MLP', 'TabNet']
accuracy_100k = [ml_accuracy[-1], mlp_accuracy[-1], tabnet_accuracy[-1]]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = plt.bar(models, accuracy_100k, color=colors, alpha=0.8, width=0.6)

plt.title('Comparación de Accuracy con 100k Datos', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0.90, 0.98)
plt.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for bar, value in zip(bars, accuracy_100k):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Análisis de resultados
print("=" * 60)
print("ANÁLISIS DE RESULTADOS")
print("=" * 60)

print("\n🏆 MEJOR ACCURACY GENERAL:")
best_acc_model = ["XGBoost", "MLP", "TabNet"][np.argmax([max(ml_accuracy), max(mlp_accuracy), max(tabnet_accuracy)])]
print(f"   {best_acc_model}: {max([max(ml_accuracy), max(mlp_accuracy), max(tabnet_accuracy)]):.4f}")

print("\n🎯 MEJOR F1-SCORE CLASE 1 (DIABETES):")
best_f1_model = ["XGBoost", "MLP", "TabNet"][np.argmax([max(ml_f1_class1), max(mlp_f1_class1), max(tabnet_f1_class1)])]
print(f"   {best_f1_model}: {max([max(ml_f1_class1), max(mlp_f1_class1), max(tabnet_f1_class1)]):.4f}")

print("\n📊 RENDIMIENTO CON 100K DATOS:")
print(f"   XGBoost: Acc={ml_accuracy[-1]:.4f}, F1_diabetes={ml_f1_class1[-1]:.2f}")
print(f"   MLP:     Acc={mlp_accuracy[-1]:.4f}, F1_diabetes={mlp_f1_class1[-1]:.2f}")
print(f"   TabNet:  Acc={tabnet_accuracy[-1]:.4f}, F1_diabetes={tabnet_f1_class1[-1]:.2f}")