import matplotlib.pyplot as plt
import numpy as np

# Datos de los modelos
data_sizes = [1000, 5000, 10000, 50000, 100000]

# ML (XGBoost) - Recall
ml_recall_class0 = [0.92, 0.96, 0.98, 0.95, 0.96]
ml_recall_class1 = [0.94, 0.82, 0.77, 0.80, 0.80]

# MLP - Recall
mlp_recall_class0 = [0.94, 0.91, 0.90, 0.92, 0.92]
mlp_recall_class1 = [0.82, 0.85, 0.87, 0.89, 0.90]

# TabNet - Recall
tabnet_recall_class0 = [0.98, 0.95, 0.94, 0.94, 0.93]
tabnet_recall_class1 = [0.53, 0.82, 0.82, 0.83, 0.86]

# FIGURA 1: Recall Clase 0 (Sin diabetes)
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, ml_recall_class0, 'o-', label='XGBoost', linewidth=3, markersize=10, color='#1f77b4')
plt.plot(data_sizes, mlp_recall_class0, 's-', label='MLP', linewidth=3, markersize=10, color='#ff7f0e')
plt.plot(data_sizes, tabnet_recall_class0, '^-', label='TabNet', linewidth=3, markersize=10, color='#2ca02c')
plt.title('Recall Clase 0 (Sin Diabetes)', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de Datos', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(data_sizes, [f'{x//1000}k' if x >= 1000 else str(x) for x in data_sizes])
plt.ylim(0.85, 1.02)
plt.tight_layout()
plt.show()

# FIGURA 2: Recall Clase 1 (Con diabetes)
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, ml_recall_class1, 'o-', label='XGBoost', linewidth=3, markersize=10, color='#1f77b4')
plt.plot(data_sizes, mlp_recall_class1, 's-', label='MLP', linewidth=3, markersize=10, color='#ff7f0e')
plt.plot(data_sizes, tabnet_recall_class1, '^-', label='TabNet', linewidth=3, markersize=10, color='#2ca02c')
plt.title('Recall Clase 1 (Con Diabetes)', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de Datos', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xticks(data_sizes, [f'{x//1000}k' if x >= 1000 else str(x) for x in data_sizes])
plt.ylim(0.25, 0.95)
plt.tight_layout()
plt.show()

# FIGURA 3: Gráfico de barras - Recall en 100k datos
plt.figure(figsize=(12, 6))
models = ['XGBoost', 'MLP', 'TabNet']
recall_class0_100k = [ml_recall_class0[-1], mlp_recall_class0[-1], tabnet_recall_class0[-1]]
recall_class1_100k = [ml_recall_class1[-1], mlp_recall_class1[-1], tabnet_recall_class1[-1]]

x = np.arange(len(models))
width = 0.35

bars1 = plt.bar(x - width/2, recall_class0_100k, width, label='Recall Clase 0 (Sin Diabetes)', 
                color='#87CEEB', alpha=0.8)
bars2 = plt.bar(x + width/2, recall_class1_100k, width, label='Recall Clase 1 (Con Diabetes)', 
                color='#FF6B6B', alpha=0.8)

plt.title('Comparación de Recall con 100k Datos', fontsize=16, fontweight='bold')
plt.ylabel('Recall', fontsize=14)
plt.xticks(x, models)
plt.legend(fontsize=12)
plt.ylim(0.0, 1.05)
plt.grid(True, alpha=0.3, axis='y')

# Agregar valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                 f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# Análisis de resultados
print("=" * 60)
print("ANÁLISIS DE RECALL")
print("=" * 60)

print("\n🎯 MEJOR RECALL CLASE 1 (DIABETES):")
best_recall_diabetes = ["XGBoost", "MLP", "TabNet"][np.argmax([max(ml_recall_class1), max(mlp_recall_class1), max(tabnet_recall_class1)])]
print(f"   {best_recall_diabetes}: {max([max(ml_recall_class1), max(mlp_recall_class1), max(tabnet_recall_class1)]):.3f}")

print("\n🏥 MEJOR RECALL CLASE 0 (SIN DIABETES):")
best_recall_no_diabetes = ["XGBoost", "MLP", "TabNet"][np.argmax([max(ml_recall_class0), max(mlp_recall_class0), max(tabnet_recall_class0)])]
print(f"   {best_recall_no_diabetes}: {max([max(ml_recall_class0), max(mlp_recall_class0), max(tabnet_recall_class0)]):.3f}")

print("\n📊 RENDIMIENTO CON 100K DATOS:")
print(f"   XGBoost: R0={ml_recall_class0[-1]:.2f}, R1={ml_recall_class1[-1]:.2f}")
print(f"   MLP:     R0={mlp_recall_class0[-1]:.2f}, R1={mlp_recall_class1[-1]:.2f}")
print(f"   TabNet:  R0={tabnet_recall_class0[-1]:.2f}, R1={tabnet_recall_class1[-1]:.2f}")

print("\n🔍 OBSERVACIONES CLAVE:")
print("   • TabNet: Perfecto recall clase 0 (1.0) pero bajo en diabetes")
print("   • MLP: Mejor balance general, mejora con más datos")
print("   • XGBoost: Recall diabetes decrece con más datos")