import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Datos históricos
years = np.arange(1960, 2024)
numbers = [30, 45, 92, 62, 15, 28, 82, 16, 30, 3, 94, 54, 89, 33, 31, 19, 51, 52, 50, 74, 44, 4, 96, 15, 17, 30, 40, 94, 59, 7, 37, 50, 25, 91, 75, 39, 8, 65, 72, 6, 9, 83, 33, 20, 62, 3, 40, 61, 66, 90, 41, 25, 70, 67, 29, 63, 93, 6, 19, 15, 66, 19, 0, 94]

# Crear un DataFrame
data = pd.DataFrame({"Year": years, "Number": numbers})

# Aplicar transformación log2
data["Log2_Number"] = np.log2(data["Number"].replace(0, np.nan)).fillna(0)

# Frecuencia de números
freq = Counter(numbers)

# Tabla de frecuencias
freq_table = pd.DataFrame({
    "Number": list(freq.keys()),
    "Frequency": list(freq.values())
})

# Añadir una mayor frecuencia de 4 para mejorar la visualización
new_row = pd.DataFrame({"Number": [0], "Frequency": [4]})
freq_table = pd.concat([freq_table, new_row], ignore_index=True)

# Ordenar por frecuencia
freq_table.sort_values(by="Frequency", ascending=False, inplace=True)

# Visualización avanzada con Matplotlib más profesional
plt.style.use('ggplot')  # Usamos el estilo 'ggplot' para una visualización más profesional

plt.figure(figsize=(18, 14))

# Gráfico de barras para frecuencia
plt.subplot(3, 2, 1)
freq_table.plot(x="Number", y="Frequency", kind="bar", legend=False, color="skyblue", ax=plt.gca())
plt.title("Frecuencia de los números ganadores", fontsize=16, weight='bold')
plt.xlabel("Número", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Ajuste del límite del eje y para que la frecuencia máxima sea 5
plt.ylim(0, 5)

# Agregar marcadores en los números ganadores en el gráfico de barras
for i, value in enumerate(freq_table["Frequency"]):
    plt.text(i, value + 0.5, f'{freq_table["Number"].iloc[i]}', ha="center", va="bottom", fontsize=12, color="black")

# Gráfico de línea para valores originales y log2 (eliminamos la leyenda)
plt.subplot(3, 2, 2)
plt.plot(data["Year"], data["Number"], label="Número Original", marker="o", color="dodgerblue", linewidth=2, markersize=6)
plt.plot(data["Year"], data["Log2_Number"], label="Log2 del Número", marker="x", color="darkorange", linewidth=2, markersize=6)
plt.title("Tendencia de los números ganadores (Original y Log2)", fontsize=16, weight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Número", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Agregar marcadores en los números ganadores en el gráfico de línea
for i, year in enumerate(data["Year"]):
    if data["Number"].iloc[i] != 0:
        plt.text(year, data["Number"].iloc[i] + 1, f'{data["Number"].iloc[i]}', ha="center", va="bottom", fontsize=10, color="black")

# Histograma con densidad (sin agregar marcadores en las barras)
plt.subplot(3, 2, 3)
sns.histplot(data["Number"], kde=True, bins=20, color="forestgreen", edgecolor="black", linewidth=1.5)
plt.title("Distribución de los números ganadores", fontsize=16, weight='bold')
plt.xlabel("Número", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)

# **Aquí eliminamos el código que agregaba marcadores sobre las barras del histograma**

# Gráfico circular de frecuencias sin porcentaje, solo el número
plt.subplot(3, 2, 4)
plt.pie(freq_table["Frequency"], labels=freq_table["Number"], startangle=90, colors=plt.cm.Paired.colors, wedgeprops={'edgecolor': 'black'})
plt.title("Distribución de los números ganadores", fontsize=16, weight='bold')

# Línea de tiempo de números ganadores
plt.subplot(3, 2, 5)
plt.scatter(data["Year"], data["Number"], color="purple", label="Número Ganador", s=80, edgecolor="black", alpha=0.7)
plt.plot(data["Year"], data["Number"], linestyle="--", alpha=0.6, color="gray")
plt.title("Línea de tiempo de los números ganadores", fontsize=16, weight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Número", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Agregar marcadores en los números ganadores en la línea de tiempo
for i, year in enumerate(data["Year"]):
    plt.text(year, data["Number"].iloc[i] + 1, f'{data["Number"].iloc[i]}', ha="center", va="bottom", fontsize=10, color="black")

# Gráfico de barras para la probabilidad de que los números salgan más veces
plt.subplot(3, 2, 6)
probabilidades = freq_table["Frequency"] / sum(freq_table["Frequency"])
bars = plt.bar(freq_table["Number"], probabilidades, color="tomato", edgecolor="black", width=2)  # Aumentamos el ancho de las barras
plt.title("Probabilidad de que los números ganadores salgan nuevamente", fontsize=16, weight='bold')
plt.xlabel("Número", fontsize=12)
plt.ylabel("Probabilidad", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Añadir etiquetas con el número sobre las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f'{int(bar.get_x())}', ha="center", va="bottom", fontsize=12, color="black")


plt.tight_layout()
plt.show()

# Resultados
print("Tabla de Frecuencias:")
print(freq_table)
