import pandas as pd

# --- 1. Cargar el archivo CSV ---
# Asegúrate de que el archivo tenga las siguientes columnas como mínimo:
# BMI, Daily_Steps, Exercise_Hours_per_Week, Hours_of_Sleep, Smoker,
# Alcohol_Consumption_per_Week, Diabetic, Heart_Disease

# Reemplaza con la ruta a tu archivo
input_file = "Datasets\health_activity_data.csv"
df = pd.read_csv(input_file)

# --- 2. Clasificación del estado de salud general ---
def clasificar_estado_salud_ponderado(row):
    puntaje_riesgo = 0

    # Enfermedades crónicas diagnosticadas
    if str(row['Diabetic']).lower() in ['yes', 'si', '1', 'true']:
        puntaje_riesgo += 3
    if str(row['Heart_Disease']).lower() in ['yes', 'si', '1', 'true']:
        puntaje_riesgo += 3

    # Fumador
    if str(row['Smoker']).lower() in ['yes', 'si', '1', 'true']:
        puntaje_riesgo += 2

    # Alcohol alto
    if row['Alcohol_Consumption_per_Week'] > 4:
        puntaje_riesgo += 1.5

    # BMI fuera de rango saludable
    if row['BMI'] < 18.5 or row['BMI'] > 30:
        puntaje_riesgo += 2

    # Poco ejercicio
    if row['Exercise_Hours_per_Week'] < 1:
        puntaje_riesgo += 2

    # Pocos pasos diarios
    if row['Daily_Steps'] < 5000:
        puntaje_riesgo += 1
        
        
    # Calorías elevadas con baja actividad: penaliza
    if row['Calories_Intake'] > 2800:
        if row['Exercise_Hours_per_Week'] < 3 and row['Daily_Steps'] < 5000:
            puntaje_riesgo += 1

    # Poco sueño
    if row['Hours_of_Sleep'] < 6:
        puntaje_riesgo += 1

    # Clasificación final (ajustada)
    if puntaje_riesgo <= 2:
        return "Bueno"
    elif puntaje_riesgo <= 5:
        return "Regular"
    else:
        return "Malo"

# --- 3. Aplicar la función al DataFrame ---
df['Estado_Salud'] = df.apply(clasificar_estado_salud_ponderado, axis=1)

# --- 4. Guardar el resultado ---
output_file = "pacientes_clasificados.csv"
df.to_csv(output_file, index=False)

print(f"Archivo generado: {output_file}")
print(df[['ID', 'Estado_Salud']].head())
