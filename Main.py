# Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data frame pour les données
data = {
    'Superficie' : [50, 60, 70, 80, 90, 100],
    'Prix' : [100000, 120000, 140000, 160000, 180000, 200000]
}
df = pd.DataFrame(data)

X = df[['Superficie']]
Y = df[['Prix']]

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X, Y)

# Prédire le prix pour une superficie de 200 m²
superficie_nouvelle = np.array([[200]])  # 200 m²
prix_predits = model.predict(superficie_nouvelle)

# Affichage du prix prédit
print(f"Le prix prédit pour une superficie de 200 m² est : {prix_predits[0].item() : .2f} euro")

# Afficher le DataFrame
print(df)

# Tracer la ligne de régression
plt.scatter(df['Superficie'], df['Prix'], color='blue')
plt.plot(df['Superficie'], model.predict(X), color='red')
plt.title('Superficie vs Prix')
plt.xlabel('Superficie (m²)')
plt.ylabel('Prix (€)')
plt.show()
