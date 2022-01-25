import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('salarios.csv')
print(dataset.head())
print(dataset.columns)
print(dataset.shape)

x = dataset[['Aexperiencia', 'Pais']]
y = dataset['Salario']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

linear_model = LinearRegression()


linear_model.fit(x_train, y_train)
linear_model.score(x_test, y_test)

figura = plt.figure(dpi=100)
etiq = figura.add_subplot(111, projection='3d')
etiq.scatter(x_train['Aexperiencia'], x_train['Pais'],
             y_train, color='#17becf', marker='v')
etiq.scatter(x_test['Aexperiencia'], x_test['Pais'],
             linear_model.predict(x_test), color='#7f7f7f', marker='*')
etiq.set_zlabel('Salario')
etiq.set_xlabel('Experiencia')
etiq.set_ylabel('Pais')
plt.show()
print("Modelo tiene un porcentarce de score: ",
      linear_model.score(x_test, y_test))
