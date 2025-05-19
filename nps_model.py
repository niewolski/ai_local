import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pylab as plt

df = pd.read_csv('NPStimeseries.csv')
df.groupby(["Month"])["NPS"].mean()

# Grupowanie po miesiącu i liczenie średniego NPS
srednia_nps_miesiac = df.groupby("Month")["NPS"].mean()

# Wyświetlenie wyników
print(srednia_nps_miesiac)


# tworzymy dane do przewidywania przyszlych miesiecy
przyszlosc = pd.DataFrame({
    'Day': [15]*8,  # srodkowy dzień miesiaca
    'Month': list(range(5, 13)),  # miesiace od maja do grudnia
    'Quarter': [2, 2, 3, 3, 3, 4, 4, 4],
    'Market_MEX': [0]*8,
    'Market_UK': [0]*8,
    'Market_US': [1]*8,  # tylko US
})


print(df.head()) # pokazanie pierwszych 5 wierszy
print(df.info()) # informacje o danych
print(df.describe()) # statystyki liczbowe
print(df.isnull().sum()) # sprawdza brakujace dane

df = pd.get_dummies(df, columns=['Market'])

df["Survey date"] = pd.to_datetime(df["Survey date"], dayfirst = True)
df["Month"] = df["Survey date"].dt.month
df["Day"] = df["Survey date"].dt.day

df = df.drop(columns=["Survey date"])
df = df.drop(columns=["Customer Name"])
df = df.drop(columns=["ID"])

df = df.reindex(columns=['Day', 'Month', 'Quarter', 'Market_MEX', 'Market_UK', 'Market_US', 'NPS'])
print(df.head())

X = df.drop('NPS', axis=1) # wszystkie kolumny poza NPS
y = df['NPS'] # kolumna, którą chcesz przewidzieć

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# przewidywanie dla przyszlosci
nps_future = model.predict(przyszlosc)

# wyswietlanie wynikow
for miesiac, prognoza in zip(przyszlosc['Month'], nps_future):
    print(f"Prognozowany NPS w miesiącu {miesiac}: {prognoza:.2f}")


y_pred =  model.predict(X_test) # przewidywanie

# porownanie przewidywan z rzeczywistoscia
print('blad sredniokwadratowy: ', mean_squared_error(y_test, y_pred))
print('r^2 score (skutecnzosc)', r2_score(y_test, y_pred))

plt.plot(przyszlosc["Month"], nps_future, marker="o")
plt.xlabel("Miesiąc")
plt.ylabel("Prognozowany NPS")
plt.title("Prognozowany NPS w kolejnych miesiącach (USA)")
plt.grid()
plt.show()
