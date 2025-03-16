# %%
import pandas as pd

# Φόρτωση του dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Εμφάνιση των πρώτων 5 γραμμών
print(df.head())

# Επισκόπηση τύπων δεδομένων
print(df.info())

# Έλεγχος για ελλείπουσες τιμές
print(df.isnull().sum())

# Εμφάνιση αριθμού στηλών
I = len(df.columns)
print(f"Αριθμός στηλών {len(df.columns)}")


# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Αφαίρεση μη χρήσιμων στηλών
df_cleaned = df.drop(columns=["PatientID", "DoctorInCharge"])

# Διαχωρισμός χαρακτηριστικών (X) και στόχου (y)
X = df_cleaned.drop(columns=["Diagnosis"])
y = df_cleaned["Diagnosis"]

# Εμφάνιση νέου αριθμού στηλών
I = len(X.columns)
print(f"Αριθμός στηλών για είσοδο {I}")


# Κανονικοποίηση αριθμητικών χαρακτηριστικών
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Διαχωρισμός σε σύνολα εκπαίδευσης και ελέγχου (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Εκτύπωση μεγεθών συνόλων
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

H  = I//2  # Αριθμός νευρώνων στο κρυφό επίπεδο

# Δημιουργία του μοντέλου
model = Sequential([
    Dense(H, activation='relu', input_shape=(X_train.shape[1],)),  # Κρυφό επίπεδο
    Dense(1, activation='sigmoid')  # Έξοδος για binary classification
])

# Σύνταξη του μοντέλου
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Εκτύπωση περίληψης του μοντέλου
model.summary()


# %%
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, val_index in kf.split(X_train):
    print(f"Training fold {fold}...")
    
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Εκπαίδευση
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, validation_data=(X_val_fold, y_val_fold), verbose=1)
    
    fold += 1



