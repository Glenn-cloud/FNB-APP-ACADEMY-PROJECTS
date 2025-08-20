import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import ttk, messagebox

# ----------------- Load & Prepare Data -----------------
try:
    df = pd.read_csv("student_Accommodation_dataset.csv")
except FileNotFoundError:
    raise FileNotFoundError("The file 'student_Accommodation_dataset.csv' was not found. Please ensure it exists in the working directory.")
except Exception as e:
    raise Exception(f"An error occurred while loading the CSV file: {e}")

categorical_cols = ['preferred_campus', 'accomodation_type', 'room_type',
                    'safety_priority', 'distance_priority']

encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

scaler = MinMaxScaler()
df[['monthly_budget']] = scaler.fit_transform(df[['monthly_budget']])

features = ['preferred_campus', 'accomodation_type', 'room_type', 'monthly_budget',
            'safety_priority', 'distance_priority', 'high_speed_wifi', 'secure_parking',
            'laundry_facilities', 'kitchen_access', 'security_24_7', 'gym_access',
            'study_areas', 'public_transport']

X = df[features]

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X)

# ----------------- Recommendation Function -----------------
def recommend_accommodation(user_input):
    for col in categorical_cols:
        user_input[col] = encoders[col].transform([user_input[col]])[0]
    user_input['monthly_budget'] = scaler.transform([[user_input['monthly_budget']]])[0][0]
    user_df = pd.DataFrame([user_input])[features]
    distances, indices = knn.kneighbors(user_df)
    return df.iloc[indices[0]]

# ----------------- GUI -----------------
root = tk.Tk()
root.title("Student Accommodation Recommender")
root.geometry("600x700")

# Dropdown options
campus_options = df['preferred_campus'].unique()
campus_options = encoders['preferred_campus'].inverse_transform(campus_options)

accommodation_options = df['accomodation_type'].unique()
accommodation_options = encoders['accomodation_type'].inverse_transform(accommodation_options)

room_options = df['room_type'].unique()
room_options = encoders['room_type'].inverse_transform(room_options)

priority_levels = ["low", "medium", "high"]
distance_levels = ["close", "moderate", "flexible"]

# Inputs
tk.Label(root, text="Preferred Campus").pack()
campus_var = ttk.Combobox(root, values=list(campus_options))
campus_var.pack()

tk.Label(root, text="Accommodation Type").pack()
accommodation_var = ttk.Combobox(root, values=list(accommodation_options))
accommodation_var.pack()

tk.Label(root, text="Room Type").pack()
room_var = ttk.Combobox(root, values=list(room_options))
room_var.pack()

tk.Label(root, text="Monthly Budget").pack()
budget_var = tk.Entry(root)
budget_var.pack()

tk.Label(root, text="Safety Priority").pack()
safety_var = ttk.Combobox(root, values=priority_levels)
safety_var.pack()

tk.Label(root, text="Distance Priority").pack()
distance_var = ttk.Combobox(root, values=distance_levels)
distance_var.pack()

# Amenities
amenities = ["high_speed_wifi", "secure_parking", "laundry_facilities",
             "kitchen_access", "security_24", "gym_access", "study_areas", "public_transport"]
amenity_vars = {}
for amenity in amenities:
    var = tk.IntVar()
    chk = tk.Checkbutton(root, text=amenity.replace("_", " ").title(), variable=var)
    chk.pack(anchor="w")
    amenity_vars[amenity] = var

# ----------------- On Submit -----------------
def submit():
    try:
        user_input = {
            "preferred_campus": campus_var.get(),
            "accomodation_type": accommodation_var.get(),
            "room_type": room_var.get(),
            "monthly_budget": float(budget_var.get()),
            "safety_priority": safety_var.get(),
            "distance_priority": distance_var.get()
        }
        for amenity in amenities:
            user_input[amenity] = amenity_vars[amenity].get()

        recommendations = recommend_accommodation(user_input)
        messagebox.showinfo("Recommendations", recommendations.to_string())
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="Find Accommodation", command=submit).pack(pady=10)

root.mainloop()
