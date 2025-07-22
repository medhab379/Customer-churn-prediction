import pickle

# Load churn_features.pkl
with open("churn_features.pkl", "rb") as f:
    features = pickle.load(f)

# Print all feature names
print("Features used to train the model:")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")
