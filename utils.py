import pickle

def load_model(file_name):
    with open(file_name, "rb") as file:
        model = pickle.load(file)
        return model