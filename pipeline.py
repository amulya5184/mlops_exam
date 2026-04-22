from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from prefect import get_run_logger

@task
def evaluate(model, X_test, y_test):
    logger = get_run_logger()
    accuracy = model.score(X_test, y_test)
    logger.info(f"Accuracy: {accuracy}")

@task
def load_data():
    data = load_iris()
    return data.data, data.target

@task
def preprocess(X, y):
    return train_test_split(X, y, test_size=0.2)

@task
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

@task
def evaluate(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

@flow
def ml_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess(X, y)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    ml_pipeline()