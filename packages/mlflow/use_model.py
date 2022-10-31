import logging
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def use_model():
    with mlflow.start_run():
        logged_model = 'runs:/56bb0519598144e6b588c1acb87e2adb/model'

        # Load model as a PyFuncModel.
        clf = mlflow.pyfunc.load_model(logged_model)

        # Read data
        df = pd.read_csv('../optuna/cover_type.csv')

        # Split data
        logger.info('Splitting data')
        X = df.drop(['target'], axis=1)
        y = df.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=23
        )

        # mlflow.log_params(clf.get_params())

        # Log results
        accuracy = accuracy_score(y_train, clf.predict(X_train))
        mlflow.log_metric('Train accuracy', accuracy)

        accuracy = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric('Test accuracy', accuracy)

if __name__ == '__main__':
    use_model()