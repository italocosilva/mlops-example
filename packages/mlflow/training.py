import logging
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models import infer_signature

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)
    

def train_model(args):
    with mlflow.start_run():
        # Read data
        logger.info('Reading data')
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
        
        # Train model
        logger.info('Training model')
        clf = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=23
        )
        mlflow.log_params(clf.get_params())
        clf.fit(X_train, y_train)

        # Log results
        accuracy = accuracy_score(y_train, clf.predict(X_train))
        mlflow.log_metric('Train accuracy', accuracy)

        accuracy = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric('Test accuracy', accuracy)

        # Log model
        signature = infer_signature(X_train, clf.predict(X_train))
        mlflow.sklearn.log_model(
            clf,
            'model',
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_train.iloc[:2],
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tree parameters'
    )

    parser.add_argument(
        '--max_depth',
        type=int,
        help='Maximum tree depth',
        required=True
    )

    parser.add_argument(
        '--min_samples_split',
        type=int,
        help='Minimum node samples for new split',
        required=True
    )

    args = parser.parse_args()

    train_model(args)