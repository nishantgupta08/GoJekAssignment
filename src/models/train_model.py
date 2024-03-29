
import sys
sys.path.append('../../')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    #Not using normal train test split (validation done across timestamps) 
    # df_train, df_test = train_test_split(df, test_size=config["test_size"])
    
    df_train = store.get_processed("transformed_dataset_train.csv")
    df_test = store.get_processed("transformed_dataset_test.csv")
    

    rf_estimator = RandomForestClassifier(**config["random_forest"])
    model = SklearnClassifier(rf_estimator, config["features"], config["target"])
    model.train(df_train)

    metrics = model.evaluate(df_test)

    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
