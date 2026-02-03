import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def train_pipeline():
    try:
        logging.info("Starting model training pipeline...")
        
        # Step 1: Data Ingestion
        logging.info("Initiating data ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train: {train_data_path}, Test: {test_data_path}")
        
        # Step 2: Data Transformation
        logging.info("Initiating data transformation...")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation completed")
        
        # Step 3: Model Training
        logging.info("Initiating model training...")
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed. Result: {result}")
        
        print("✓ Model training pipeline completed successfully!")
        print(f"Model saved to: artifacts/model.pkl")
        print(f"Preprocessor saved to: artifacts/preprocessor.pkl")
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    train_pipeline()