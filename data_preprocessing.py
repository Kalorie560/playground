"""
Data preprocessing module for Spaceship Titanic competition.
Handles loading, cleaning, feature engineering, and encoding of the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpaceshipDataProcessor:
    """Data processor for Spaceship Titanic dataset."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = 'Transported'
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test datasets."""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")
            return train_df, test_df
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            # Create sample data structure for demonstration
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create sample data with the expected structure for demonstration."""
        np.random.seed(42)
        n_train, n_test = 1000, 500
        
        # Sample data structure based on Spaceship Titanic competition
        train_data = {
            'PassengerId': [f'{i:04d}_01' for i in range(n_train)],
            'HomePlanet': np.random.choice(['Europa', 'Earth', 'Mars'], n_train),
            'CryoSleep': np.random.choice([True, False], n_train),
            'Cabin': [f'{np.random.choice(["A", "B", "C"])}/{np.random.randint(1, 100)}/{np.random.choice(["P", "S"])}' 
                     for _ in range(n_train)],
            'Destination': np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], n_train),
            'Age': np.random.normal(30, 15, n_train).clip(0, 100),
            'VIP': np.random.choice([True, False], n_train),
            'RoomService': np.random.exponential(100, n_train),
            'FoodCourt': np.random.exponential(150, n_train),
            'ShoppingMall': np.random.exponential(80, n_train),
            'Spa': np.random.exponential(120, n_train),
            'VRDeck': np.random.exponential(200, n_train),
            'Name': [f'Person {i}' for i in range(n_train)],
            'Transported': np.random.choice([True, False], n_train)
        }
        
        test_data = train_data.copy()
        for key in test_data:
            if key == 'Transported':
                continue
            if key == 'PassengerId':
                test_data[key] = [f'{i:04d}_01' for i in range(n_train, n_train + n_test)]
            elif isinstance(test_data[key], list):
                if key == 'Name':
                    test_data[key] = [f'Person {i}' for i in range(n_train, n_train + n_test)]
                elif key == 'Cabin':
                    test_data[key] = [f'{np.random.choice(["A", "B", "C"])}/{np.random.randint(1, 100)}/{np.random.choice(["P", "S"])}' 
                                     for _ in range(n_test)]
                else:
                    test_data[key] = [test_data[key][i % len(test_data[key])] for i in range(n_test)]
            else:
                test_data[key] = test_data[key][:n_test] if len(test_data[key]) > n_test else np.resize(test_data[key], n_test)
        
        # Remove target from test data
        del test_data['Transported']
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        logger.info("Created sample data for demonstration")
        return train_df, test_df
    
    def handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Handle numerical columns
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in numerical_cols:
            if col in df.columns:
                if is_training:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Use stored median from training
                    median_val = getattr(self, f'{col}_median', df[col].median())
                    df[col] = df[col].fillna(median_val)
                    
        # Handle categorical columns
        categorical_cols = ['HomePlanet', 'Destination']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Handle boolean columns
        boolean_cols = ['CryoSleep', 'VIP']
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform advanced feature engineering on the dataset."""
        df = df.copy()
        
        # Extract cabin information
        if 'Cabin' in df.columns:
            df['Cabin'] = df['Cabin'].fillna('Unknown/0/Unknown')
            cabin_parts = df['Cabin'].str.split('/', expand=True)
            df['Deck'] = cabin_parts[0].fillna('Unknown')
            df['Cabin_num'] = pd.to_numeric(cabin_parts[1], errors='coerce').fillna(0)
            df['Side'] = cabin_parts[2].fillna('Unknown')
            
            # Advanced cabin features
            df['Cabin_num_binned'] = pd.cut(df['Cabin_num'], bins=10, labels=False).fillna(0)
            df['Is_premium_deck'] = (df['Deck'].isin(['A', 'B', 'T'])).astype(int)
            df['Is_port_side'] = (df['Side'] == 'P').astype(int)
            
            df.drop('Cabin', axis=1, inplace=True)
        
        # Extract family information from PassengerId
        if 'PassengerId' in df.columns:
            df['Group'] = df['PassengerId'].str.split('_').str[0]
            df['Group_size'] = df.groupby('Group')['Group'].transform('count')
            df['Person_in_group'] = df['PassengerId'].str.split('_').str[1].astype(int)
            df.drop('PassengerId', axis=1, inplace=True)
        
        # Enhanced family size features
        if 'Group_size' in df.columns:
            df['Is_solo'] = (df['Group_size'] == 1).astype(int)
            df['Is_small_group'] = ((df['Group_size'] >= 2) & (df['Group_size'] <= 4)).astype(int)
            df['Is_large_group'] = (df['Group_size'] > 4).astype(int)
            df['Group_size_log'] = np.log1p(df['Group_size'])
        
        # Advanced spending analysis
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        existing_spending_cols = [col for col in spending_cols if col in df.columns]
        if existing_spending_cols:
            df['Total_spending'] = df[existing_spending_cols].sum(axis=1)
            df['Has_spending'] = (df['Total_spending'] > 0).astype(int)
            df['Spending_per_service'] = df['Total_spending'] / len(existing_spending_cols)
            df['Total_spending_log'] = np.log1p(df['Total_spending'])
            
            # Spending ratios
            if 'Total_spending' in df.columns and df['Total_spending'].sum() > 0:
                for col in existing_spending_cols:
                    df[f'{col}_ratio'] = df[col] / (df['Total_spending'] + 1e-8)
            
            # High spender features
            spending_threshold = df['Total_spending'].quantile(0.75) if len(df) > 0 else 1000
            df['Is_high_spender'] = (df['Total_spending'] > spending_threshold).astype(int)
            
            # Service preferences
            df['Luxury_spending'] = df[['Spa', 'VRDeck']].sum(axis=1) if 'Spa' in df.columns and 'VRDeck' in df.columns else 0
            df['Basic_spending'] = df[['RoomService', 'FoodCourt']].sum(axis=1) if 'RoomService' in df.columns and 'FoodCourt' in df.columns else 0
        
        # Enhanced age features
        if 'Age' in df.columns:
            df['Age_group'] = pd.cut(df['Age'], 
                                   bins=[0, 12, 18, 25, 35, 50, 65, 100], 
                                   labels=['Child', 'Teen', 'Young', 'Adult', 'Middle_age', 'Senior', 'Elder'])
            df['Age_group'] = df['Age_group'].astype(str)
            
            # Age polynomials
            df['Age_squared'] = df['Age'] ** 2
            df['Age_cubed'] = df['Age'] ** 3
            df['Age_log'] = np.log1p(df['Age'])
            
            # Age-related patterns
            df['Is_minor'] = (df['Age'] < 18).astype(int)
            df['Is_senior'] = (df['Age'] >= 60).astype(int)
        
        # CryoSleep interaction features
        if 'CryoSleep' in df.columns:
            if 'Total_spending' in df.columns:
                df['CryoSleep_spending_interaction'] = df['CryoSleep'].astype(int) * df['Total_spending']
            if 'VIP' in df.columns:
                df['CryoSleep_VIP_interaction'] = df['CryoSleep'].astype(int) * df['VIP'].astype(int)
        
        # VIP interaction features
        if 'VIP' in df.columns:
            if 'Total_spending' in df.columns:
                df['VIP_spending_interaction'] = df['VIP'].astype(int) * df['Total_spending']
            if 'Age' in df.columns:
                df['VIP_age_interaction'] = df['VIP'].astype(int) * df['Age']
        
        # Planet-destination interactions
        if 'HomePlanet' in df.columns and 'Destination' in df.columns:
            df['Planet_destination'] = df['HomePlanet'].astype(str) + '_to_' + df['Destination'].astype(str)
        
        # Drop name as it's not useful for prediction
        if 'Name' in df.columns:
            df.drop('Name', axis=1, inplace=True)
            
        # Drop group as we have group size features
        if 'Group' in df.columns:
            df.drop('Group', axis=1, inplace=True)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Age_group', 'Planet_destination']
        
        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        unique_values = set(df[col].astype(str).unique())
                        known_values = set(le.classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            logger.warning(f"Unknown categories in {col}: {unknown_values}")
                            # Map unknown values to the most frequent class
                            most_frequent = le.classes_[0]
                            df[col] = df[col].astype(str).replace(list(unknown_values), most_frequent)
                        
                        df[col] = le.transform(df[col].astype(str))
        
        # Convert boolean columns to integers
        boolean_cols = ['CryoSleep', 'VIP']
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        # Features to scale - including new engineered features
        features_to_scale = [
            'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
            'Cabin_num', 'Group_size', 'Total_spending', 'Spending_per_service',
            'Total_spending_log', 'Age_squared', 'Age_cubed', 'Age_log',
            'Group_size_log', 'Person_in_group', 'Luxury_spending', 'Basic_spending',
            'CryoSleep_spending_interaction', 'VIP_spending_interaction', 'VIP_age_interaction'
        ]
        
        # Add ratio features dynamically
        ratio_features = [col for col in df.columns if col.endswith('_ratio')]
        features_to_scale.extend(ratio_features)
        
        existing_features = [col for col in features_to_scale if col in df.columns]
        
        if existing_features:
            if is_training:
                df[existing_features] = self.scaler.fit_transform(df[existing_features])
            else:
                df[existing_features] = self.scaler.transform(df[existing_features])
        
        return df
    
    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None, 
                  validation_split: float = 0.2) -> Dict[str, Any]:
        """Complete preprocessing pipeline."""
        
        # Store target variable
        y_train = train_df[self.target_column].astype(int)
        X_train = train_df.drop(self.target_column, axis=1)
        
        # Preprocess training data
        X_train = self.handle_missing_values(X_train, is_training=True)
        X_train = self.engineer_features(X_train)
        X_train = self.encode_categorical_features(X_train, is_training=True)
        
        # Store medians for test preprocessing
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in numerical_cols:
            if col in X_train.columns:
                setattr(self, f'{col}_median', X_train[col].median())
        
        X_train = self.scale_features(X_train, is_training=True)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Split training data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
        )
        
        result = {
            'X_train': X_train_split,
            'X_val': X_val,
            'y_train': y_train_split,
            'y_val': y_val,
            'feature_names': self.feature_names
        }
        
        # Preprocess test data if provided
        if test_df is not None:
            X_test = self.handle_missing_values(test_df, is_training=False)
            X_test = self.engineer_features(X_test)
            X_test = self.encode_categorical_features(X_test, is_training=False)
            X_test = self.scale_features(X_test, is_training=False)
            
            # Ensure same columns as training
            missing_cols = set(self.feature_names) - set(X_test.columns)
            for col in missing_cols:
                X_test[col] = 0
            
            X_test = X_test[self.feature_names]
            result['X_test'] = X_test
        
        logger.info(f"Preprocessing complete. Training shape: {X_train_split.shape}")
        return result


def main():
    """Example usage of the data processor."""
    processor = SpaceshipDataProcessor()
    
    # Load data
    train_df, test_df = processor.load_data('train.csv', 'test.csv')
    
    # Preprocess data
    data = processor.preprocess(train_df, test_df)
    
    print(f"Training set: {data['X_train'].shape}")
    print(f"Validation set: {data['X_val'].shape}")
    if 'X_test' in data:
        print(f"Test set: {data['X_test'].shape}")
    print(f"Features: {len(data['feature_names'])}")


if __name__ == "__main__":
    main()