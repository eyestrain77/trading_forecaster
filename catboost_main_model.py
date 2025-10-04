import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import warnings
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')


class MultiStockTimeSeriesCatBoost:
    """
    CatBoost implementation for multiple stock time series forecasting WITHOUT data leakage
    - only uses past information
    """

    def __init__(self, target_col, time_col, stock_col='stock_id', cat_cols=None,
                 max_lags=24, rolling_windows=[7, 14, 30], seasonal_periods=[7, 30, 252],
                 pct_change_periods=[1, 2, 3, 5, 7, 10, 15, 20, 40],
                 ewm_spans=[3, 5, 7, 10, 14, 21, 30, 60],
                 ewm_alphas=[0.1, 0.2, 0.3, 0.5], cross_stock_features=True,
                 catboost_params=None, n_jobs=-1):

        self.target_col = target_col
        self.time_col = time_col
        self.stock_col = stock_col
        self.cat_cols = cat_cols or []
        self.max_lags = max_lags
        self.rolling_windows = rolling_windows
        self.seasonal_periods = seasonal_periods
        self.cross_stock_features = cross_stock_features
        self.ewm_spans = ewm_spans
        self.ewm_alphas = ewm_alphas
        self.pct_change_periods = pct_change_periods
        self.n_jobs = n_jobs

        # Stock-specific models storage
        self.stock_models: Dict[str, CatBoostRegressor] = {}
        self.stock_feature_names: Dict[str, List[str]] = {}
        self.stock_scalers: Dict[str, StandardScaler] = {}
        self.stock_list: List[str] = []

        # Default CatBoost parameters
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'early_stopping_rounds': 100,
            'verbose': 50,
            'random_seed': 926,
            'l2_leaf_reg': 3,
            'bagging_temperature': 1,
            'bootstrap_type': 'Bayesian'
        }

        if catboost_params:
            default_params.update(catboost_params)
        self.catboost_params = default_params

    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime features - these are safe from leakage"""
        df_dt = df.copy()
        dt_series = pd.to_datetime(df_dt[self.time_col])

        # Basic datetime components
        df_dt['year'] = dt_series.dt.year
        df_dt['month'] = dt_series.dt.month
        df_dt['day'] = dt_series.dt.day
        df_dt['day_of_week'] = dt_series.dt.dayofweek
        df_dt['day_of_year'] = dt_series.dt.dayofyear
        df_dt['quarter'] = dt_series.dt.quarter
        df_dt['week_of_year'] = dt_series.dt.isocalendar().week

        # Financial market specific features
        df_dt['is_month_end'] = dt_series.dt.is_month_end.astype(int)
        df_dt['is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
        df_dt['is_year_end'] = dt_series.dt.is_year_end.astype(int)
        df_dt['days_to_month_end'] = (dt_series + pd.offsets.MonthEnd(0) - dt_series).dt.days

        # Cyclical encoding
        df_dt['month_sin'] = np.sin(2 * np.pi * df_dt['month'] / 12)
        df_dt['month_cos'] = np.cos(2 * np.pi * df_dt['month'] / 12)
        df_dt['day_of_week_sin'] = np.sin(2 * np.pi * df_dt['day_of_week'] / 7)
        df_dt['day_of_week_cos'] = np.cos(2 * np.pi * df_dt['day_of_week'] / 7)

        # Add to categorical columns if not already present
        financial_cats = ['month', 'day_of_week', 'quarter']
        for cat in financial_cats:
            if cat not in self.cat_cols:
                self.cat_cols.append(cat)

        return df_dt

    def create_target_based_features(self, stock_data: pd.DataFrame, stock_id: str) -> pd.DataFrame:
        """
        Create features that depend on target column - ONLY for training data
        """
        if len(stock_data) < max(self.max_lags, max(self.rolling_windows)):
            print(f"Warning: Stock {stock_id} has insufficient data for target-based feature creation")
            return stock_data

        # Lag features
        for lag in range(1, self.max_lags + 1):
            stock_data[f'{stock_id}_lag_{lag}'] = stock_data[self.target_col].shift(lag)

        # Percentage changes
        for period in self.pct_change_periods:
            if len(stock_data) > period:
                pct_change = stock_data[self.target_col].pct_change(period).shift(1)
                stock_data[f'{stock_id}_pct_change_{period}'] = pct_change

                # Cumulative percentage changes (momentum indicators)
                stock_data[f'{stock_id}_cum_pct_change_{period}'] = pct_change.rolling(period).sum()

        # Exponential weighted moving averages with spans
        for span in self.ewm_spans:
            if len(stock_data) > span:
                ewm_mean = stock_data[self.target_col].ewm(span=span, adjust=True).mean().shift(1)
                stock_data[f'{stock_id}_ewm_mean_{span}'] = ewm_mean

                ewm_std = stock_data[self.target_col].ewm(span=span, adjust=True).std().shift(1)
                stock_data[f'{stock_id}_ewm_std_{span}'] = ewm_std

                # EWM momentum indicator
                stock_data[f'{stock_id}_vs_ewm_{span}'] = (
                    stock_data[self.target_col].shift(1) / (ewm_mean + 1e-8)
                )

        # EWM with alpha parameters
        for alpha in self.ewm_alphas:
            ewm_alpha = stock_data[self.target_col].ewm(alpha=alpha, adjust=False).mean().shift(1)
            stock_data[f'{stock_id}_ewm_alpha_{int(alpha*100)}'] = ewm_alpha

            ewm_var = stock_data[self.target_col].ewm(alpha=alpha, adjust=False).var().shift(1)
            stock_data[f'{stock_id}_ewm_var_{int(alpha*100)}'] = ewm_var

        # Double and triple exponential smoothing approximation
        for span in [5, 10, 20]:
            if len(stock_data) > span:
                ema1 = stock_data[self.target_col].ewm(span=span).mean().shift(1)
                ema2 = ema1.ewm(span=span).mean().shift(1)
                dema = 2 * ema1 - ema2  # Double EMA
                stock_data[f'{stock_id}_dema_{span}'] = dema

                # Triple exponential smoothing approximation
                ema3 = ema2.ewm(span=span).mean().shift(1)
                tema = 3 * ema1 - 3 * ema2 + ema3  # Triple EMA
                stock_data[f'{stock_id}_tema_{span}'] = tema

        # Rolling statistics
        for window in self.rolling_windows:
            prefix = f'{stock_id}_rolling_{window}'

            # Rolling statistics (shift=1 to avoid look-ahead)
            stock_data[f'{prefix}_mean'] = stock_data[self.target_col].rolling(window).mean().shift(1)
            stock_data[f'{prefix}_std'] = stock_data[self.target_col].rolling(window).std().shift(1)
            stock_data[f'{prefix}_min'] = stock_data[self.target_col].rolling(window).min().shift(1)
            stock_data[f'{prefix}_max'] = stock_data[self.target_col].rolling(window).max().shift(1)
            stock_data[f'{prefix}_median'] = stock_data[self.target_col].rolling(window).median().shift(1)

            # Rolling percentage change statistics
            rolling_pct = stock_data[self.target_col].pct_change().rolling(window)
            stock_data[f'{prefix}_pct_mean'] = rolling_pct.mean().shift(1)
            stock_data[f'{prefix}_pct_std'] = rolling_pct.std().shift(1)

        return stock_data

    def create_non_target_features(self, df: pd.DataFrame, stock_id: str = None) -> pd.DataFrame:
        """
        Create features that DON'T depend on target column
        Only external features, datetime features, etc.
        """
        # Add datetime features
        df = self.create_datetime_features(df)

        # Add stock-specific non-target features if needed
        if stock_id:
            # Add stock identifier as categorical
            if 'stock_category' not in self.cat_cols:
                self.cat_cols.append('stock_category')
            df['stock_category'] = stock_id

            # Add time-based features specific to stock
            df['stock_time_index'] = range(len(df))

        return df

    def create_cross_stock_features_safe(self, df: pd.DataFrame, cutoff_time: pd.Timestamp = None) -> pd.DataFrame:
        """
        Create market-wide features using only historical data (no data leakage)
        """
        if not self.cross_stock_features:
            return df

        if cutoff_time is not None:
            historical_data = df[df[self.time_col] < cutoff_time].copy()
        else:
            historical_data = df.copy()

        market_features_list = []
        unique_times = sorted(df[self.time_col].unique())

        for current_time in unique_times:
            # Create market features using only historical data with proper temporal order
            past_data = historical_data[historical_data[self.time_col] < current_time]

            if len(past_data) == 0:
                market_stats = {
                    self.time_col: current_time,
                    'market_mean': 0,
                    'market_std': 1,
                    'market_median': 0
                }
            else:
                # Use last 30 observations for market statistics
                recent_window = 30
                recent_past = past_data.groupby(self.stock_col)[self.target_col].tail(recent_window)

                if len(recent_past) > 0:
                    market_stats = {
                        self.time_col: current_time,
                        'market_mean': recent_past.mean(),
                        'market_std': recent_past.std() if len(recent_past) > 1 else 1,
                        'market_median': recent_past.median()
                    }
                else:
                    market_stats = {
                        self.time_col: current_time,
                        'market_mean': 0,
                        'market_std': 1,
                        'market_median': 0
                    }

            market_features_list.append(market_stats)

        # Convert to DataFrame and merge
        market_df = pd.DataFrame(market_features_list)
        df = df.merge(market_df, on=self.time_col, how='left')
        return df

    def prepare_stock_data_safe(self, df: pd.DataFrame, stock_id: str, 
                              is_training: bool = True, cutoff_time: pd.Timestamp = None) -> Tuple:
        """
        Prepare data for a specific stock WITHOUT data leakage
        """
        # Filter data for specific stock
        stock_df = df[df[self.stock_col] == stock_id].copy()
        if len(stock_df) == 0:
            return None, None, None, None

        # Sort by time
        stock_df = stock_df.sort_values(self.time_col).reset_index(drop=True)

        # Create non-target features (always safe)
        stock_df = self.create_non_target_features(stock_df, stock_id)

        # Create target-based features ONLY for training
        if is_training and self.target_col in stock_df.columns:
            stock_df = self.create_target_based_features(stock_df, stock_id)

        # Remove rows with NaN values
        initial_rows = len(stock_df)
        stock_df = stock_df.dropna()
        removed_rows = initial_rows - len(stock_df)

        if removed_rows > 0:
            print(f"Stock {stock_id}: Removed {removed_rows} rows with NaN values")

        if len(stock_df) == 0:
            print(f"Warning: No valid data remaining for stock {stock_id}")
            return None, None, None, None

        # Prepare features and target
        feature_cols = [col for col in stock_df.columns 
                       if col not in [self.target_col, self.time_col, self.stock_col]]

        X = stock_df[feature_cols]
        y = stock_df[self.target_col] if self.target_col in stock_df.columns else None
        timestamps = stock_df[self.time_col]

        # Store feature names for this stock
        if is_training:
            self.stock_feature_names[stock_id] = feature_cols

        # Identify categorical features
        cat_indices = []
        for i, col in enumerate(feature_cols):
            if col in self.cat_cols:
                cat_indices.append(i)

        return X, y, timestamps, cat_indices

    def prepare_test_data_without_target(self, df_test: pd.DataFrame, df_train: pd.DataFrame, 
                                       stock_id: str) -> Tuple:
        """
        Prepare test data using only features that don't depend on target
        Uses trained feature names to ensure consistency
        """
        if stock_id not in self.stock_feature_names:
            raise ValueError(f"No trained model found for stock {stock_id}")

        # Get test data for this stock
        test_stock = df_test[df_test[self.stock_col] == stock_id].copy()
        if len(test_stock) == 0:
            return None, None, None

        # Sort by time
        test_stock = test_stock.sort_values(self.time_col).reset_index(drop=True)

        # Create only non-target features
        test_stock = self.create_non_target_features(test_stock, stock_id)

        # For target-dependent features, we need historical data
        train_stock = df_train[df_train[self.stock_col] == stock_id].copy()
        if len(train_stock) > 0:
            train_stock = train_stock.sort_values(self.time_col)

            # Combine train and test for feature creation
            combined_data = pd.concat([train_stock, test_stock], ignore_index=True)
            combined_data = combined_data.sort_values(self.time_col).reset_index(drop=True)

            # Create target-based features using combined data
            combined_data = self.create_target_based_features(combined_data, stock_id)

            # Extract only test portion
            test_start_idx = len(train_stock)
            test_stock = combined_data.iloc[test_start_idx:].reset_index(drop=True)

        # Select only features that were used in training
        expected_features = self.stock_feature_names[stock_id]
        available_features = [col for col in expected_features if col in test_stock.columns]

        if len(available_features) < len(expected_features):
            missing_features = set(expected_features) - set(available_features)
            print(f"Warning: Missing features for {stock_id}: {missing_features}")
            # Fill missing features with zeros or use forward fill from last train values
            for feature in missing_features:
                test_stock[feature] = 0

        X_test = test_stock[expected_features]
        timestamps = test_stock[self.time_col]

        # Identify categorical features
        cat_indices = []
        for i, col in enumerate(expected_features):
            if col in self.cat_cols:
                cat_indices.append(i)

        return X_test, timestamps, cat_indices

   
    def fit(self, df: pd.DataFrame, validation_split: float = 0.2, gap_days: int = None):
        """Train models WITHOUT data leakage using proper temporal split"""
        print("Starting multi-stock CatBoost training (leak-safe)...")
        self.stock_list = sorted(df[self.stock_col].unique())
        print(f"Training models for {len(self.stock_list)} stocks")
        
        if self.cross_stock_features:
            df = self.create_cross_stock_features_safe(df)
        # compute gap_days if not provided
        if gap_days is None:
            gap_days = max(self.max_lags, max(self.rolling_windows, default=0), max(self.ewm_spans, default=0), 5)
    
        training_results = {}
        total_mae = total_rmse = 0
    
        for stock_id in self.stock_list:
            print(f"\nTraining model for stock: {stock_id}")
            res = self.prepare_stock_data_safe(df, stock_id, is_training=True)
            if res[0] is None:
                training_results[stock_id] = {"status": "failed", "reason": "no_data"}
                continue
    
            X, y, timestamps, cat_indices = res
            if len(X) < 50:
                print(f"Warning: Insufficient data for {stock_id}")
                training_results[stock_id] = {"status": "failed", "reason": "insufficient_data"}
                continue
    
            # temporal split with gap
            times = pd.to_datetime(timestamps).sort_values().unique()
            split_idx = int(len(times) * (1 - validation_split))
            if split_idx + gap_days >= len(times):
                print(f"Warning: Not enough data after gap for {stock_id}")
                training_results[stock_id] = {"status": "failed", "reason": "no_val_after_gap"}
                continue
            train_end, val_start = times[split_idx - gap_days], times[split_idx]
            mask_tr = timestamps <= str(train_end).split()[0]
            mask_val = timestamps >= str(val_start).split()[0]

            X_train, X_val = X[mask_tr], X[mask_val]
            y_train, y_val = y[mask_tr], y[mask_val]
            if X_train.empty or X_val.empty:
                print(f"Warning: Empty split for {stock_id}")
                training_results[stock_id] = {"status": "failed", "reason": "empty_split"}
                continue
    
            # pools
            train_pool = Pool(X_train, y_train, cat_features=cat_indices)
            val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    
            # train & eval
            model = CatBoostRegressor(**self.catboost_params)
            model.fit(train_pool, eval_set=val_pool)
            preds = model.predict(val_pool)
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
    
            self.stock_models[stock_id] = model
            training_results[stock_id] = {
                "status": "success",
                "val_mae": mae,
                "val_rmse": rmse,
                "train_period": f"{train_end}",
                "val_period": f"{val_start}",
                "gap_days": gap_days
            }
            print(f"Stock {stock_id}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            total_mae += mae
            total_rmse += rmse
    
        avg_mae = total_mae / len(self.stock_models)
        avg_rmse = total_rmse / len(self.stock_models)
        print(f"\nTrained {len(self.stock_models)}/{len(self.stock_list)} models. Avg MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}")



    def predict(self, df_test: pd.DataFrame, df_train: pd.DataFrame = None, bin_model = None) -> pd.DataFrame:
        """
        Make predictions for test data WITHOUT using target column
        """
        if not self.stock_models:
            raise ValueError("Models must be trained before making predictions")

        all_predictions = []

        for stock_id in self.stock_list:
            if stock_id not in self.stock_models:
                print(f"Warning: No trained model for stock {stock_id}")
                continue

            print(f"Predicting for stock: {stock_id}")

            # Prepare test data without using target
            if df_train is not None:
                result = self.prepare_test_data_without_target(df_test, df_train, stock_id)
            else:
                # Use only non-target features
                result = self.prepare_stock_data_safe(df_test, stock_id, is_training=False)

            if result[0] is None:
                print(f"Warning: No test data available for stock {stock_id}")
                continue

            if df_train is not None:
                X_test, timestamps, cat_indices = result
            else:
                X_test, _, timestamps, cat_indices = result
            # Make predictions
                        # Исправление категориальных фичей перед созданием Pool
            categorical_columns = ['year', 'month', 'day_of_week', 'quarter', 'stock_category', 
                                  'is_month_end', 'is_quarter_end', 'is_year_end']
            
            for col in categorical_columns:
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype(str).fillna('missing')
                    X_test[col] = X_test[col].replace(['nan', 'None'], 'missing')
            
            # Используйте имена колонок для категориальных фичей
            cat_features = [col for col in categorical_columns if col in X_test.columns]
            pred_pool = Pool(data=X_test, cat_features=cat_features)  # Имена, не индексы!

            pred_pool = Pool(data=X_test, cat_features=cat_indices)
            predictions_values = self.stock_models[stock_id].predict(pred_pool)

            # Create result DataFrame
            stock_predictions = pd.DataFrame({
                'i': X_test['i'],
                self.stock_col: stock_id,
                self.time_col: timestamps,
                'predicted_value': predictions_values
            })

            all_predictions.append(stock_predictions)

        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_feature_importance(self, stock_id: str = None, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for specific stock or average across all stocks"""
        if not self.stock_models:
            raise ValueError("Models must be trained first")

        if stock_id and stock_id in self.stock_models:
            importance = self.stock_models[stock_id].get_feature_importance()
            feature_names = self.stock_feature_names[stock_id]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'stock': stock_id
            }).sort_values('importance', ascending=False)
        else:
            # Average importance across all stocks
            all_importances = []

            for stock_id, model in self.stock_models.items():
                importance = model.get_feature_importance()
                feature_names = self.stock_feature_names[stock_id]

                stock_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance,
                    'stock': stock_id
                })
                all_importances.append(stock_importance)

            if all_importances:
                combined_importance = pd.concat(all_importances)
                importance_df = combined_importance.groupby('feature')['importance'].mean().reset_index()
                importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df.head(top_n) 