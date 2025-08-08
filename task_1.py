import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from tqdm.notebook import tqdm


class DataAugmentationPipeline:
    """A class to manage data loading, preprocessing, feature selection, and model training for olfactory data with configurable hyperparameters."""

    def __init__(self, stimulus_file: str, training_file: str, descriptors_file: str = None, enrich_data: bool = False,
                 rf_hyperparams: dict = None,
                 use_meta_parameters=False):
        """
        Initialize the pipeline with data files, enrichment option, and Random Forest hyperparameters.

        Args:
            stimulus_file (str): Path to TASK1_Stimulus_definition.csv
            training_file (str): Path to TASK1_training.csv
            descriptors_file (str, optional): Path to Mordred_Descriptors.csv for feature enrichment
            enrich_data (bool): Whether to enrich features with molecular descriptors
            rf_hyperparams (dict, optional): Hyperparameters for RandomForestClassifier (e.g., {'n_estimators': 200, 'max_depth': None, 'random_state': 42})
        """
        self.stimulus_file = stimulus_file
        self.training_file = training_file
        self.descriptors_file = descriptors_file
        self.enrich_data = enrich_data
        self.rf_hyperparams = rf_hyperparams or {
            'n_estimators': 300,
            'random_state': 42
        }  # Default hyperparameters
        self.X = None  # Original features
        self.X_enriched = None  # Enriched features (if applicable)
        self.y = None  # Target labels
        self.final_model = None  # Trained Random Forest model
        self.feature_importances = None  # Feature importances from final model
        self.cv_results = None  # Store cross-validation results
        self.use_meta_parameters = use_meta_parameters
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """
        Load and preprocess data, aligning stimuli and optionally enriching with molecular descriptors.
        """
        # Load training data
        training_df = pd.read_csv(self.training_file)

        # Check if Intensity_label is in training file and we don't need meta parameters
        has_intensity_label = 'Intensity_label' in training_df.columns
        skip_stimulus_file = has_intensity_label and not self.use_meta_parameters

        if skip_stimulus_file:
            print("Using Intensity_label from training file directly - skipping stimulus definition file")

            # Extract features (drop target and other non-feature columns)
            columns_to_drop = ["Intensity", "Pleasantness", "Intensity_label"]
            columns_to_drop = [col for col in columns_to_drop if col in training_df.columns]
            self.X = training_df.drop(columns=columns_to_drop).set_index("stimulus")

            # Extract target directly from training file
            self.y = training_df.set_index("stimulus")["Intensity_label"]

        else:
            print("Loading stimulus definition file for Intensity_label and/or meta parameters")

            # Load stimulus definition file
            stimulus_definition_df = pd.read_csv(self.stimulus_file)

            # Filter X to only stimuli that appear in Y
            if self.use_meta_parameters:
                self.X = training_df.set_index("stimulus")
                self.X = self.X.join(stimulus_definition_df.set_index('stimulus')[['dilution', 'molecule']])
                self.X = pd.get_dummies(self.X, columns=['solvent'], prefix='solvent', dummy_na=False,
                                        dtype=int) if 'solvent' in self.X.columns else self.X
            else:
                self.X = training_df.drop(columns=["Intensity", "Pleasantness"]).set_index("stimulus")

                # FIXED: Only keep stimuli that exist in both X and stimulus_definition_df
            available_stimuli = set(self.X.index) & set(stimulus_definition_df['stimulus'])
            print(f"Found {len(available_stimuli)} stimuli present in both training and stimulus definition files")

            # Filter X to only available stimuli
            self.X = self.X.loc[list(available_stimuli)]

            # Filter y to only available stimuli
            self.y = stimulus_definition_df[stimulus_definition_df["stimulus"].isin(available_stimuli)].copy()

            # Set 'stimulus' as index and align - this should now work without KeyError
            self.y = self.y.set_index("stimulus").loc[self.X.index]

        if self.enrich_data and self.descriptors_file:
            # For enrichment, we need molecule info
            if skip_stimulus_file:
                print("Warning: Cannot enrich data without stimulus definition file (need molecule mapping)")
                print("Proceeding without enrichment...")
                self.X_enriched = self.X.copy()
            else:
                # Load molecular descriptors
                mordred_df = pd.read_csv(self.descriptors_file, encoding='latin-1')

                # Create a mapping from stimulus to molecule
                stimuli_molecules = self.y[['molecule']].copy()

                # Set molecule ID as index for mordred_df
                if 'molecule' in mordred_df.columns:
                    mordred_df = mordred_df.set_index('molecule')

                # Convert column names to strings and drop SMILES
                mordred_df.columns = mordred_df.columns.astype(str)
                if 'SMILES' in mordred_df.columns:
                    mordred_df = mordred_df.drop(columns='SMILES')

                # Merge molecular descriptors
                molecular_features = stimuli_molecules.join(mordred_df, on='molecule', how='left')
                molecular_features = molecular_features.drop(columns=['molecule'])

                # Merge with original features
                assert all(self.X.index == molecular_features.index), "Indices don't match!"
                self.X_enriched = pd.concat([self.X, molecular_features], axis=1)

                # Handle NaN values
                self.X_enriched = self.X_enriched.fillna(self.X_enriched.mean())
        else:
            self.X_enriched = self.X.copy()

        # Extract target labels (handle both paths)
        if not skip_stimulus_file and hasattr(self.y, 'columns') and 'Intensity_label' in self.y.columns:
            self.y = self.y['Intensity_label']
        # If skip_stimulus_file=True, self.y is already the Intensity_label series

        print(f"Original features shape: {self.X.shape}")
        print(f"Enriched features shape: {self.X_enriched.shape}")

    def find_optimal_features(self, n_folds: int = 6, max_features: int = 54, step_size: int = 1):
        """
        Find the optimal number of features using cross-validation.

        Args:
            n_folds (int): Number of folds for cross-validation
            max_features (int): Maximum number of top features to evaluate
            step_size (int): Step size for feature evaluation (e.g., 1 = test every number, 5 = test every 5th)

        Returns:
            dict: Results with optimal number of features and performance metrics
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.rf_hyperparams.get('random_state', 42))

        # Get feature importances
        rf_model = RandomForestClassifier(**self.rf_hyperparams)
        rf_model.fit(self.X_enriched, self.y)
        importances = rf_model.feature_importances_
        feature_indices = np.argsort(importances)[::-1]

        results = {'n_features': [], 'accuracy': [], 'f1': [], 'accuracy_std': [], 'f1_std': []}

        feature_range = range(1, max_features + 1, step_size)

        for n_top_features in tqdm(feature_range, desc="Finding Optimal Features"):
            top_features = self.X_enriched.columns[feature_indices[:n_top_features]]
            X_top = self.X_enriched[top_features]

            cv_accuracies = []
            cv_f1s = []

            for train_idx, test_idx in kf.split(X_top):
                X_train, X_test = X_top.iloc[train_idx], X_top.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                model = RandomForestClassifier(**self.rf_hyperparams)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                cv_accuracies.append(accuracy_score(y_test, y_pred))
                cv_f1s.append(f1_score(y_test, y_pred, average='macro'))

            results['n_features'].append(n_top_features)
            results['accuracy'].append(np.mean(cv_accuracies))
            results['f1'].append(np.mean(cv_f1s))
            results['accuracy_std'].append(np.std(cv_accuracies))
            results['f1_std'].append(np.std(cv_f1s))

            print(f"Features: {n_top_features:2d}, "
                  f"Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}, "
                  f"F1: {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

        # Find optimal number of features
        best_f1_idx = np.argmax(results['f1'])
        optimal_features = results['n_features'][best_f1_idx]

        print(f"\nOptimal number of features: {optimal_features}")
        print(f"Best F1 Score: {results['f1'][best_f1_idx]:.4f} ± {results['f1_std'][best_f1_idx]:.4f}")
        print(f"Best Accuracy: {results['accuracy'][best_f1_idx]:.4f} ± {results['accuracy_std'][best_f1_idx]:.4f}")

        return {
            'optimal_n_features': optimal_features,
            'results': results,
            'best_f1': results['f1'][best_f1_idx],
            'best_accuracy': results['accuracy'][best_f1_idx]
        }

    def train_final_model(self, use_enriched: bool = True, n_top_features: int = None, hyperparams: dict = None):
        """
        Train a Random Forest Classifier on all or a subset of features with specified or default hyperparameters.

        Args:
            use_enriched (bool): Whether to use enriched features (if available) or original features
            n_top_features (int, optional): Number of top features to use; if None, use all features
            hyperparams (dict, optional): Hyperparameters for RandomForestClassifier; if None, uses self.rf_hyperparams

        Returns:
            RandomForestClassifier: Trained model stored in self.final_model
        """
        hyperparams = hyperparams or self.rf_hyperparams
        print(f"Training model with hyperparameters: {hyperparams}")
        # 1. Pick the data you’ll train on (raw vs enriched), exactly as below:
        X_data_source = self.X_enriched if use_enriched and self.X_enriched is not None else self.X

        if 'source' in X_data_source.columns:
            X_data_source = X_data_source.drop(columns=['source'], errors='ignore')
        if 'molecule' in X_data_source:
            X_data_source = X_data_source.drop(columns=['molecule'], errors='ignore')

        # 2. Run 5-fold CV to get accuracy and F1
        cv_results = cross_validate(
            RandomForestClassifier(**hyperparams),
            X_data_source,
            self.y,
            cv=5,
            scoring=['accuracy', 'f1_macro'],
            return_train_score=False
        )

        print(f"5-Fold CV Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
        print(
            f"5-Fold CV F1-macro : {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")

        # Determine the actual features to be used for fitting the final model
        if n_top_features and n_top_features < X_data_source.shape[1]:
            print(f"Using {n_top_features} top features selected based on a temporary model.")
            rf_temp = RandomForestClassifier(**hyperparams)
            rf_temp.fit(X_data_source, self.y)  # Fit on all available features in X_data_source
            importances = rf_temp.feature_importances_
            # Get indices of features, sorted by importance
            top_indices = np.argsort(importances)[::-1][:n_top_features]
            # X_for_fitting will have columns selected and ordered by importance
            X_for_fitting = X_data_source.iloc[:, top_indices]
        else:
            if n_top_features:
                print(
                    f"Requested {n_top_features} top features, but only {X_data_source.shape[1]} available. Using all available features.")
            else:
                print(f"Using all {X_data_source.shape[1]} features.")
            X_for_fitting = X_data_source  # Use all features in their current order

        self.final_model = RandomForestClassifier(**hyperparams)
        # The model is trained on X_for_fitting.
        # self.final_model.feature_names_in_ will be X_for_fitting.columns
        self.final_model.fit(X_for_fitting, self.y)

        # Store feature importances. The 'Feature' column MUST be in the same order as X_for_fitting.columns
        # for the predict method to correctly align features.
        self.feature_importances = pd.DataFrame({
            'Feature': X_for_fitting.columns,  # Features in the order the model was trained on
            'Importance': self.final_model.feature_importances_  # Corresponding importances
        })

        # For display or other analyses, create a version sorted by importance
        feature_importances_sorted_for_display = self.feature_importances.sort_values('Importance', ascending=False)

        print(f"\nTraining final model with {'enriched' if use_enriched else 'original'} features complete.")
        print(f"Model trained on {len(X_for_fitting.columns)} features.")
        print(f"First 10 features the model was trained on (order as in training): {list(X_for_fitting.columns[:10])}")
        print(f"Top 10 most important features (from sorted list for display):")
        print(feature_importances_sorted_for_display.head(10))

        return self.final_model

    def predict(self, new_data: pd.DataFrame):
        """
        Make predictions on new data using the trained model.

        Args:
            new_data (pd.DataFrame): New data with the same feature structure as the training subset

        Returns:
            np.ndarray: Predicted labels
        """
        if self.final_model is None:
            raise ValueError("No trained model available. Run train_final_model first.")

        if self.feature_importances is None or 'Feature' not in self.feature_importances:
            raise ValueError("Feature importances not found. Did you train the model with selected features?")

        selected_features = self.feature_importances['Feature'].values

        if not all(f in new_data.columns for f in selected_features):
            missing = [f for f in selected_features if f not in new_data.columns]
            raise ValueError(f"New data is missing required features: {missing}")

        X_input = new_data.loc[:, selected_features]
        return self.final_model.predict(X_input)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from tqdm.notebook import tqdm


class MultiSensorRegressor:
    """
    1. Splits folds on TASK1_training.csv internally.
    2. For a given fold, creates a “combined” CSV (training + task2) at that fold.
    3. Later, can train/evaluate on that combined CSV or on held-out data.
    """

    def __init__(
            self,
            stimulus_file: str,
            folds_dir: str,
            descriptors_file: str = None,
            rf_params: dict = None
    ):
        self.stimulus_file = stimulus_file
        self.folds_dir = folds_dir
        self.descriptors_file = descriptors_file
        self.rf_params = rf_params or {'n_estimators': 100, 'random_state': 42, 'max_depth': 15}
        self.model = None
        self.X_train_columns = None
        self.mordred_df = None
        self.combined_csv_path = None
        self.train_fold_file = None

        if self.descriptors_file:
            self._load_mordred_descriptors()

    def create_combined_csv_path(
            self,
            fold_number: int,
            task2_path: str,
            training_path: str,
            stimulus_def_path: str,
            descriptors_path: str,
            output_folder: str = "."
    ):
        """
        1. Splits TASK1_training.csv into folds (if not already saved).
        2. Loads fold #fold_number as test, other folds as train.
        3. Runs DataAugmentationPipeline on the train folds + TASK2 to produce
           `combined_training_and_task2_predictions.csv`.
        4. Saves that combined CSV to disk and stores its path in `self.combined_csv_path`.

        Args:
            fold_number (int): which fold to hold out (1-based).
            task2_path (str): path to TASK2_single_RATA.csv
            training_path (str): path to TASK1_training.csv
            stimulus_def_path (str): path to TASK1_Stimulus_definition.csv
            descriptors_path (str): path to Mordred_Descriptors.csv
            output_folder (str): where to save the combined CSV.
        """
        # (A) Ensure the 5 folds exist on disk
        save_folds(
            training_path=training_path,
            n_folds=5,
            output_dir=self.folds_dir
        )

        # (B) Load the requested fold as test, the rest as train
        train_df, _ = load_fold(
            fold_dir=self.folds_dir,
            fold_number=fold_number
        )

        # (C) Write the train‐only file to disk
        temp_path = f"TASK1_training_fold{fold_number}.csv"
        train_df.to_csv(temp_path, index=False)

        # (D) Run the existing DataAugmentationPipeline on that train file
        pipeline = DataAugmentationPipeline(
            stimulus_file=stimulus_def_path,
            training_file=temp_path,
            descriptors_file=descriptors_path,
            enrich_data=True,
            rf_hyperparams={'n_estimators': 200, 'max_depth': 15, 'random_state': 42},
            use_meta_parameters=True
        )

        # (E) Find optimal features & train the RF on those train folds
        #        optimal_results = pipeline.find_optimal_features(n_folds=5, max_features=55, step_size=1)
        pipeline.train_final_model(
            use_enriched=False,
            #            n_top_features=optimal_results['optimal_n_features'],
            hyperparams={'n_estimators': 200, 'max_depth': 15, 'random_state': 42}
        )

        # (F) Load and process TASK2, then predict on it
        task_2_data = pd.read_csv(task2_path)
        original_columns = pipeline.X.columns
        task_2_data = process_task2_data(task_2_data, original_columns)

        predictions = pipeline.predict(task_2_data.set_index('stimulus'))
        results_df = pd.DataFrame({
            'stimulus': task_2_data['stimulus'],
            'Intensity_label': predictions
        })

        # (G) Merge original train‐fold data with its true Intensity_label & molecule
        stim_def = pd.read_csv(stimulus_def_path)[['stimulus', 'Intensity_label', 'molecule']]

        original_training_df_with_pred_col = (
            pd.read_csv(temp_path)
            .merge(stim_def, on='stimulus', how='left')
            .assign(source='training data')
        )

        # (H) Merge TASK2 + predictions
        task_2_df_with_predictions = (
            task_2_data
            .merge(results_df, on='stimulus', how='left')
            .assign(source='task_2')
        )

        # (I) Concatenate the two DataFrames
        combined_df = pd.concat(
            [original_training_df_with_pred_col, task_2_df_with_predictions],
            ignore_index=True,
            sort=False
        )

        # (J) Reorder columns: original TRAIN columns plus Intensity_label, source, molecule
        original_file_columns = list(pd.read_csv(temp_path).columns)
        final_ordered_columns = original_file_columns[:]
        if 'Intensity_label' not in final_ordered_columns:
            final_ordered_columns.append('Intensity_label')
        if 'source' not in final_ordered_columns:
            final_ordered_columns.append('source')
        if 'molecule' not in final_ordered_columns:
            final_ordered_columns.append('molecule')

        combined_df = combined_df[final_ordered_columns]

        # (K) Save to CSV under output_folder
        output_csv_path = f"{output_folder}/combined_training_and_task2_fold{fold_number}.csv"
        combined_df.to_csv(output_csv_path, index=False)

        print(f"\nSuccessfully saved combined data to: {output_csv_path}")
        print(f"Combined DataFrame shape: {combined_df.shape}")

        # (L) Store for later use
        self.combined_csv_path = output_csv_path

    def _load_mordred_descriptors(self):
        """
        Loads the full Mordred_Descriptors.csv into self.mordred_df,
        indexed by 'molecule' (CID). Assumes a 'molecule' column exists.
        """
        md = pd.read_csv(self.descriptors_file, encoding='latin-1', low_memory=False)
        if 'molecule' not in md.columns:
            raise ValueError("Mordred CSV must contain a 'molecule' column.")
        md = md.set_index('molecule')
        self.mordred_df = md.apply(pd.to_numeric, errors='coerce')

    def _get_mordred_for(self, molecule_series: pd.Series) -> pd.DataFrame:
        """
        Given a Series of CIDs (molecule IDs), return a DataFrame of Mordred descriptors
        aligned to that index. Missing CIDs become NaN and are filled with the column mean.
        """
        if self.mordred_df is None:
            raise ValueError("Mordred descriptors not loaded. Provide descriptors_file in __init__.")
        # Select rows in the same order as molecule_series
        desc = self.mordred_df.reindex(molecule_series.values)

        # --- Print missing CIDs ---
        missing_mask = desc.isnull().all(axis=1)
        if missing_mask.any():
            missing_cids = desc.index[missing_mask].tolist()
            print(f"⚠️  {len(missing_cids)} CIDs missing Mordred descriptors: {missing_cids}")

        # Fill NaNs with column means

        desc = desc.fillna(self.mordred_df.mean())
        desc.index = molecule_series.index
        return desc

    def _get_pom_for(self, molecule_series: pd.Series) -> pd.DataFrame:
        """
        Given a Series of CIDs (molecule IDs), return a DataFrame of pom embeddings
        aligned to that index. Missing CIDs become NaN and are filled with the column mean.
        """
        if self.pom_df is None:
            try:
                self.pom_df = pd.read_csv('pom.csv')
            except:
                raise ValueError("Mordred descriptors not loaded. Provide file")
        # Select rows in the same order as molecule_series
        desc = self.pom_df.reindex(molecule_series.values)
        # Fill NaNs with column means
        desc = desc.fillna(self.pom_df.mean())
        desc.index = molecule_series.index
        return desc

    def _prepare_training_data(self):
        """
        Reads combined CSV, drops Intensity/Pleasantness/stimulus,
        then builds:
          X_df = ['molecule', 'source', 'Intensity_label']
                 + (if descriptors_file) Mordred descriptors
          Y_df = all remaining numeric sensor columns
        One-hot encodes 'source' and 'Intensity_label', and merges descriptors.
        """
        combined_df = pd.read_csv(self.combined_csv_path)

        combined_df = combined_df.drop(columns=['Intensity', 'Pleasantness', 'stimulus'], errors='ignore')

        # Base X: molecule, source, Intensity_label
        X_base = combined_df[['molecule', 'source', 'Intensity_label']].copy()

        # If descriptors are provided, attach them
        if self.mordred_df is not None:
            desc_df = self._get_mordred_for(X_base['molecule'])
            # Merge descriptor columns into X_base
            X_base = pd.concat([X_base, desc_df], axis=1)

        # Y = all remaining columns (sensor values), keep only numeric
        Y_df = combined_df.drop(columns=['molecule', 'source', 'Intensity_label'], errors='ignore')
        Y_df = Y_df.select_dtypes(include=[np.number])

        # One‐hot encode 'source' and 'Intensity_label' (leave descriptors as-is)
        X_processed = pd.get_dummies(
            X_base,
            columns=['source', 'Intensity_label'],
            drop_first=False
        )

        self.X_train_columns = X_processed.columns
        return X_processed, Y_df

    def train(self):
        """
        (Optional) Train on the entire combined CSV at once.
        """
        X_train, Y_train = self._prepare_training_data()
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train, Y_train)
        self.model = rf

    def evaluate_on_fold(self, fold_number: int) -> dict:
        """
        Train on the combined CSV (must have been created) and evaluate on a held-out fold.

        Args:
            fold_number (int): which fold to hold out for testing (1-based)
        Returns:
            dict of {sensor_column_name: pearson_correlation}
        """
        # --- (0) Ensure combined data is ready ---
        if not self.combined_csv_path:
            raise ValueError(
                "No combined_csv_path set. Run create_combined_csv_path (and optionally augment_with_gslf) first.")

        # --- (1) Load training data from combined CSV ---
        train_df = pd.read_csv(self.combined_csv_path)

        # --- (2) Load the held-out test fold and merge in true labels ---
        _, test_df = load_fold(self.folds_dir, fold_number)
        stim_def = pd.read_csv(self.stimulus_file)[['stimulus', 'molecule', 'Intensity_label']]
        test_df = test_df.merge(stim_def, on='stimulus', how='left')
        test_df['source'] = 'training data'

        # --- (3) Drop unwanted columns ---
        train_df = train_df.drop(columns=['Intensity', 'Pleasantness', 'stimulus'], errors='ignore')
        test_df = test_df.drop(columns=['Intensity', 'Pleasantness', 'stimulus'], errors='ignore')

        # --- (4) Prepare X_train and Y_train ---
        X_train_base = train_df[['molecule', 'source', 'Intensity_label']].copy()
        Y_train_df = train_df.drop(columns=['molecule', 'source', 'Intensity_label'], errors='ignore')

        Y_train_df = Y_train_df.select_dtypes(include=[np.number])

        if self.mordred_df is not None:
            print('using mordred_df')
            desc_train = self._get_mordred_for(X_train_base['molecule'])
            X_train_base = pd.concat([X_train_base, desc_train], axis=1)
        else:
            print('not using mordred_df')

        X_train_proc = pd.get_dummies(
            X_train_base,
            columns=['source', 'Intensity_label'],
            drop_first=False
        )
        self.X_train_columns = X_train_proc.columns

        # --- (5) Fit RandomForestRegressor on combined training data ---
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train_proc, Y_train_df)
        self.model = rf

        # --- (6) Prepare X_test and Y_test ---
        X_test_base = test_df[['molecule', 'source', 'Intensity_label']].copy()
        Y_test_df = test_df.drop(columns=['molecule', 'source', 'Intensity_label'], errors='ignore')

        Y_test_df = Y_test_df.select_dtypes(include=[np.number])

        if self.mordred_df is not None:
            desc_test = self._get_mordred_for(X_test_base['molecule'])
            X_test_base = pd.concat([X_test_base, desc_test], axis=1)

        X_test_proc = pd.get_dummies(
            X_test_base,
            columns=['source', 'Intensity_label'],
            drop_first=False
        )
        X_test_proc = X_test_proc.reindex(columns=self.X_train_columns, fill_value=0)

        # --- (7) Predict & compute Pearson correlation per sensor ---
        Y_pred = self.model.predict(X_test_proc)
        Y_pred_df = pd.DataFrame(
            Y_pred,
            columns=Y_test_df.columns,
            index=Y_test_df.index
        )

        correlations = {}
        for col in Y_test_df.columns:
            true_vals = Y_test_df[col].values
            pred_vals = Y_pred_df[col].values
            if np.std(true_vals) == 0 or np.std(pred_vals) == 0:
                correlations[col] = np.nan
            else:
                correlations[col], _ = pearsonr(true_vals, pred_vals)

        return correlations


    def regress_gslf(
            self,
            processed_gslf_df: pd.DataFrame,
            output_path: str = None,
            threshold=None,
            top_k=None
    ) -> pd.DataFrame:
        """
        2. Compute per-sensor thresholds so P(train > thresh) == mean_GSLF.
        3. X_train = [binary_train_after_threshold + Mordred], Y_train = continuous train values.
        4. Fit RF, then X_pred = [binary_GSLF + Mordred] → predict continuous sensor values.
        5. Use DataAugmentationPipeline to predict Intensity_label for GSLF.
        6. Append both the regressed sensor values and the Intensity_label rows
           (tagged source='GSLF') to the combined CSV.
        """
        if not self.combined_csv_path:
            raise ValueError("Must run create_combined_csv_path before regress_gslf.")

        # --- load & isolate training-only rows ---
        combined_df = pd.read_csv(self.combined_csv_path)
        train_df = combined_df

        # --- identify sensor columns ---
        sensor_cols = processed_gslf_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'molecule' in sensor_cols:
            sensor_cols.remove('molecule')
        if 'stimulus' in sensor_cols:
            sensor_cols.remove('stimulus')

        if threshold is None:
            overall_gslf_prevalence = processed_gslf_df[sensor_cols].values.mean()

            global_threshold = train_df[sensor_cols].values.flatten()
            global_threshold = np.quantile(global_threshold, 1.0 - overall_gslf_prevalence)
        else:
            global_threshold = threshold

        print(f'global_threshold: {global_threshold}')
        binary_train = pd.DataFrame({
            col: (train_df[col] > global_threshold).astype(int)
            for col in sensor_cols
        }, index=train_df.index)

        desc_train = self._get_mordred_for(train_df['molecule'])
        X_train = pd.concat([binary_train, desc_train], axis=1)
        Y_train = train_df[sensor_cols]

        # --- fit multi-output RF for continuous sensor regression ---
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train, Y_train)

        # --- prepare binary + descriptors for GSLF ---
        # ensure we have a Series of CIDs
        if 'molecule' in processed_gslf_df.columns:
            gslf_cids = processed_gslf_df['molecule']
        else:
            gslf_cids = pd.Series(
                processed_gslf_df.index,
                index=processed_gslf_df.index,
                name='molecule'
            )
        binary_gslf = processed_gslf_df[sensor_cols].astype(int)
        desc_gslf = self._get_mordred_for(gslf_cids)
        X_pred = pd.concat([binary_gslf, desc_gslf], axis=1)

        # --- predict continuous sensor values ---
        Y_pred = rf.predict(X_pred)
        Y_pred_df = pd.DataFrame(Y_pred, columns=sensor_cols, index=processed_gslf_df.index)

        # --- keep only the top-32 sensor responses per stimulus -----------------
        if top_k:
            TOP_K = top_k
            ranks = Y_pred_df.rank(axis=1, method="first", ascending=False)  # per-row ranks
            Y_pred_df[ranks > TOP_K] = 0  # zero everything below rank 32
            print(f"Applied top-{TOP_K} masking: values ranked >{TOP_K} per row set to 0")

        # --- assemble GSLF regression block ---
        gslf_rows = pd.DataFrame({'molecule': gslf_cids})
        for col in sensor_cols:
            gslf_rows[col] = Y_pred_df[col]

        # --- now your final pipeline for Intensity_label ---
        # save training-only data to temp for pipeline
        temp_train = "temp_training_only.csv"
        train_df.drop(columns=['source', 'molecule'], errors='ignore') \
            .to_csv(temp_train, index=False)

        pipeline = DataAugmentationPipeline(
            stimulus_file=self.stimulus_file,
            training_file=temp_train,
            descriptors_file=self.descriptors_file,
            enrich_data=True,
            rf_hyperparams=self.rf_params,
            use_meta_parameters=False
        )
        pipeline.train_final_model(
            use_enriched=False,
            hyperparams=self.rf_params
        )
        preds = pipeline.predict(gslf_rows)

        # attach Intensity_label and source
        gslf_rows['Intensity_label'] = preds
        gslf_rows['source'] = 'GSLF'

        if isinstance(preds, pd.Series) or isinstance(preds, np.ndarray):
            counts = pd.Series(preds).value_counts()
        else:
            counts = preds.value_counts()

        h = counts.get('H', 0)
        l = counts.get('L', 0)
        ratio = h / l if l > 0 else np.inf
        print(f"Intensity_label distribution in GSLF: H={h}, L={l}, H/L ratio = {ratio:.2f}")

        # --- align to combined schema & append ---
        for col in combined_df.columns:
            if col not in gslf_rows.columns:
                gslf_rows[col] = np.nan
        gslf_rows = gslf_rows[combined_df.columns]

        merged = pd.concat([combined_df, gslf_rows], ignore_index=True, sort=False)
        if output_path:
            merged.to_csv(output_path, index=False)
            self.combined_csv_path = output_path

        print('merged')
        print(merged)

        return merged


    def create_combined_csv_path_all_folds(
            self,
            task2_path: str,
            training_path: str,
            stimulus_def_path: str,
            descriptors_path: str,
            output_folder: str = ".",
    ):
        """
        Build ONE “super-training” CSV:
            • all TASK1 rows      (source='training data')
            • all Task2 rows + predicted Intensity_label (source='task_2')
        and store its path in self.combined_csv_path
        """
        import pandas as pd, numpy as np
        from pathlib import Path

        # ── A. TASK-1 rows ────────────────────────────────────────────────
        train_df = pd.read_csv(training_path).copy()
        train_df["source"] = "training data"

        stim_def = pd.read_csv(stimulus_def_path)[["stimulus", "Intensity_label", "molecule"]]
        train_df = train_df.merge(stim_def, on="stimulus", how="left")

        # ── B.  train a pipeline ONLY on a copy **without** the molecule col ──
        temp_train = "TASK1_training_all_folds_TEMP.csv"
        train_df_nodup = train_df.drop(columns=['molecule', 'source', 'Intensity_label'])

        # 2) DEBUG – show any non-numeric columns that remain
        import numpy as np
        print("⛔ non-numeric in temp_train →",
              train_df_nodup.select_dtypes(exclude=[np.number]).columns.tolist())

        # 3) write the temp CSV
        train_df_nodup.to_csv(temp_train, index=False)

        pipe = DataAugmentationPipeline(
            stimulus_file=stimulus_def_path,
            training_file=temp_train,
            descriptors_file=descriptors_path,
            enrich_data=True,
            rf_hyperparams=self.rf_params,
            use_meta_parameters=True,
        )
        pipe.train_final_model(use_enriched=False,
                               hyperparams=self.rf_params)

        # ── C.  predict Intensity_label for Task-2 rows ────────────────────
        task2_raw = pd.read_csv(task2_path)
        task2_proc = process_task2_data(task2_raw, original_columns=pipe.X.columns)

        task2_preds = pipe.predict(task2_proc.set_index("stimulus"))
        task2_proc["Intensity_label"] = task2_preds
        task2_proc["source"] = "task_2"

        if "molecule" not in task2_proc.columns:
            task2_proc = task2_proc.merge(
                stim_def[["stimulus", "molecule"]], on="stimulus", how="left"
            )

        # ── D.  concatenate & reorder ──────────────────────────────────────
        combined = pd.concat([train_df, task2_proc], ignore_index=True, sort=False)

        first_cols = list(pd.read_csv(training_path).columns)  # original order
        for extra in ["Intensity_label", "source", "molecule"]:
            if extra not in first_cols:
                first_cols.append(extra)

        combined = combined[first_cols]

        # ── E.  save & remember ────────────────────────────────────────────
        out_path = Path(output_folder) / "combined_training_and_task2_all.csv"
        combined.to_csv(out_path, index=False)
        print(f"[combined] saved → {out_path}   shape={combined.shape}")

        self.combined_csv_path = str(out_path)
        return self.combined_csv_path

    # ------------------------------------------------------------------------
    def predict_new_data(
            self,
            new_data_path: str,
            stimulus_def_path: str = None,
            return_df: bool = False,
            output_csv: str | None = None,
    ):
        """
        Fit on *all* rows in self.combined_csv_path and predict the sensor
        values for an unseen CSV (no true sensor values available).

        Parameters
        ----------
        new_data_path : str
            CSV with at least a 'stimulus' column.
        stimulus_def_path : str, optional
            Needed to merge 'molecule' & 'Intensity_label' for the new stimuli.
            If None, falls back to self.stimulus_file.
        return_df : bool
            If True, return a DataFrame (stimulus + sensor columns).
        output_csv : str or Path, optional
            If provided, save the predictions to this file.
        """
        if not self.combined_csv_path:
            raise ValueError(
                "Run create_combined_csv_path_all_folds (and optionally augment_with_gslf) first."
            )

        stimulus_def_path = stimulus_def_path or self.stimulus_file

        # ------------- 1. fit on the combined CSV -------------
        X_train, Y_train = self._prepare_training_data()  # uses self.combined_csv_path
        rf = RandomForestRegressor(**self.rf_params)
        rf.fit(X_train.fillna(X_train.mean()), Y_train)  # fill NaNs defensively
        self.model = rf

        # remember sensor column order for convenience
        sensor_cols = Y_train.columns.tolist()

        # ------------- 2. build feature matrix for new data -------------
        new_df = pd.read_csv(new_data_path)
        stim_def = pd.read_csv(stimulus_def_path)[["stimulus", "molecule", "Intensity_label"]]
        new_df = new_df.merge(stim_def, on="stimulus", how="left")
        new_df["source"] = "training data"

        # base features
        X_base = new_df[["molecule", "source", "Intensity_label"]].copy()

        # descriptors
        if self.mordred_df is not None:
            print('using mordred_df')
            desc = self._get_mordred_for(X_base["molecule"])
            X_base = pd.concat([X_base, desc], axis=1)

        X_proc = pd.get_dummies(X_base, columns=["source", "Intensity_label"], drop_first=False)
        X_proc = X_proc.reindex(columns=self.X_train_columns, fill_value=0)

        # ------------- 3. predict -------------
        Y_pred = self.model.predict(X_proc)
        pred_df = pd.DataFrame(Y_pred, columns=sensor_cols)
        pred_df.insert(0, "stimulus", new_df["stimulus"])

        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(output_csv, index=False)
            print(f"[predict] saved predictions → {output_csv}")

        return pred_df if return_df else None

