# Important libraries
import pandas as pd 
import math 
import numpy as np 
import logging 
from pathlib import Path
import plotly.express as px

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import learning_curve 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline  
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Wrangle class object
class WrangleRepository:
    """Control our loading and data cleaning
    - Load the csvfile into a dataframe

    Parameters:
        root_path: str/path object
            -> path where you are currently on
        file_name: str 
            -> Name of the csv file, remember to include `.csv` extension
    """
    # instance of our class 
    def __init__(
        self,
        sub_class = None,
        root_path = Path.cwd(),
        file_name = "train.csv"
    ):
        logging.info("Inintialized our class instances!")
        self.sub_class = sub_class
        self.filepath = root_path / file_name

    # Get the DataFrame 
    def wrangle(self):
        """Load the csv file into a DataFrame
        """
        logging.info("Loading the csv file into a dataframe")
        # loading the csv file 
        df = pd.read_csv(self.filepath).set_index("Id")
        
        self.df_wrangled = df
    
        return df

    # Basic cleaning - function
    def basic_cleaning(self, missing_values_pct=None, clean=True):
        """Basic clearning of our dataset
        - Drop high missing values in our features
    
        Parameters:
            missing_values_pct: pd.Series
                -> Series data of high missing values columns
            clean: bool (True/False)
                
        """
        # getting the DataFrame(df)
        df = self.wrangle()
        
        if clean:
            # compute missing numerical values 
            if missing_values_pct is None:
                missing_values = (df.isnull().sum()[df.isnull().sum() > 1]).sort_values()
                # percentage counts 
                missing_values_pct = pd.Series(((100 * missing_values.values / len(df))),
                                               index=missing_values.index, name="missing_pct")
            # drop high missing values features 
            mask = missing_values_pct > 50
            missing_cols = missing_values_pct[mask].index.to_list()
            logging.info(f"Dropped  high missing values features: \n {missing_cols}")
            
            # drop columns 
            df.drop(columns=missing_cols, inplace=True)

        self.df_basic = df
    
        # return 
        return df

    # Feature selection 
    def feature_selection(self, variance_selector=True, threshold_num=0.05, threshold_cat=0.95):
        """Selecting most important and useful features
        - High variance check
    
        Parameters:
            variance_selector: bool 
                -> If we want to compute variance selection
            threshold_num: float 
                -> Decimal threshold, eg. 0.05 means 5% 
            threshold_cat: float
                -> Decimal threshold for categorical variables, by default is 95%
        """
        # Getting the df, basic cleaning 
        df = self.df_basic
        
        if variance_selector:
            logging.info("Computing low variance feature selection")
            # get numerical features 
            num_feat = df.select_dtypes(include="number")
            
            # instantiating
            val_selector = VarianceThreshold(threshold=threshold_num)
            # fitting and transforming
            val_selector.fit_transform(num_feat)
            high_val_feat = num_feat.columns[val_selector.get_support()]
            
            # making the dataframe 
            df = pd.concat([df[high_val_feat], df.select_dtypes(include="object")], axis=1)
    
            # high variance for categorical variables 
            high_val_cat = [col for col in df.select_dtypes(include="object")
                            if df[col].value_counts(normalize=True).iloc[0] > threshold_cat
            ]
            df.drop(columns=high_val_cat, inplace=True)

        self.df_selected = df
        # return 
        return df

    # Feature Engineering - function
    def feature_engineering(self, engineer=True):
        """Engineering our features 
    
        Parameters:
            sub_class: dict 
                -> A mapping dictionary with a int: descriptions eg.. 20: "1-STORY 1946 & NEWER  ALL STYLES"
            enginering: bool
                -> to do feature enginerring or not, either True/False
        """
        # Getting the df from selected features
        df = self.df_selected
        # subclass modification
        df["MSSubClass"] = df["MSSubClass"].replace(self.sub_class)
        if engineer:
            # Remodified date 
            df["RemodAfter"] = df["YearRemodAdd"] - df["YearBuilt"]
            # Remodified buildings 
            df["Remod"] = df["RemodAfter"] > 0
            # Total sq feets 
            df["BsmtFinished"] = df["TotalBsmtSF"] - df["BsmtUnfSF"]
            # full bathrooms 
            df["FullBathrooms"] = df["BsmtFullBath"] + df["FullBath"]
            # half bathrooms 
            df["HalfBathrooms"] = df["BsmtHalfBath"] + df["HalfBath"]
            # getting the age of the building
            df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]) + (df["MoSold"]/12)

        self.df_engineered = df
        # return 
        return df
    def remove_outliers(self, upper_quantile=0.9, lower_quantile=0.1):
        """Remove outliers from out numerical columns
        Parameters:
            upper_quantile: int
                -> upper quantile to check
            lower_quantile: int
                -> lower floor to get our data
        """
        # Get the df 
        df = self.df_engineered
        # Remove outliers
        for col in df.select_dtypes(include="number").columns:
            q_l, q_u = df[col].quantile([lower_quantile, upper_quantile])
            mask = df[df[col].between(q_l, q_u)]

        self.df_outlier = mask
        return mask

    # finally getting the data 
    def get_data(self, stage="outlier"):
        """Get the final data at all stages:
        - wrangled: imported data
        - basic: basic cleaned data
        - selected: feature selected data
        - engineered: feature engineered data
        - outlier removed features
        """
        return getattr(self, f"df_{stage}", None)
    def __repr__(self):
        return f"WrangleRepository filepath={self.filepath}"


class MakePipeline:
    """This class will make all the necessary pipelines 
    - column transformer
    """

    def __init__(self, X_train):
        self.X_train = X_train

    def make_column_pipeline(self):
        """Make the column transformer pipeline"""
        # Numerical pipeline
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Column transformer
        col_pipeline = ColumnTransformer([
            ("NumericalFeatures", num_pipeline, self.X_train.select_dtypes(include="number").columns),
            ("CategoricalFeatures", cat_pipeline, self.X_train.select_dtypes(include="object").columns)
        ])

        self._columns_pipeline = col_pipeline
        return col_pipeline
    def make_pca_pipeline(self):
        """Pca pipeline"""
        col_pipeline = self.make_column_pipeline()
        pca_pipeline = Pipeline(
            [
                ("preprocess", col_pipeline),
                ("PCA Algorithm", PCA(n_components=1, random_state=42))
            ]
        )

        # return
        self._pca_pipeline = pca_pipeline
        return pca_pipeline
        
    def make_linear_pipeline(self):
        """Making the linear regression model pipeline"""
        col_pipeline = self.make_column_pipeline()
        # linear regression
        linear_pipeline = Pipeline(
            [
                ("preprocess", col_pipeline),
                ("linear_model", LinearRegression())
            ]
        ) 

        # return 
        self._linear_pipeline = linear_pipeline
        
        return linear_pipeline

    def make_decision_tree_pipeline(self):
        """Make the decision tree model"""
        col_pipeline = self.make_column_pipeline()

        # decision tree pipeline 
        tree_pipeline = Pipeline(
                    [
                        ("preprocess", col_pipeline),
                        ("tree_model", DecisionTreeRegressor(
                            random_state=42,
                            min_samples_leaf=1,
                            min_samples_split=6,
                            max_depth=7
                        ))
                    ]
                )
        # return the model pipeline 
        self._tree_pipeline = tree_pipeline

        return tree_pipeline
    def make_random_forest_pipeline(self):
        """Make the random forest pipeline"""
        col_pipeline = self.make_column_pipeline()

        # decision tree pipeline 
        forest_pipeline = Pipeline(
                    [
                        ("preprocess", col_pipeline),
                        ("forest_model", RandomForestRegressor(
                            random_state=42,
                            n_estimators = 60,
                            min_samples_leaf=1,
                            min_samples_split=4,
                            max_depth= 10
                        ))
                    ]
                )
        # return the model pipeline 
        self._forest_pipeline = forest_pipeline

        return forest_pipeline
    def make_gradient_boosting_pipeline(self):
        """Make the random forest pipeline"""
        col_pipeline = self.make_column_pipeline()

        # decision tree pipeline 
        gradient_pipeline = Pipeline(
                    [
                        ("preprocess", col_pipeline),
                        ("forest_model", GradientBoostingRegressor(
                            random_state=42,
                            n_estimators = 100,
                            min_samples_leaf=1,
                            min_samples_split=3,
                            max_depth= 2
                        ))
                    ]
                )
        # return the model pipeline 
        self._gradient_pipeline = gradient_pipeline

        return gradient_pipeline


    def get_pipeline(self, stage):
        """Get the pipeline in the following stages:
        - columns
        - pca
        - linear
        - tree
        """
        return getattr(self, f"_{stage}_pipeline", None)

    def __repr__(self):
        return f"Pipeline Stage presenter:"


# Learning curve plotting
class LearningCurve:
    """Train and build learning curve plot
    """
    # class instantiation 
    def __init__(self,estimator, X, y):
        """
        Parameters:
            estimator:
                -> A trained model 
            train: pd.DataFrame 
                -> A training feature matrix(x,y)
            val: pd.DataFrame
                -> A validation feature matrix
        """
        self.estimator = estimator 
        self.X = X 
        self.y = y
    # learning curve building
    def learning_curve(self):
        """Building the learning curve and returning results"""
        train_size, train_score, val_score = learning_curve(
            estimator=self.estimator,
            X = self.X,
            y = self.y,
            random_state=42,
            verbose=1,
            n_jobs=-1,
            scoring='r2',
            shuffle=True
        )
        
        # converting negatibe maes 
        train_score = np.mean(train_score, axis=1)
        val_score = np.mean(val_score, axis=1)

        # return 
        self._lc = [train_size, train_score, val_score]
        return train_size, train_score, val_score

    def make_dataframe(self): 
        """Making the dataframe from learning curve results"""
        # Get the data 
        train_size, train_score, val_score = self.learning_curve()
        # Making the dataframe 
        lc = pd.DataFrame(
            {
                "Train Size": train_size,
                "Train R2": train_score,
                "Validation R2": val_score
            }
        )
        self._df_lc = lc
        
        return lc
    def melt_dataframe(self):
        """Melting the dataframe for easy plotting using plotly express"""
        # Get the data 
        lc = self.make_dataframe()
        # melt our datframe 
        lc_melt = lc.melt(id_vars = "Train Size",
                          value_vars=["Train R2", "Validation R2"],
                          value_name="R2",
                          var_name="Set")

        self._melt_lc = lc_melt

        return lc_melt
    def plot_lc(self): 
        """Plotting the learning curves(train and val) under one plot"""
        # Get the data 
        lc_melt = self.melt_dataframe()
        fig = px.line(
            data_frame=lc_melt,
            x = "Train Size",
            y = "R2", 
            color="Set",
            markers=True,
            title="Training and Validation learning curves"
        )
        fig.update_layout(xaxis_title="No. of variables",
                          yaxis_title="R2 Score",
                          legend_title="Sample Data",
                          template="plotly_white")
        self._fig = fig

        return fig
    def get_data(self, item="fig"):
        """Get the data eg dataframe, figure or a list you need at each point
        Attributes:
        1. lc: LearningCurves
        2. df_lc: learning curve dataframe 
        3. melt_lc: melt dataframe 
        4. fig: figure itself
        """
        return getattr(self, f"_{item}", None)

    def __repr__(self):
        return f"LearningCurve: {type(self.estimator)} "

# Test Predictions
class TestPredicter:
    """Get the csv test file and do the predictions
    """
    def __init__(self, test_data, model, filepath=Path.cwd()):
        """Get the data and the model
        Parameters:
            test_data: pd.DataFrame
                -> test wrangled data, test_data.columns == X_train.colums
            model: model 
                -> trained model 
            filepath: str/Path object 
                -> Path where to save our mapped id
        """
        self.test_data=test_data
        self.model = model
        self.filepath = filepath
    # making predictions 
    def predict(self):
        """Gets the data and makes a  prediction"""
        pred = self.model.predict(self.test_data)
        self._df_prediction = pred
        return pred
    # prediction function
    def id_mapper(self, label):
        # get the predictions 
        pred = self.predict()
        # mapping to an id
        sub = pd.DataFrame(
            {
                "Id": self.test_data.index,
                "SalePrice": pred
            }
        ).set_index("Id")
    
        # Saving the form into a csv file 
        file_path = self.filepath/f"{label}_submission.csv"
        sub.to_csv(file_path)

        self._df_mapped = sub
        
        return sub
    # getting the data 
    def get_data(self, section):
        """Get the saved data in each section
        Sections:
        1.prediction
        2. mapped
        """
        return getattr(self, f"_df_{section}", None)
    # message 
    def __repr__(self):
        return f"TestMapper on {self.filepath}"  

