from Training import WrangleRepository, MakePipeline, LearningCurve
from pathlib import Path

# defining sub class
sub_class = {
        20: "1-STORY 1946 & NEWER ALL STYLES",
        30: "1-STORY 1945 & OLDER",
        40:	"1-STORY W/FINISHED ATTIC ALL AGES",
        45:	"1-1/2 STORY - UNFINISHED ALL AGES",
        50:	"1-1/2 STORY FINISHED ALL AGES",
        60:	"2-STORY 1946 & NEWER",
        70:	"2-STORY 1945 & OLDER",
        75:	"2-1/2 STORY ALL AGES",
        80:	"SPLIT OR MULTI-LEVEL",
        85:	"SPLIT FOYER",
        90:	"DUPLEX - ALL STYLES AND AGES",
       120:	"1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
       150:	"1-1/2 STORY PUD - ALL AGES",
       160:	"2-STORY PUD - 1946 & NEWER",
       180:	"PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
       190:	"2 FAMILY CONVERSION - ALL STYLES AND AGES"

}


class GetData:
    """Help us organize our data nicely
    Parameters:
        root_path: str
            -> root path where our data is located eg. Path.cwd()
        file_name: str
            -> a path or the data `csv` file
    """
    def __init__(self, sub_class=sub_class, X_train=None):
        # Getting the repo
        self.X_train = X_train
        self.repo = WrangleRepository(sub_class = sub_class)
        self.pipe = MakePipeline(self.X_train)
        
    def training_data(self):
        try:
            # get the raw data 
            self.repo.wrangle()
            df_raw = self.repo.get_data("wrangled")
        except Exception as e:
            print(f"Error getting the data: {e}")

            
        # get the data (basic cleaning)
        self.repo.basic_cleaning()
        df = self.repo.get_data("basic")

        # get features that are selected
        self.repo.feature_selection()
        df = self.repo.get_data("selected")
        
        # get the engineered data 
        self.repo.feature_engineering()
        df = self.repo.get_data("engineered")

        # remove outliers in our data
        self.repo.remove_outliers()
        df = self.repo.get_data() 
        
        return df, df_raw
        
    def get_sale_price(self):
        """From the raw data we get the sale price data"""
        self.repo.basic_cleaning()
        df = self.repo.get_data("basic")
        sale_price = df["SalePrice"]

        # return 
        self.sale_price = sale_price
        return sale_price


    def _make_pca(self):
        """Get feature decomposed pipeline"""
        pipe = self.pipe 
        pca_pipeline = pipe.make_pca_pipeline()

        self.pca_pipeline = pca_pipeline
        return pca_pipeline

    def get_pca_data(self):
        """Get the decompoese features data"""
        pca_pipeline = self._make_pca()
        pca_data = pca_pipeline.fit_transform(self.X_train)

        self.pca_data = pca_data
        return pca_data

# Buidling the class
class GetModel:
    """Build models"""
    def __init__(self, X_train):
        "Nothing here"
        self.X_train = X_train
        
    def build_linear_model(self):
        """Building a linear regression model"""
        # Getting the pipeline
        pipe = MakePipeline(X_train=self.X_train)
        
        # Getting the model pipeline 
        linear_pipeline = pipe.make_linear_pipeline() 

        self.linear_pipeline = linear_pipeline

        return linear_pipeline
        
    def build_tree_model(self):
        """Building the decision tree regressor model"""
        # Get the pipeline 
        pipe = MakePipeline(X_train=self.X_train)
        
        # Get the model pipeline
        tree_pipeline = pipe.make_decision_tree_pipeline() 

        # return the tree pipeline
        self.tree_pipeline = tree_pipeline

        return tree_pipeline
    def build_forest_model(self):
        """Build the forest model"""
        # Get the pipeline 
        pipe = MakePipeline(X_train=self.X_train)
        
        # Get the model pipeline
        forest_pipeline = pipe.make_random_forest_pipeline() 

        # return the tree pipeline
        self.forest_pipeline = forest_pipeline

        return forest_pipeline

    def build_gradient_model(self):
        """Build the forest model"""
        # Get the pipeline 
        pipe = MakePipeline(X_train=self.X_train)
        
        # Get the model pipeline
        gradient_pipeline = pipe.make_gradient_boosting_pipeline() 

        # return the tree pipeline
        self.gradient_pipeline = gradient_pipeline

        return gradient_pipeline


class LearningCurvePlotter:
    """Building the learning curve plots"""
    def __init__(self, estimator, X, y):
        self.lc = LearningCurve(estimator=estimator, X=X, y=y)
    def plot_lc(self):
        # build lc
        self.lc.learning_curve()

        # plotting learning curve
        fig = self.lc.plot_lc()

        return fig

class IDMapping:
    def __init__(self, sub_class=sub_class):
        """Initialization for Mapping"""
        # Getting the repo
        self.repo = WrangleRepository(file_name = "test.csv", sub_class = sub_class)
    def get_test_data(self):
        # Getting the csv data
        self.repo.wrangle()
        df = self.repo.get_data("wrangled")
        
        # basic cleaning (no)
        self.repo.basic_cleaning(clean=False)
        df = self.repo.get_data("basic")

        # feature selction (n0)
        self.repo.feature_selection(variance_selector=False)
        df = self.repo.get_data("selected")
        
        # feature engineering 
        self.repo.feature_engineering()
        df = self.repo.get_data("engineered")

        # final mapping 
        df, df_raw = GetData(). training_data()# Get the trainingdata
        X_train = df.drop(columns = "SalePrice") # Make the training feature matrix
        df_test = df[X_train.columns]

        return df_test








