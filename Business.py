from Service import GetData, GetModel, LearningCurve, IDMapping
from Training import TestPredicter

import plotly.express as px 
import pandas as pd
from sklearn.pipeline import Pipeline

class ModelBuilder:
    def __init__(self):
        """Init sections"""
        
    def linear_model(self):
        # get the data 
        df, df_raw = GetData().training_data()
        # splitting the data 
        target = "SalePrice"
        X_train = df.drop(columns = target)
        y_train = df[target]

        
        # get the pipeline
        pipe = GetModel(X_train=X_train).build_linear_model()


        # Training the model 
        model = pipe.fit(X_train,y_train)

        self.model = model 
        self.X_train = X_train
        self.y_train = y_train

        return model, X_train, y_train
        
    def tree_model(self):
        # get the data 
        df, df_raw = GetData().training_data()
        # splitting the data 
        target = "SalePrice"
        X_train = df.drop(columns = target)
        y_train = df[target]

        
        # get the pipeline
        pipe = GetModel(X_train=X_train).build_tree_model()


        # Training the model 
        model = pipe.fit(X_train,y_train)

        self.model = model 
        self.X_train = X_train
        self.y_train = y_train

        return model, X_train, y_train
    def forest_model(self):
        # get the data 
        df, df_raw = GetData().training_data()
        # splitting the data 
        target = "SalePrice"
        X_train = df.drop(columns = target)
        y_train = df[target]

        
        # get the pipeline
        pipe = GetModel(X_train=X_train).build_forest_model()


        # Training the model 
        model = pipe.fit(X_train,y_train)

        self.model = model 
        self.X_train = X_train
        self.y_train = y_train

        return model, X_train, y_train
    def gradient_model(self):
        # get the data 
        df, df_raw = GetData().training_data()
        # splitting the data 
        target = "SalePrice"
        X_train = df.drop(columns = target)
        y_train = df[target]

        
        # get the pipeline
        pipe = GetModel(X_train=X_train).build_gradient_model()


        # Training the model 
        model = pipe.fit(X_train,y_train)

        self.model = model 
        self.X_train = X_train
        self.y_train = y_train

        return model, X_train, y_train

class MapId:
    def __init__(self):
        """Class Initialization"""
    def get_id(self, label):
        # linear model
        model, X_train, y_train = ModelBuilder().linear_model()
    
        # Get the test set
        test_data = IDMapping().get_test_data()
    
        # predictions
        sub = TestPredicter(test_data = test_data, model=model).id_mapper(label=label)
        return sub

# Graph builder 
class GraphBuilder:
    """This module has functions that will help in building the graphs
    -> Building the histogram(saleprice)
    """
    def house_price_hist(self):
        """Plot a histogram for house prices"""
        # get the data 
        house_prices = GetData().get_sale_price() 

        # plotting the histogram 
        fig = px.histogram(
            house_prices, 
            title="Distribution: Amos House Sale Price", 
            nbins=50
        )
        fig.update_layout(
            xaxis_title="Sale Price",
            yaxis_title="Frequency [counts]",
            legend_title="House Sale Price"
        )

        # return figure 
        self.fig = fig 
        return fig 
    def pca_plot(self):
        """Build a pca plot figure"""
        # Get the raw dataset with all the features 
        df, raw_data = GetData().training_data()
        # get the pca data 
        pca_data = GetData(X_train=raw_data).get_pca_data()
    
        fig = px.scatter(
            x=pca_data.ravel(),
            y=raw_data["SalePrice"],
            title="Scatter Plot: Decomposed Features vs. Sale Price"
        )
    
        fig.update_layout(
            xaxis_title="Decomposed Feature(s)",
            yaxis_title="Sale Price ($)",
            legend_title="Plot Type", 
        )

        
        # return
        self.fig = fig
        return fig

    # plotting leaning curve
    def learning_curve_linear(self):
        # Get the model(linear model) 
        model, X, y = ModelBuilder().linear_model()

        # Build Lc
        lc = LearningCurve(estimator=model, X=X, y=y)

        # Plotting 
        fig = lc.plot_lc()

        return fig
    
    def learning_curve_tree(self):
        # Get the model(linear model) 
        model, X, y = ModelBuilder().tree_model()

        # Build Lc
        lc = LearningCurve(estimator=model, X=X, y=y)

        # Plotting 
        fig = lc.plot_lc()

        return fig
    def learning_curve_forest(self):
        
        model, X, y = ModelBuilder().forest_model()
        # Build Lc
        lc = LearningCurve(estimator=model, X=X, y=y)

        # Plotting 
        fig = lc.plot_lc()

        return fig
    def learning_curve_gradient(self):
        
        model, X, y = ModelBuilder().gradient_model()
        # Build Lc
        lc = LearningCurve(estimator=model, X=X, y=y)

        # Plotting 
        fig = lc.plot_lc()

        return fig

    # plotting scatter plot 
    def scatter_plot(self, plot_type):
        """Make a scatter plot comparing actual vs. predicted values"""
        
        if plot_type == "linear":
            model, X, y = ModelBuilder().linear_model()
            x = GetData(X_train=X).get_pca_data()
            y = y
            y_pred = model.predict(X)
            label = "Linear Regression"
        
        elif plot_type == "tree":
            model, X, y = ModelBuilder().tree_model()
            x = GetData(X_train=X).get_pca_data()
            y = y
            y_pred = model.predict(X)
            label = "Decision Tree Regression"
            
        elif plot_type == "forest":
            model, X, y = ModelBuilder().forest_model()
            x = GetData(X_train=X).get_pca_data()
            y = y
            y_pred = model.predict(X)
            label = "Random Forest Regression"

        else:
            model, X, y = ModelBuilder().gradient_model()
            x = GetData(X_train=X).get_pca_data()
            y = y
            y_pred = model.predict(X)
            label = "Gradient Boosting Regression"
        
        # making the dataframe
        df = pd.DataFrame({
            "x": x.ravel(),
            "y": y,
            "y_pred": y_pred
        })

        # melting the dataframe for easier plotting 
        df_melt = pd.melt(
            frame=df,
            id_vars="x",
            value_vars=["y", "y_pred"],
            var_name="Set",
            value_name="Sale Price"
        )

        # Making the scatter plot
        fig = px.scatter(
            data_frame=df_melt,
            x="x",
            y="Sale Price",
            color="Set",
            title=f"{label} Scatter Plot: Decomposed Features vs. Sale Price"
        )
    
        fig.update_layout(
            xaxis_title="Decomposed Feature(s)",
            yaxis_title="Sale Price ($)",
            legend_title="Plot Type", 
            template = "plotly_white"
        )
    
        # return 
        return fig

    def residual_plot(self):
        # Get the model 
        model, X, y = ModelBuilder().linear_model()

        residuals = y - model.predict(X)
        # residual distributions
        fig = px.histogram(
            residuals,
            nbins=50,
            labels = "Residuals",
            title = "Predicted Sale Price: Linear Regression Model Residuals Distribution"
        )
        fig.update_layout(
            xaxis_title = "Residuals",
            yaxis_title = "Frequency (counts)",
            legend_title = "Resid Parameter",
            template = "plotly_white"
        )

        return fig
    def residual_tree_plot(self, model_type):
        """Displaying the educative text and also getting the figure"""
        text = f"Tree models, are more conserned with purity.\n A random scatter of the residuals may still shows that the tree is fitting well. \nPattern might hint underfitting or missing feature. Keep this in mind!👌"
        if model_type == "tree":
            # plotting the residual plot for random forest 
            model, X, y = ModelBuilder().tree_model() 
            label = "Decision Tree Regressor"
        elif model_type == "forest":
            # Getting random forest model
            model, X, y = ModelBuilder().forest_model() 
            label = "Random Forest Regressor"
        else:
            # Getting random forest model
            model, X, y = ModelBuilder().gradient_model() 
            label = "Gradient Boosting Regressor"
            
        residuals = y - model.predict(X)
        # residual distributions
        fig = px.histogram(
            residuals,
            nbins=50,
            labels = "Residuals",
            title = f"Predicted Sale Price: {label} Model Residuals Distribution"
        )
        fig.update_layout(
            xaxis_title = "Residuals",
            yaxis_title = "Frequency (counts)",
            legend_title = "Resid Parameter",
            template = "plotly_white"
        )
        
        return text, fig
        
        

    def feature_importance(self, model_type):
        if model_type == "linear":
            # geting the model 
            model, X, y = ModelBuilder().linear_model()
    
            # Getting feature importances
            coeficients = model.named_steps["linear_model"].coef_
            features  = model.named_steps["preprocess"].get_feature_names_out().ravel()
            features = [f.split("__")[1] for f in features]
            
            # making feature importances
            feat_imp = pd.Series(coeficients, index=features, name="feature importance").sort_values(key=abs)
                    
            # feature imporances plot 
            fig = px.bar(
                feat_imp.tail(10), 
                orientation = "h",
                title = "Top 10: Linear Model Feature importance plot"
            )
            fig.update_layout(
                xaxis_title = "Importances",
                yaxis_title = "Features",
                legend_title = "Item",
                template = "plotly_white"
            )
                    
            return fig   
        elif model_type == "tree":
            # Get the model
            model, X, y = ModelBuilder().tree_model()
            
            # Get feature importances
            coeficients = model.named_steps["tree_model"].feature_importances_
            features  = model.named_steps["preprocess"].get_feature_names_out().ravel()
            features = [f.split("__")[1] for f in features]
            
            # making feature importances
            feat_imp = pd.Series(coeficients, index=features, name="feature importance").sort_values(key=abs)

            # Making the bar plot
            fig = px.bar(
                    feat_imp.tail(10), 
                    orientation = "h",
                    title = "Top 10: Decision Tree Model Feature importances plot"
                )
            fig.update_layout(
                xaxis_title = "Importances",
                yaxis_title = "Features",
                template = "plotly_white",
                legend_title = "Item"
                    
            )
            return fig
        elif model_type == "forest":
            # Get the model
            model, X, y = ModelBuilder().forest_model()
            
            # Get feature importances
            coeficients = model.named_steps["forest_model"].feature_importances_
            features  = model.named_steps["preprocess"].get_feature_names_out().ravel()
            features = [f.split("__")[1] for f in features]
            
            # making feature importances
            feat_imp = pd.Series(coeficients, index=features, name="feature importance").sort_values(key=abs)

            # Making the bar plot
            fig = px.bar(
                    feat_imp.tail(10), 
                    orientation = "h",
                    title = "Top 10: Random Forest Model Feature importances plot"
                )
            fig.update_layout(
                xaxis_title = "Importances",
                yaxis_title = "Features",
                legend_title = "Item"
                    
            )
            return fig

        else:
            # Get the model
            model, X, y = ModelBuilder().gradient_model()
            
            # Get feature importances
            coeficients = model.named_steps["forest_model"].feature_importances_
            features  = model.named_steps["preprocess"].get_feature_names_out().ravel()
            features = [f.split("__")[1] for f in features]
            
            # making feature importances
            feat_imp = pd.Series(coeficients, index=features, name="feature importance").sort_values(key=abs)

            # Making the bar plot
            fig = px.bar(
                    feat_imp.tail(10), 
                    orientation = "h",
                    title = "Top 10: Gradient Boosting Model Feature importances plot"
                )
            fig.update_layout(
                xaxis_title = "Importances",
                yaxis_title = "Features",
                legend_title = "Item"
                    
            )
            return fig