
from ipywidgets import Layout, Button, Box, VBox
from IPython.display import display
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import os
import sys
pd.options.mode.chained_assignment = None 
import os.path

import ipywidgets as widgets
from ipywidgets import interactive
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class Hyper_Parameter_Guide(widgets.DOMWidget):
    def __init__(self, X_train, X_test, y_train, y_test, dataset_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.current_algo = ''
        self.current_hp_box = None
        
        self.run = widgets.Button(description='Confirm!', disabled=False, button_style='info',
                                  tooltip='Click me', icon='check')

        self.algo_level = 2
        self.guidance_level = 4
        self.param_level = 6
        self.training_level = 8

    def init(self):
        type_question = widgets.HTML('<h1>What kind of algorithm do you want to use?</h1>')
        
        self.ml_types = [
            Button(description='Classification', layout=Layout(flex='2 1 0%', width='auto')),
            Button(description='Regression', layout=Layout(flex='2 1 0%', width='auto'))
        ]
        for ml_type in self.ml_types:
            ml_type.on_click(self.show_types)
        
        self.box_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='100%')
        ml_types_box = Box(children=self.ml_types, layout=self.box_layout)
        
        self.container = VBox([type_question, ml_types_box])
        display(self.container)

    def show_types(self, button):
        for btn in self.ml_types:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        
        algo_question = widgets.HTML('<h2>Which {} algorithm do you want to run?</h2>'.format(button.description))
        self.container.children = tuple(list(self.container.children)[:self.algo_level] + [algo_question])
        
        algo_box = Box()
 
        if button.description == 'Classification':
            algo_box = self.get_classification_algos()
        elif button.description == 'Regression':
            algo_box = self.get_regression_algos()
            
        self.container.children = tuple(list(self.container.children)[:self.algo_level+1] + [algo_box])

    def get_classification_algos(self):
        self.classification_algos = [
            Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='knn', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='SVM', layout=Layout(flex='3 1 auto', width='auto'))
        ]
        for algo in self.classification_algos:
            algo.on_click(self.show_algos)
        return Box(children=self.classification_algos, layout=self.box_layout)
    
    def show_algos(self, button):
        for btn in self.get_current_algo_btns():
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        guidance_question = widgets.HTML('<h3>In which setting do you prefer to run {}?</h3>'.format(button.description))
        self.container.children = tuple(list(self.container.children)[:self.guidance_level] + [guidance_question])
                
        self.guidance_types = [Button(description='Default', layout=Layout(flex='3 1 auto', width='auto')),
                                 Button(description='Supported', layout=Layout(flex='3 1 auto', width='auto')),
                                 Button(description='Profi', layout=Layout(flex='3 1 auto', width='auto'))]
        
        guidance_box = Box(children=self.guidance_types, layout=self.box_layout)
        for btn in self.guidance_types:
            btn.on_click(self.show_hyperparamters)
        self.container.children = tuple(list(self.container.children)[:self.guidance_level+1] + [guidance_box])

    def get_regression_algos(self):
        self.regression_algos = [
            Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='Linear Regression', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='Logistic Regression', layout=Layout(flex='3 1 auto', width='auto'))
        ]
        for algo in self.regression_algos:
            algo.on_click(self.show_algos)
        return Box(children=self.regression_algos, layout=self.box_layout)
    
    def show_hyperparamters(self, button):
        for btn in self.guidance_types:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        if self.get_active_btn(self.ml_types).description == 'Classification':
            self.show_classification_hyperparams(button)
        else:
            self.show_regression_hyperparams(button)
            
    def show_classification_hyperparams(self, button):
        if self.get_active_btn(self.classification_algos).description == 'Random Forest':
            self.show_rf_classification_hyperparams(button)
        elif self.get_active_btn(self.classification_algos).description == 'knn':
            self.show_knn_hyperparams(button)
        else:
            self.show_svm_params(button)
            
    def show_regression_hyperparams(self, button):
        if self.get_active_btn(self.regression_algos).description == 'Random Forest':
            self.show_rf_regression_hyperparams(button)
        elif self.get_active_btn(self.regression_algos).description == 'Linear Regression':
            self.show_lin_regression_hyperparams(button)
        else:
            self.show_log_regression_params(button)
            
    def show_rf_regression_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'reg_rf_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] +
                                            [widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                                                          "{n_estimators: 100, <br/>criterion: 'squared_error', <br/>max_depth: None, <br/>"
                                                          "min_samples_split: 2, <br/>min_samples_leaf: 1, <br/>min_weight_fraction_leaf: 0.0, <br/>"
                                                          "max_features: 'auto', <br/>max_leaf_nodes: None, <br/>min_impurity_decrease: 0.0, <br/>"
                                                          "bootstrap: True, <br/>oob_score: False, <br/>warm_start: False, <br/>"
                                                          "ccp_alpha: 0.0, <br/>max_samples: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'reg_rf_sup'
            self.current_hp_box = self.create_box_reg_rf_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'reg_rf_pro'
            self.current_hp_box = self.create_box_reg_rf_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
            
        self.container.children = tuple(list(self.container.children)[:self.param_level+1] + [self.run])

        self.run.on_click(self.reg_rf)

    def show_rf_classification_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'class_rf_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] +
                                            [widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                                                          "{n_estimators: 100, <br/>criterion: 'gini', <br/>"
                                                          "max_depth: None, <br/>min_samples_split: 2, <br/>"
                                                          "min_samples_leaf: 1, <br/>min_weight_fraction_leaf: 0.0,<br/>"
                                                          "max_features: 'auto', <br/>max_leaf_nodes: None, <br/>"
                                                          "min_impurity_decrease: 0.0, <br/>bootstrap: True,<br/>"
                                                          "oob_score: False, <br/>warm_start: 'False', <br/>"
                                                          "class_weight: None, <br/>ccp_alpha: 0.0, <br/>max_samples: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'class_rf_sup'
            self.current_hp_box = self.create_box_class_rf_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'class_rf_pro'
            self.current_hp_box = self.create_box_class_rf_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level+1] + [self.run])

        self.run.on_click(self.class_rf)

    def show_lin_regression_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'reg_lin_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{fit_intercept: True, <br/>normalize: False, <br/>copy_X: True, <br/>n_jobs: None,"
                             "<br/>positive: False}")])
        elif button.description == 'Supported':
            self.current_algo = 'reg_lin_sup'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters (the same as default) will be used for training. Please confirm. <br/>"
                             "{fit_intercept: True, <br/>normalize: False, <br/>copy_X: True, <br/>n_jobs: None},"
                             "<br/>positive: False")])
        elif button.description == 'Profi':
            self.current_algo = 'reg_lin_pro'
            self.current_hp_box = self.create_box_reg_lin_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run])
        self.run.on_click(self.reg_lin)
        
    def show_log_regression_params(self, button):
        if button.description == 'Default':
            self.current_algo = 'reg_log_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{penalty: 'l2', <br/>dual: False, <br/>tol: 0.0001, <br/>C: 1.0, <br/>fit_intercept: True, <br/>"
                             "intercept_scaling: 1.0, <br/>class_weight: None, <br/>solver: 'lbfgs', <br/>max_iter: 100, <br/>"
                             "multi_class: 'auto', <br/>warm_start: False, <br/>l1_ratio: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'reg_log_sup'
            self.current_hp_box = self.create_box_reg_log_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'reg_log_pro'
            self.current_hp_box = self.create_box_reg_log_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run])
        self.run.on_click(self.reg_log)
        
    def show_knn_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'class_knn_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{n_neighbors: 5, <br/>weights: 'uniform', <br/>algorithm: 'auto', <br/>leaf_size: 30, <br/>"
                             "metric_params: None, <br/>p: 2, <br/>metric: 'minkowski', <br/>n_jobs: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'class_knn_sup'
            self.current_hp_box = self.create_box_class_knn_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'class_knn_pro'
            self.current_hp_box = self.create_box_class_knn_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run])

        self.run.on_click(self.class_knn)
        
    def show_svm_params(self, button):
        if button.description == 'Default':
            self.current_algo = 'class_svm_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{C: 1.0, <br/>kernel: 'rbf', <br/>degree: 3, <br/>gamma: 'scale', <br/>coef0: 0.0, <br/>"
                             "shrinking: True, <br/>probability: False, <br/>tol: 0.001, <br/>cache_size: 200.0, <br/>"
                             "class_weight: None, <br/>max_iter: -1, <br/>verbose: False, <br/>"
                             "decision_function_shape: 'ovr', <br/>break_ties: False, <br/>random_state: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'class_svm_sup'
            self.current_hp_box = self.create_box_class_svm_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'class_svm_pro'
            self.current_hp_box = self.create_box_class_svm_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run])

        self.run.on_click(self.class_svm)

    def get_active_btn(self, btn_array):
        return [btn for btn in btn_array if btn.style.button_color == 'lightblue'][0]
    
    def get_current_algo_btns(self):
        return self.classification_algos if self.get_active_btn(self.ml_types).description == 'Classification' \
            else self.regression_algos

    def create_box_class_knn_sup(self):
        class_knn_sup_n_neighbors = widgets.IntSlider(min=1, max=len(self.X_train) / 2,
                                                      value=len(self.X_train) ** (1 / 2),
                                                      step=1, description="n-neighbors",
                                                      style={'description_width': 'initial'})
        def react(slider):
            class_knn_sup_n_neighbors.style.handle_color = 'green' if slider <= len(self.X_train) ** (
                        1 / 2) + 5 and slider >= (len(self.X_train) ** (1 / 2)) / 2 - 5 else 'red'

        box_class_knn_sup = interactive(react, slider=class_knn_sup_n_neighbors)
        return box_class_knn_sup

    def create_box_class_knn_pro(self):
        class_knn_pro_fields = {
            'n-neighbors': ('', 5),
            'weights': (['uniform', 'distance'], 'uniform'),
            'algorithm': (['auto', 'ball_tree', 'kd_tree', 'brute'], 'auto'),
            'leaf size': ('', 30),
            'p': ('int', 2),
            'metric': ('str', 'minkowski'),
            'n_jobs': ('', -1)}

        widget_class_knn_pro = {}
        widget_class_knn_pro['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_class_knn_pro_left = []
        vbox_widgets_class_knn_pro_right = []
        for hp_name, hp_tuple in class_knn_pro_fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n-neighbors':
                layout = widgets.Layout(width='70%')
                text_box = widgets.BoundedIntText(placeholder=hp_tuple[0], value=hp_tuple[1], min=1, max=len(self.X_train), layout=layout, disabled=False)
            if hp_name == 'leaf size' or hp_name == 'p' or hp_name == 'n_jobs':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            elif hp_name == 'algorithm' or hp_name == 'weights':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            elif  hp_name == 'metric':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)

            widget_class_knn_pro[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'n-neighbors' or hp_name == 'algorithm' or hp_name == 'p':
                vbox_widgets_class_knn_pro_left.append(widget_class_knn_pro[hp_name])
            else:
                vbox_widgets_class_knn_pro_right.append(widget_class_knn_pro[hp_name])

        widget_class_knn_pro['grid'][0, 0] = widgets.VBox(children=vbox_widgets_class_knn_pro_left)
        widget_class_knn_pro['grid'][0, 1] = widgets.VBox(children=vbox_widgets_class_knn_pro_right)
        widget_class_knn_pro['grid'].grid_gap = '20px'
        children_class_knn_pro = [widget_class_knn_pro['grid']]
        box_class_knn_pro = widgets.VBox(children=children_class_knn_pro)
        return box_class_knn_pro

    def create_box_reg_lin_pro(self):
        reg_lin_pro_fields = {
            'fit intercept': (True, ''),
            'normalize': (False, ''),
            'copy X': (True, ''),
            'n_jobs': ('', -1),
            'positive': (False, '')}

        widget_reg_lin_pro = {}
        widget_reg_lin_pro['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_reg_lin_pro_left = []
        vbox_widgets_reg_lin_pro_right = []
        for hp_name, hp_tuple in reg_lin_pro_fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n_jobs':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            else:
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget_reg_lin_pro[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'fit intercept' or hp_name == 'copy X' or hp_name == 'positive':
                vbox_widgets_reg_lin_pro_left.append(widget_reg_lin_pro[hp_name])
            else:
                vbox_widgets_reg_lin_pro_right.append(widget_reg_lin_pro[hp_name])

        widget_reg_lin_pro['grid'][0, 0] = widgets.VBox(children=vbox_widgets_reg_lin_pro_left)
        widget_reg_lin_pro['grid'][0, 1] = widgets.VBox(children=vbox_widgets_reg_lin_pro_right)
        widget_reg_lin_pro['grid'].grid_gap = '20px'
        children_reg_lin_pro = [widget_reg_lin_pro['grid']]
        box_reg_lin_pro = widgets.VBox(children=children_reg_lin_pro)
        return box_reg_lin_pro

    def create_box_reg_log_pro(self):
        reg_log_pro_fields = {
            'penalty': (['l1', 'l2', 'elasticnet', 'none'], 'l2'),
            'dual': (False, ''),
            'tol': ('', 0.0001),
            'C': ('', 1.0),
            'fit intercept': (True, ''),
            'intercept scaling': ('', 1.0),
            'class weight': (['balanced', None], 'balanced'),
            'random state': ('int or None', 'None'),
            'solver': (['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'lbfgs'),
            'max iter': ('', 100),
            'multi class': (['auto', 'ovr', 'multinomial'], 'auto'),
            'verbose': (False, ''),
            'warm start': (False, ''),
            'n_jobs': ('int or None', 'None'),
            'l1 ration': ('float or None', 'None')}

        widget_reg_log_pro = {}
        widget_reg_log_pro['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_reg_log_pro_left = []
        vbox_widgets_reg_log_pro_right = []
        for hp_name, hp_tuple in reg_log_pro_fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'max iter':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'class weight' or hp_name == 'penalty' or hp_name == 'solver' or hp_name == 'multi class':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], layout=layout, disabled=False)
            if hp_name == 'random state' or hp_name == 'n_jobs' or hp_name == 'l1 ration':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'tol' or hp_name == 'C' or hp_name == 'intercept scaling':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False,
                                             style={'description_width': 'initial'})
            if hp_name == 'dual' or hp_name == 'fit intercept' or hp_name == 'verbose' or hp_name == 'warm start':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget_reg_log_pro[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'penalty' or hp_name == 'fit intercept' or hp_name == 'class weight' or hp_name == 'solver' or hp_name == 'multi class':
                vbox_widgets_reg_log_pro_left.append(widget_reg_log_pro[hp_name])
            else:
                vbox_widgets_reg_log_pro_right.append(widget_reg_log_pro[hp_name])

        widget_reg_log_pro['grid'][0, 0] = widgets.VBox(children=vbox_widgets_reg_log_pro_left)
        widget_reg_log_pro['grid'][0, 1] = widgets.VBox(children=vbox_widgets_reg_log_pro_right)
        widget_reg_log_pro['grid'].grid_gap = '20px'

        children_reg_log_pro = [widget_reg_log_pro['grid']]

        box_reg_log_pro = widgets.VBox(children=children_reg_log_pro)
        return box_reg_log_pro

    def create_box_reg_log_sup(self):
        reg_log_sp_penalty = widgets.RadioButtons(options=['none', 'l2'], description="penalty")
        reg_log_sp_c = widgets.RadioButtons(options=[0.001, 0.01, 0.1, 1, 10.0, 100.0, 1000.0], description="C")

        box_reg_log_sup = widgets.HBox(children=[reg_log_sp_penalty, reg_log_sp_c])
        return box_reg_log_sup

    def create_box_class_svm_pro(self):
        class_svm_pro_fields = {
            'C': ('', 1.0),
            'kernel': (['linear','poly','rbf','sigmoid'], 'rbf'),
            'degree': ('', 3),
            'gamma': ('scale, auto or float', 'scale'),
            'coef0': ('', 0.0),
            'shrinking': (True, ''),
            'probability': (False,''),
            'tol': ('', 0.001),
            'cache size': ('', 200.0),
            'class weight': (['balanced', None], 'balanced'),
            'verbose': (False, ''),
            'max iter': ('', -1),
            'decision function shape': (['ovo','ovr'],'ovr'),
            'break ties': (False, ''),
            'random state': ('int or None', 'None')}

        widget_class_svm_pro = {}
        widget_class_svm_pro['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_class_svm_pro_left = []
        vbox_widgets_class_svm_pro_right = []
        for hp_name, hp_tuple in class_svm_pro_fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'degree' or hp_name == 'max iter':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'kernel' or hp_name == 'decision function shape' or hp_name == 'class weight':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'gamma' or hp_name == 'random state':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'C' or hp_name == 'coef0' or hp_name == 'tol' or hp_name == 'cache size':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'shrinking' or hp_name == 'probability' or hp_name == 'verbose' or hp_name=='break ties':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget_class_svm_pro[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'C' or hp_name == 'degree' or hp_name == 'gamma' or hp_name == 'coef0' or hp_name == 'probability' or hp_name == 'cache size' or hp_name == 'verbose' or hp_name == 'decision function shape':
                vbox_widgets_class_svm_pro_left.append(widget_class_svm_pro[hp_name])
            else:
                vbox_widgets_class_svm_pro_right.append(widget_class_svm_pro[hp_name])

        widget_class_svm_pro['grid'][0, 0] = widgets.VBox(children=vbox_widgets_class_svm_pro_left)
        widget_class_svm_pro['grid'][0, 1] = widgets.VBox(children=vbox_widgets_class_svm_pro_right)
        widget_class_svm_pro['grid'].grid_gap = '20px'
        children_class_svm_pro = [widget_class_svm_pro['grid']]
        box_class_svm_pro = widgets.VBox(children = children_class_svm_pro)

        return box_class_svm_pro

    def create_box_class_svm_sup(self):
        class_svm_sup_C = widgets.RadioButtons(options=[0.001,0.01,0.1,1.0,10.0,100.0,1000.0], value=1, description="C")
        class_svm_sup_kernel = widgets.RadioButtons(options=['rbf','poly'], value='rbf', description="kernel")
        class_svm_sup_gamma = widgets.RadioButtons(options=['scale',0.001,0.01,0.1,1.0,10.0,100.0,1000.0], value='scale', description="gamma")

        box_class_svm_sup = widgets.HBox(children=[class_svm_sup_C, class_svm_sup_kernel, class_svm_sup_gamma])
        return box_class_svm_sup

    def create_box_class_rf_pro(self):
        class_rf_pro_fields = {
            'n-estimators': ('', 1),
            'criterion': (['gini','entropy'], 'gini'),
            'max depth': ('int or None', 'None'),
            'min samples split': ('int or float in range (0.0, 1.0]', 0.1),
            'min samples leaf': ('int or float in range (0, 0.5]', 0.1),
            'min weight fraction leaf': ('float in range [0, 0.5]', 0),
            'max features': ('auto, sqrt, log2, int or float','auto'),
            'max leaf nodes': ('int or None','None'),
            'min impurity decrease': ('', 0) ,
            'bootstrap': (True,''),
            'oob score': (False,''),
            'n_jobs': ('', -1),
            'verbose': ('int', 0),
            'warm start': (False,''),
            'class weight': (['balanced','balanced_subsample',None], 'balanced'),
            'ccp alpha': ('', 0),
            'max samples': ('float in range (0,1] or None', 'None'),
            'random state': ('int or None', 'None')}

        widget_class_rf_pro = {}
        widget_class_rf_pro['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_class_rf_pro_left = []
        vbox_widgets_class_rf_pro_right = []
        for hp_name, hp_tuple in class_rf_pro_fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n-estimators' or hp_name == 'n_jobs' or hp_name == 'verbose':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'criterion' or hp_name == 'class weight':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'max depth' or hp_name == 'max features' or hp_name == 'max leaf nodes' or hp_name == 'max samples' or hp_name == 'random state':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'min samples split' or hp_name == 'min samples leaf' or hp_name == 'min impurity decrease' or hp_name == 'ccp alpha':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'min weight fraction leaf':
                layout = widgets.Layout(width='70%')
                text_box = widgets.BoundedFloatText(placeholder=hp_tuple[0], value=hp_tuple[1], min=0, max=0.5, step=0.001, layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'bootstrap' or hp_name == 'oob score' or hp_name == 'warm start':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget_class_rf_pro[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'n-estimators' or hp_name == 'max depth' or hp_name == 'min samples leaf' or hp_name == 'max features' or hp_name == 'min impurity decrease' or hp_name == 'oob score' or hp_name == 'verbose' or hp_name == 'class weight' or hp_name == 'max samples':
                vbox_widgets_class_rf_pro_left.append(widget_class_rf_pro[hp_name])
            else:
                vbox_widgets_class_rf_pro_right.append(widget_class_rf_pro[hp_name])

        widget_class_rf_pro['grid'][0, 0] = widgets.VBox(children=vbox_widgets_class_rf_pro_left)
        widget_class_rf_pro['grid'][0, 1] = widgets.VBox(children=vbox_widgets_class_rf_pro_right)
        widget_class_rf_pro['grid'].grid_gap = '20px'
        children_class_rf_pro = [widget_class_rf_pro['grid']]
        box_class_rf_pro = widgets.VBox(children = children_class_rf_pro)
        return box_class_rf_pro

    def create_box_class_rf_sup(self):
        class_rf_sup_n_estimators = widgets.RadioButtons(options=[50, 100, 500, 1000], value=50, description="n-estimators", style={'description_width': 'initial'})
        class_rf_sup_max_depth = widgets.IntSlider(min=1, max=len(self.X_train) * 2, value=len(self.X_train), step=1, description="max depth", style={'description_width': 'initial'})
        class_rf_sup_max_features = widgets.IntSlider(min=1, max=len(self.X_train[0]), value=len(self.X_train[0]) ** 0.5, step=1, description="max features", style={'description_width': 'initial'})

        def react_1(slider_1):
            class_rf_sup_max_depth.style.handle_color = 'green' if slider_1 <= len(self.X_train) + len(self.X_train) / 10 and slider_1 >= len(self.X_train) - len(self.X_train) / 10 else 'red'

        def react_2(slider_2):
            class_rf_sup_max_features.style.handle_color = 'green' if slider_2 <= len(self.X_train[0]) ** 0.5 + 1 and slider_2 >= len(self.X_train[0]) ** 0.5 - 1 else 'red'

        box_1_class_rf_sup = interactive(react_1, slider_1=class_rf_sup_max_depth)
        box_2_class_rf_sup = interactive(react_2, slider_2=class_rf_sup_max_features)

        box_class_rf_sup = widgets.HBox(children=[class_rf_sup_n_estimators, box_1_class_rf_sup, box_2_class_rf_sup])
        return box_class_rf_sup

    def create_box_reg_rf_pro(self):
        reg_rf_pro_fields = {
            'n-estimators': ('', 1),
            'criterion': (['mse','mae'], 'mse'),
            'max depth': ('int or None', 'None'),
            'min samples split': ('int or float in range (0.0, 1.0]', 0.1),
            'min samples leaf': ('int or float in range (0, 0.5]', 0.1),
            'min weight fraction leaf': ('float in range [0, 0.5]', 0),
            'max features': ('auto, sqrt, log2, int or float','auto'),
            'max leaf nodes': ('int or None', 'None'),
            'min impurity decrease': ('', 0.0),
            'bootstrap': (True,''),
            'oob score': (False,''),
            'n_jobs': ('', -1),
            'verbose': ('', 0),
            'warm start': (False,''),
            'ccp alpha': ('', 0.0),
            'max samples': ('float in range (0,1] or None', 'None'),
            'random state': ('int or None', 'None')
            }

        widget_reg_rf_pro = {}
        widget_reg_rf_pro['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_reg_rf_pro_left = []
        vbox_widgets_reg_rf_pro_right = []
        for hp_name, hp_tuple in reg_rf_pro_fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n-estimators' or hp_name == 'n_jobs' or hp_name == 'verbose':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'criterion':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], layout=layout, disabled=False)
            if hp_name == 'max depth' or hp_name == 'max features' or hp_name == 'max leaf nodes' or hp_name == 'max samples' or hp_name == 'random state':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'min samples split' or hp_name == 'min samples leaf' or hp_name == 'min impurity decrease' or hp_name == 'ccp alpha':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'min weight fraction leaf':
                layout = widgets.Layout(width='70%')
                text_box = widgets.BoundedFloatText(placeholder=hp_tuple[0], value=hp_tuple[1], min=0, max=0.5, step=0.001, layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'bootstrap' or hp_name == 'oob score' or hp_name == 'warm start':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget_reg_rf_pro[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'n-estimators' or hp_name == 'max depth' or hp_name == 'min samples split' or hp_name == 'min weight fraction leaf' or hp_name == 'max leaf nodes' or hp_name == 'bootstrap' or hp_name == 'n_jobs' or hp_name == 'warm start' or hp_name == 'max samples':
                vbox_widgets_reg_rf_pro_left.append(widget_reg_rf_pro[hp_name])
            else:
                vbox_widgets_reg_rf_pro_right.append(widget_reg_rf_pro[hp_name])

        widget_reg_rf_pro['grid'][0, 0] = widgets.VBox(children=vbox_widgets_reg_rf_pro_left)
        widget_reg_rf_pro['grid'][0, 1] = widgets.VBox(children=vbox_widgets_reg_rf_pro_right)
        widget_reg_rf_pro['grid'].grid_gap = '20px'
        children_reg_rf_pro = [widget_reg_rf_pro['grid']]
        box_reg_rf_pro = widgets.VBox(children = children_reg_rf_pro)
        return box_reg_rf_pro

    def create_box_reg_rf_sup(self):
        reg_rf_sup_n_estimators = widgets.RadioButtons(options=[50,100,500,1000], value =50, description="n-estimators", style={'description_width': 'initial'})
        reg_rf_sup_max_depth = widgets.IntSlider(min=1, max=len(self.X_train)*2, value=len(self.X_train), step=1, description="max depth", style={'description_width': 'initial'})
        reg_rf_sup_max_features = widgets.IntSlider(min=1, max=len(self.X_train[0]), value=len(self.X_train[0])/3, step=1,  description="max features", style={'description_width': 'initial'})

        def react_1(slider_1):
            reg_rf_sup_max_depth.style.handle_color = 'green' if slider_1<=len(self.X_train)+len(self.X_train)/10 and slider_1>=len(self.X_train)-len(self.X_train)/10  else 'red'

        def react_2(slider_2):
            reg_rf_sup_max_features.style.handle_color = 'green' if slider_2<=len(self.X_train[0]) and slider_2>=len(self.X_train[0])/3-1  else 'red'

        box_1_reg_rf_sup = interactive(react_1, slider_1=reg_rf_sup_max_depth)
        box_2_reg_rf_sup = interactive(react_2, slider_2=reg_rf_sup_max_features)

        box_reg_rf_sup = widgets.HBox(children=[reg_rf_sup_n_estimators, box_1_reg_rf_sup, box_2_reg_rf_sup])
        return box_reg_rf_sup

    def class_knn(self, run):
        if self.current_algo == 'class_knn_def':
            filename = 'class_knn.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_knn_def = pd.read_csv(fullname ,sep=';')

            else:
                df_class_knn_def = pd.DataFrame(columns = ['n_neighbors', 'weights', 'algorithm', 'leaf_size',
                                                           'p', 'metric', 'train_accuracy', 'test_accuracy'])

            trained_before_df_class_knn_def = df_class_knn_def.loc[(df_class_knn_def['n_neighbors']==5)&
                                                  (df_class_knn_def['weights']=='uniform')&
                                                  (df_class_knn_def['algorithm']=='auto')&
                                                  (df_class_knn_def['leaf_size']==30)&
                                                  (df_class_knn_def['p']==2)&
                                                  (df_class_knn_def['metric']=='minkowski')]
            if len(trained_before_df_class_knn_def) > 0:
                result_class_knn_def = widgets.Output()
                with result_class_knn_def:
                    display(trained_before_df_class_knn_def)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_knn_def = widgets.HBox([result_class_knn_def], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_knn_def])
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf_class_knn_def = KNeighborsClassifier()
                clf_class_knn_def.fit(self.X_train, self.y_train)
                train_pred_class_knn_def = clf_class_knn_def.predict(self.X_train)
                test_pred_class_knn_def = clf_class_knn_def.predict(self.X_test)
                train_acc_class_knn_def = accuracy_score(self.y_train, train_pred_class_knn_def)
                test_acc_class_knn_def = accuracy_score(self.y_test, test_pred_class_knn_def)

                result_class_knn_def = f'Training Accuracy: {round(train_acc_class_knn_def,2)*100}%. Test Accuracy:  {round(test_acc_class_knn_def,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: '+ result_class_knn_def)])

                df_class_knn_def = df_class_knn_def.append({'n_neighbors': 5, 'weights': 'uniform',
                                                            'algorithm': 'auto', 'leaf_size': 30,
                                                            'p': 2, 'metric': 'minkowski',
                                                            'train_accuracy': train_acc_class_knn_def,
                                                            'test_accuracy': test_acc_class_knn_def}, ignore_index = True)
                df_class_knn_def.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_knn_sup':
            filename = 'class_knn.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_knn_sup = pd.read_csv(fullname ,sep=';')

            else:
                df_class_knn_sup = pd.DataFrame(columns = ['n_neighbors', 'weights', 'algorithm', 'leaf_size',
                                                           'p', 'metric', 'train_accuracy', 'test_accuracy'])

            n_neighbors = self.current_hp_box.children[0].value

            trained_before_df_class_knn_sup = df_class_knn_sup.loc[(df_class_knn_sup['n_neighbors']==n_neighbors)]
            if len(trained_before_df_class_knn_sup) > 0:
                result_class_knn_sup = widgets.Output()
                with result_class_knn_sup:
                    display(trained_before_df_class_knn_sup)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_knn_sup = widgets.HBox([result_class_knn_sup], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_knn_sup])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_class_knn_sup = KNeighborsClassifier(n_neighbors=n_neighbors)
                    clf_class_knn_sup.fit(self.X_train, self.y_train)
                    train_pred_class_knn_sup = clf_class_knn_sup.predict(self.X_train)
                    test_pred_class_knn_sup = clf_class_knn_sup.predict(self.X_test)
                    train_acc_class_knn_sup = accuracy_score(self.y_train, train_pred_class_knn_sup)
                    test_acc_class_knn_sup = accuracy_score(self.y_test, test_pred_class_knn_sup)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_class_knn_sup = f'Training Accuracy: {round(train_acc_class_knn_sup,2)*100}%. Test Accuracy:  {round(test_acc_class_knn_sup,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: '+ result_class_knn_sup)])

                df_class_knn_sup = df_class_knn_sup.append({'n_neighbors': n_neighbors, 'weights': 'uniform',
                                                            'algorithm': 'auto', 'leaf_size': 30,
                                                            'p': 2, 'metric': 'minkowski',
                                                            'train_accuracy': train_acc_class_knn_sup,
                                                            'test_accuracy': test_acc_class_knn_sup}, ignore_index = True)
                df_class_knn_sup.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_knn_pro':
            filename = 'class_knn.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_knn_pro = pd.read_csv(fullname, sep=';')

            else:
                df_class_knn_pro = pd.DataFrame(columns=['n_neighbors', 'weights', 'algorithm', 'leaf_size',
                                                        'p', 'metric', 'train_accuracy', 'test_accuracy'])

            n_neighbors= self.current_hp_box.children[0].children[0].children[0].children[1].value
            weights=self.current_hp_box.children[0].children[1].children[0].children[1].value
            algorithm=self.current_hp_box.children[0].children[0].children[1].children[1].value
            leaf_size= self.current_hp_box.children[0].children[1].children[1].children[1].value
            p= self.current_hp_box.children[0].children[0].children[2].children[1].value
            metric=self.current_hp_box.children[0].children[1].children[2].children[1].value
            n_jobs=self.current_hp_box.children[0].children[1].children[3].children[1].value

            trained_before_df_class_knn_pro = df_class_knn_pro.loc[(df_class_knn_pro['n_neighbors']==n_neighbors)&
                                                  (df_class_knn_pro['weights']==weights)&
                                                  (df_class_knn_pro['algorithm']==algorithm)&
                                                  (df_class_knn_pro['leaf_size']==leaf_size)&
                                                  (df_class_knn_pro['p']==p)&
                                                  (df_class_knn_pro['metric']==metric)]

            if len(trained_before_df_class_knn_pro) > 0:
                result_class_knn_pro = widgets.Output()
                with result_class_knn_pro:
                    display(trained_before_df_class_knn_pro)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_knn_pro = widgets.HBox([result_class_knn_pro], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_knn_pro])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_class_knn_pro = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                                             leaf_size=leaf_size, p=p, metric=metric,n_jobs=n_jobs)
                    clf_class_knn_pro.fit(self.X_train, self.y_train)
                    train_pred_class_knn_pro = clf_class_knn_pro.predict(self.X_train)
                    test_pred_class_knn_pro = clf_class_knn_pro.predict(self.X_test)
                    train_acc_class_knn_pro = accuracy_score(self.y_train, train_pred_class_knn_pro)
                    test_acc_class_knn_pro = accuracy_score(self.y_test, test_pred_class_knn_pro)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_class_knn_pro = f'Training Accuracy: {round(train_acc_class_knn_pro,2)*100}%. Test Accuracy:  {round(test_acc_class_knn_pro,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: '+ result_class_knn_pro)])

                df_class_knn_pro = df_class_knn_pro.append({'n_neighbors': n_neighbors, 'weights': weights,
                                                            'algorithm': algorithm, 'leaf_size': leaf_size,
                                                            'p': p, 'metric': metric,
                                                            'train_accuracy': train_acc_class_knn_pro,
                                                            'test_accuracy': test_acc_class_knn_pro}, ignore_index = True)
                df_class_knn_pro.to_csv(fullname, sep=';', index=False)

    def reg_rf(self, run):
        if self.current_algo == 'reg_rf_def':
            filename = 'reg_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_rf_def = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_rf_def = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'ccp_alpha', 'max_samples',
                                                        'random_state', 'train_mse', 'test_mse'])
            df_reg_rf_def['max_depth'] = df_reg_rf_def['max_depth'].astype('str')
            df_reg_rf_def['max_features'] = df_reg_rf_def['max_features'].astype('str')
            df_reg_rf_def['max_leaf_nodes'] = df_reg_rf_def['max_leaf_nodes'].astype('str')
            df_reg_rf_def['max_samples'] = df_reg_rf_def['max_samples'].astype('str')

            trained_before_df_reg_rf_def = df_reg_rf_def.loc[(df_reg_rf_def['n_estimators']==100)&
                                                             (df_reg_rf_def['criterion']=='mse')&
                                                             (df_reg_rf_def['max_depth']=='None')&
                                                             (df_reg_rf_def['min_samples_split']==2.0)&
                                                             (df_reg_rf_def['min_samples_leaf']==1.0)&
                                                             (df_reg_rf_def['min_weight_fraction_leaf']==0.0)&
                                                             (df_reg_rf_def['max_features']=='auto')&
                                                             (df_reg_rf_def['max_leaf_nodes']=='None')&
                                                             (df_reg_rf_def['min_impurity_decrease']==0.0)&
                                                             (df_reg_rf_def['bootstrap']==True)&
                                                             (df_reg_rf_def['oob_score']==False)&
                                                             (df_reg_rf_def['warm_start']==False)&
                                                             (df_reg_rf_def['ccp_alpha']==0.0)&
                                                             (df_reg_rf_def['max_samples']=='None')]
            if len(trained_before_df_reg_rf_def) > 0:
                result_reg_rf_def = widgets.Output()
                with result_reg_rf_def:
                    display(trained_before_df_reg_rf_def)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_rf_def = widgets.HBox([result_reg_rf_def], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                            [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_rf_def])
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf_reg_rf_def = RandomForestRegressor()
                clf_reg_rf_def.fit(self.X_train, self.y_train)
                train_pred_reg_rf_def = clf_reg_rf_def.predict(self.X_train)
                test_pred_reg_rf_def = clf_reg_rf_def.predict(self.X_test)
                train_acc_reg_rf_def = mean_squared_error(self.y_train, train_pred_reg_rf_def)
                test_acc_reg_rf_def = mean_squared_error(self.y_test, test_pred_reg_rf_def)

                result_reg_rf_def = f'Training MSE: {round(train_acc_reg_rf_def,5)}. Test MSE:  {round(test_acc_reg_rf_def,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_reg_rf_def)])
                df_reg_rf_def = df_reg_rf_def.append({'n_estimators': 100, 'criterion': 'mse',
                                                      'max_depth': str(None), 'min_samples_split': 2.0,
                                                      'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                      'max_features': 'auto', 'max_leaf_nodes': str(None),
                                                      'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                      'oob_score': False, 'warm_start': False,'ccp_alpha': 0.0,
                                                      'max_samples': str(None), 'random_state': str(None),
                                                      'train_mse': train_acc_reg_rf_def,
                                                      'test_mse': test_acc_reg_rf_def}, ignore_index = True)
                df_reg_rf_def.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_rf_sup':
            filename = 'reg_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_rf_sup = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_rf_sup = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'ccp_alpha', 'max_samples',
                                                        'random_state', 'train_mse', 'test_mse'])

            df_reg_rf_sup['max_depth'] = df_reg_rf_sup['max_depth'].astype('str')
            df_reg_rf_sup['max_features'] = df_reg_rf_sup['max_features'].astype('str')

            n_estimators = self.current_hp_box.children[0].value
            max_depth = self.current_hp_box.children[1].children[0].value
            max_features = self.current_hp_box.children[2].children[0].value

            trained_before_df_reg_rf_sup = df_reg_rf_sup.loc[(df_reg_rf_sup['n_estimators']==n_estimators)&
                                                             (df_reg_rf_sup['max_depth']==str(max_depth))&
                                                             (df_reg_rf_sup['max_features']==str(max_features))]
            if len(trained_before_df_reg_rf_sup) > 0:
                result_reg_rf_sup = widgets.Output()
                with result_reg_rf_sup:
                    display(trained_before_df_reg_rf_sup)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_rf_sup = widgets.HBox([result_reg_rf_sup], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_rf_sup])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_reg_rf_sup = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,
                                                           max_features=max_features)
                    clf_reg_rf_sup.fit(self.X_train, self.y_train)
                    train_pred_reg_rf_sup = clf_reg_rf_sup.predict(self.X_train)
                    test_pred_reg_rf_sup = clf_reg_rf_sup.predict(self.X_test)
                    train_acc_reg_rf_sup = mean_squared_error(self.y_train, train_pred_reg_rf_sup)
                    test_acc_reg_rf_sup = mean_squared_error(self.y_test, test_pred_reg_rf_sup)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_reg_rf_sup = f'Training MSE: {round(train_acc_reg_rf_sup,5)}. Test MSE:  {round(test_acc_reg_rf_sup,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: '+ result_reg_rf_sup)])

                df_reg_rf_sup = df_reg_rf_sup.append({'n_estimators': n_estimators, 'criterion': 'mse',
                                                      'max_depth': str(max_depth), 'min_samples_split': 2.0,
                                                      'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                      'max_features': str(max_features), 'max_leaf_nodes': str(None),
                                                      'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                      'oob_score': False, 'warm_start': False,'ccp_alpha': 0.0,
                                                      'max_samples': str(None), 'random_state': str(None),
                                                      'train_mse': train_acc_reg_rf_sup,
                                                      'test_mse': test_acc_reg_rf_sup}, ignore_index = True)

                df_reg_rf_sup['max_depth'] = df_reg_rf_sup['max_depth'].astype('str')
                df_reg_rf_sup['max_features'] = df_reg_rf_sup['max_features'].astype('str')
                df_reg_rf_sup['max_leaf_nodes'] = df_reg_rf_sup['max_leaf_nodes'].astype('str')
                df_reg_rf_sup['max_samples'] = df_reg_rf_sup['max_samples'].astype('str')

                df_reg_rf_sup.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_rf_pro':
            filename = 'reg_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_rf_pro = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_rf_pro = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'ccp_alpha', 'max_samples',
                                                        'random_state', 'train_mse', 'test_mse'])

            df_reg_rf_pro['max_depth'] = df_reg_rf_pro['max_depth'].astype('str')
            df_reg_rf_pro['max_features'] = df_reg_rf_pro['max_features'].astype('str')
            df_reg_rf_pro['max_leaf_nodes'] = df_reg_rf_pro['max_leaf_nodes'].astype('str')
            df_reg_rf_pro['max_samples'] = df_reg_rf_pro['max_samples'].astype('str')

            n_estimators = self.current_hp_box.children[0].children[0].children[0].children[1].value
            criterion = self.current_hp_box.children[0].children[1].children[0].children[1].value
            try:
                max_depth= int(self.current_hp_box.children[0].children[0].children[1].children[1].value)
            except ValueError:
                max_depth=None

            if self.current_hp_box.children[0].children[0].children[2].children[1].value <= 1:
                min_samples_split = float(self.current_hp_box.children[0].children[0].children[2].children[1].value)
            elif self.current_hp_box.children[0].children[0].children[2].children[1].value > 1:
                min_samples_split = int(self.current_hp_box.children[0].children[0].children[2].children[1].value)

            if self.current_hp_box.children[0].children[1].children[1].children[1].value <= 0.5:
                min_samples_leaf = float(self.current_hp_box.children[0].children[1].children[1].children[1].value)
            else:
                min_samples_leaf = int(self.current_hp_box.children[0].children[1].children[1].children[1].value)
            min_weight_fraction_leaf = self.current_hp_box.children[0].children[0].children[3].children[1].value
            try:
                max_features = float(self.current_hp_box.children[0].children[1].children[2].children[1].value)
            except ValueError:
                if self.current_hp_box.children[0].children[1].children[2].children[1].value == 'None':
                    max_features=None
                else:
                    max_features = self.current_hp_box.children[0].children[1].children[2].children[1].value
            try:
                max_leaf_nodes = int(self.current_hp_box.children[0].children[0].children[4].children[1].value)
            except ValueError:
                max_leaf_nodes = None
            min_impurity_decrease=self.current_hp_box.children[0].children[1].children[3].children[1].value
            bootstrap = self.current_hp_box.children[0].children[0].children[5].children[1].value
            oob_score=self.current_hp_box.children[0].children[1].children[4].children[1].value
            n_jobs=self.current_hp_box.children[0].children[0].children[6].children[1].value
            verbose=self.current_hp_box.children[0].children[1].children[5].children[1].value
            warm_start=self.current_hp_box.children[0].children[0].children[7].children[1].value
            ccp_alpha=self.current_hp_box.children[0].children[1].children[6].children[1].value
            try:
                max_samples=float(self.current_hp_box.children[0].children[0].children[8].children[1].value)
            except ValueError:
                max_samples = None
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[7].children[1].value)
            except:
                random_state = None

            trained_before_df_reg_rf_pro = df_reg_rf_pro.loc[(df_reg_rf_pro['n_estimators']==n_estimators)&
                                                             (df_reg_rf_pro['criterion']==criterion)&
                                                             (df_reg_rf_pro['max_depth']==str(max_depth))&
                                                             (df_reg_rf_pro['min_samples_split']==min_samples_split)&
                                                             (df_reg_rf_pro['min_samples_leaf']==min_samples_leaf)&
                                                             (df_reg_rf_pro['min_weight_fraction_leaf']==min_weight_fraction_leaf)&
                                                             (df_reg_rf_pro['max_features']==str(max_features))&
                                                             (df_reg_rf_pro['max_leaf_nodes']==str(max_leaf_nodes))&
                                                             (df_reg_rf_pro['min_impurity_decrease']==min_impurity_decrease)&
                                                             (df_reg_rf_pro['bootstrap']==bootstrap)&
                                                             (df_reg_rf_pro['oob_score']==oob_score)&
                                                             (df_reg_rf_pro['warm_start']==warm_start)&
                                                             (df_reg_rf_pro['ccp_alpha']==ccp_alpha)&
                                                             (df_reg_rf_pro['max_samples']==str(max_samples))]
            if len(trained_before_df_reg_rf_pro) > 0:
                result_reg_rf_pro = widgets.Output()
                with result_reg_rf_pro:
                    display(trained_before_df_reg_rf_pro)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_rf_pro = widgets.HBox([result_reg_rf_pro], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_rf_pro])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_reg_rf_pro = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                           bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, verbose=verbose,
                                                           warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples,
                                                           random_state=random_state)
                    clf_reg_rf_pro.fit(self.X_train, self.y_train)
                    train_pred_reg_rf_pro = clf_reg_rf_pro.predict(self.X_train)
                    test_pred_reg_rf_pro = clf_reg_rf_pro.predict(self.X_test)
                    train_acc_reg_rf_pro = mean_squared_error(self.y_train, train_pred_reg_rf_pro)
                    test_acc_reg_rf_pro = mean_squared_error(self.y_test, test_pred_reg_rf_pro)
                except ValueError or ZeroDivisionError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_reg_rf_pro = f'Training MSE: {round(train_acc_reg_rf_pro,5)}. Test MSE:  {round(test_acc_reg_rf_pro,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: '+ result_reg_rf_pro)])

                df_reg_rf_pro = df_reg_rf_pro.append({'n_estimators': n_estimators, 'criterion': criterion,
                                                      'max_depth': str(max_depth), 'min_samples_split': min_samples_split,
                                                      'min_samples_leaf': min_samples_leaf, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                                      'max_features': str(max_features), 'max_leaf_nodes': str(max_leaf_nodes),
                                                      'min_impurity_decrease': min_impurity_decrease, 'bootstrap': bootstrap,
                                                      'oob_score': oob_score, 'warm_start': warm_start,'ccp_alpha': ccp_alpha,
                                                      'max_samples': str(max_samples), 'random_state': str(random_state),
                                                      'train_mse': train_acc_reg_rf_pro,
                                                      'test_mse': test_acc_reg_rf_pro}, ignore_index = True)
                df_reg_rf_pro.to_csv(fullname, sep=';', index=False)

    def class_rf(self, run):
        if self.current_algo == 'class_rf_def':
            filename = 'class_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_rf_def = pd.read_csv(fullname ,sep=';')

            else:
                df_class_rf_def = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples',
                                                        'random_state', 'train_accuracy', 'test_accuracy'])

            df_class_rf_def['max_depth'] = df_class_rf_def['max_depth'].astype('str')
            df_class_rf_def['max_features'] = df_class_rf_def['max_features'].astype('str')
            df_class_rf_def['max_leaf_nodes'] = df_class_rf_def['max_leaf_nodes'].astype('str')
            df_class_rf_def['max_samples'] = df_class_rf_def['max_samples'].astype('str')
            df_class_rf_def['class_weight'] = df_class_rf_def['class_weight'].astype('str')

            trained_before_df_class_rf_def = df_class_rf_def.loc[(df_class_rf_def['n_estimators']==100)&
                                                             (df_class_rf_def['criterion']=='gini')&
                                                             (df_class_rf_def['max_depth']=='None')&
                                                             (df_class_rf_def['min_samples_split']==2.0)&
                                                             (df_class_rf_def['min_samples_leaf']==1.0)&
                                                             (df_class_rf_def['min_weight_fraction_leaf']==0.0)&
                                                             (df_class_rf_def['max_features']=='auto')&
                                                             (df_class_rf_def['max_leaf_nodes']=='None')&
                                                             (df_class_rf_def['min_impurity_decrease']==0.0)&
                                                             (df_class_rf_def['bootstrap']==True)&
                                                             (df_class_rf_def['oob_score']==False)&
                                                             (df_class_rf_def['warm_start']==False)&
                                                             (df_class_rf_def['class_weight']=='None')&
                                                             (df_class_rf_def['ccp_alpha']==0.0)&
                                                             (df_class_rf_def['max_samples']=='None')]
            if len(trained_before_df_class_rf_def) > 0:
                result_class_rf_def = widgets.Output()
                with result_class_rf_def:
                    display(trained_before_df_class_rf_def)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_rf_def = widgets.HBox([result_class_rf_def], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_rf_def])
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf_class_rf_def = RandomForestClassifier()
                clf_class_rf_def.fit(self.X_train, self.y_train)
                train_pred_class_rf_def = clf_class_rf_def.predict(self.X_train)
                test_pred_class_rf_def = clf_class_rf_def.predict(self.X_test)
                train_acc_class_rf_def = accuracy_score(self.y_train, train_pred_class_rf_def)
                test_acc_class_rf_def = accuracy_score(self.y_test, test_pred_class_rf_def)
                result_class_rf_def = f'Training Accuracy: {round(train_acc_class_rf_def,2)*100}%. Test Accuracy:  {round(test_acc_class_rf_def,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_class_rf_def)])
                df_class_rf_def = df_class_rf_def.append({'n_estimators': 100, 'criterion': 'gini',
                                                      'max_depth': str(None), 'min_samples_split': 2.0,
                                                      'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                      'max_features': 'auto', 'max_leaf_nodes': str(None),
                                                      'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                      'oob_score': False, 'warm_start': False,'class_weight': str(None), 'ccp_alpha': 0.0,
                                                      'max_samples': str(None), 'random_state': str(None),
                                                      'train_accuracy': train_acc_class_rf_def,
                                                      'test_accuracy': test_acc_class_rf_def}, ignore_index = True)
                df_class_rf_def.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_rf_sup':
            filename = 'class_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_rf_sup = pd.read_csv(fullname ,sep=';')

            else:
                df_class_rf_sup = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                          'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                          'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                          'oob_score', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples',
                                                          'random_state', 'train_accuracy', 'test_accuracy'])

            n_estimators = self.current_hp_box.children[0].value
            max_depth = self.current_hp_box.children[1].children[0].value
            max_features = self.current_hp_box.children[2].children[0].value

            df_class_rf_sup['max_depth'] = df_class_rf_sup['max_depth'].astype('str')
            df_class_rf_sup['max_features'] = df_class_rf_sup['max_features'].astype('str')

            trained_before_df_class_rf_sup = df_class_rf_sup.loc[(df_class_rf_sup['n_estimators']==n_estimators)&
                                                             (df_class_rf_sup['max_depth']==str(max_depth))&
                                                             (df_class_rf_sup['max_features']==str(max_features))]
            if len(trained_before_df_class_rf_sup) > 0:
                result_class_rf_sup = widgets.Output()
                with result_class_rf_sup:
                    display(trained_before_df_class_rf_sup)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_rf_sup = widgets.HBox([result_class_rf_sup], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_rf_sup])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_class_rf_sup = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                                           max_features=max_features)
                    clf_class_rf_sup.fit(self.X_train, self.y_train)
                    train_pred_class_rf_sup = clf_class_rf_sup.predict(self.X_train)
                    test_pred_class_rf_sup = clf_class_rf_sup.predict(self.X_test)
                    train_acc_class_rf_sup = accuracy_score(self.y_train, train_pred_class_rf_sup)
                    test_acc_class_rf_sup = accuracy_score(self.y_test, test_pred_class_rf_sup)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_class_rf_sup = f'Training Accuracy: {round(train_acc_class_rf_sup,2)*100}%. Test Accuracy:  {round(test_acc_class_rf_sup,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: '+ result_class_rf_sup)])

                df_class_rf_sup = df_class_rf_sup.append({'n_estimators': n_estimators, 'criterion': 'gini',
                                                          'max_depth': str(max_depth), 'min_samples_split': 2.0,
                                                          'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                          'max_features': str(max_features), 'max_leaf_nodes': str(None),
                                                          'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                          'oob_score': False, 'warm_start': False,'class_weight': str(None), 'ccp_alpha': 0.0,
                                                          'max_samples': str(None), 'random_state': str(None),
                                                          'train_accuracy': train_acc_class_rf_sup,
                                                          'test_accuracy': test_acc_class_rf_sup}, ignore_index = True)

                df_class_rf_sup.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_rf_pro':
            filename = 'class_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_rf_pro = pd.read_csv(fullname ,sep=';')

            else:
                df_class_rf_pro = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                          'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                          'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                          'oob_score', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples',
                                                          'random_state', 'train_accuracy', 'test_accuracy'])


            df_class_rf_pro['max_depth'] = df_class_rf_pro['max_depth'].astype('str')
            df_class_rf_pro['max_features'] = df_class_rf_pro['max_features'].astype('str')
            df_class_rf_pro['max_leaf_nodes'] = df_class_rf_pro['max_leaf_nodes'].astype('str')
            df_class_rf_pro['max_samples'] = df_class_rf_pro['max_samples'].astype('str')
            df_class_rf_pro['class_weight'] = df_class_rf_pro['class_weight'].astype('str')

            n_estimators = self.current_hp_box.children[0].children[0].children[0].children[1].value
            criterion = self.current_hp_box.children[0].children[1].children[0].children[1].value
            try:
                max_depth=int(self.current_hp_box.children[0].children[0].children[1].children[1].value)
            except ValueError:
                max_depth=None

            if self.current_hp_box.children[0].children[1].children[1].children[1].value <= 1:
                min_samples_split = float(self.current_hp_box.children[0].children[1].children[1].children[1].value)
            elif self.current_hp_box.children[0].children[1].children[1].children[1].value > 1:
                min_samples_split = int(self.current_hp_box.children[0].children[1].children[1].children[1].value)

            if self.current_hp_box.children[0].children[0].children[2].children[1].value <= 0.5:
                min_samples_leaf = float(self.current_hp_box.children[0].children[0].children[2].children[1].value)
            else:
                min_samples_leaf = int(self.current_hp_box.children[0].children[0].children[2].children[1].value)

            min_weight_fraction_leaf = self.current_hp_box.children[0].children[1].children[2].children[1].value
            try:
                max_features = float(self.current_hp_box.children[0].children[0].children[3].children[1].value)
            except ValueError:
                if self.current_hp_box.children[0].children[0].children[3].children[1].value == 'None':
                    max_features=None
                else:
                    max_features = self.current_hp_box.children[0].children[0].children[3].children[1].value
            try:
                max_leaf_nodes = int(self.current_hp_box.children[0].children[1].children[3].children[1].value)
            except ValueError:
                max_leaf_nodes = None
            min_impurity_decrease = self.current_hp_box.children[0].children[0].children[4].children[1].value
            bootstrap = self.current_hp_box.children[0].children[1].children[4].children[1].value
            oob_score=self.current_hp_box.children[0].children[0].children[5].children[1].value
            n_jobs=self.current_hp_box.children[0].children[1].children[5].children[1].value
            verbose= self.current_hp_box.children[0].children[0].children[6].children[1].value
            warm_start=self.current_hp_box.children[0].children[1].children[6].children[1].value
            class_weight = self.current_hp_box.children[0].children[0].children[7].children[1].value
            ccp_alpha= self.current_hp_box.children[0].children[1].children[7].children[1].value
            try:
                max_samples= float(self.current_hp_box.children[0].children[0].children[8].children[1].value)
            except ValueError:
                max_samples = None
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[8].children[1].value)
            except:
                random_state = None

            trained_before_df_class_rf_pro = df_class_rf_pro.loc[(df_class_rf_pro['n_estimators']==n_estimators)&
                                                                 (df_class_rf_pro['criterion']==criterion)&
                                                                 (df_class_rf_pro['max_depth']==str(max_depth))&
                                                                 (df_class_rf_pro['min_samples_split']==min_samples_split)&
                                                                 (df_class_rf_pro['min_samples_leaf']==min_samples_leaf)&
                                                                 (df_class_rf_pro['min_weight_fraction_leaf']==min_weight_fraction_leaf)&
                                                                 (df_class_rf_pro['max_features']==str(max_features))&
                                                                 (df_class_rf_pro['max_leaf_nodes']==str(max_leaf_nodes))&
                                                                 (df_class_rf_pro['min_impurity_decrease']==min_impurity_decrease)&
                                                                 (df_class_rf_pro['bootstrap']==bootstrap)&
                                                                 (df_class_rf_pro['oob_score']==oob_score)&
                                                                 (df_class_rf_pro['warm_start']==warm_start)&
                                                                 (df_class_rf_pro['class_weight']==str(class_weight))&
                                                                 (df_class_rf_pro['ccp_alpha']==ccp_alpha)&
                                                                 (df_class_rf_pro['max_samples']==str(max_samples))]
            if len(trained_before_df_class_rf_pro) > 0:
                result_class_rf_pro = widgets.Output()
                with result_class_rf_pro:
                    display(trained_before_df_class_rf_pro)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_rf_pro = widgets.HBox([result_class_rf_pro], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_rf_pro])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_class_rf_pro = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                           bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, verbose=verbose,
                                                           warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                                           max_samples=max_samples, random_state=random_state)
                    clf_class_rf_pro.fit(self.X_train, self.y_train)
                    train_pred_class_rf_pro = clf_class_rf_pro.predict(self.X_train)
                    test_pred_class_rf_pro = clf_class_rf_pro.predict(self.X_test)
                    train_acc_class_rf_pro = accuracy_score(self.y_train, train_pred_class_rf_pro)
                    test_acc_class_rf_pro = accuracy_score(self.y_test, test_pred_class_rf_pro)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_class_rf_pro = f'Training Accuracy: {round(train_acc_class_rf_pro,2)*100}%. Test Accuracy:  {round(test_acc_class_rf_pro,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: '+ result_class_rf_pro)])

                df_class_rf_pro = df_class_rf_pro.append({'n_estimators': n_estimators, 'criterion': criterion,
                                                          'max_depth': str(max_depth), 'min_samples_split': min_samples_split,
                                                          'min_samples_leaf': min_samples_leaf, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                                          'max_features': str(max_features), 'max_leaf_nodes': str(max_leaf_nodes),
                                                          'min_impurity_decrease': min_impurity_decrease, 'bootstrap': bootstrap,
                                                          'oob_score': oob_score, 'warm_start': warm_start,'class_weight': str(class_weight), 'ccp_alpha': ccp_alpha,
                                                          'max_samples': str(max_samples), 'random_state': str(random_state),
                                                          'train_accuracy': train_acc_class_rf_pro,
                                                          'test_accuracy': test_acc_class_rf_pro}, ignore_index = True)
                df_class_rf_pro.to_csv(fullname, sep=';', index=False)

    def class_svm(self, run):
        if self.current_algo == 'class_svm_def':
            filename = 'class_svm.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_svm_def = pd.read_csv(fullname ,sep=';')

            else:
                df_class_svm_def = pd.DataFrame(columns = ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking',
                                                           'probability', 'tol', 'cache_size', 'class_weight',
                                                           'max_iter', 'decision_function_shape',
                                                           'break_ties', 'random_state', 'train_accuracy', 'test_accuracy'])

            trained_before_df_class_svm_def = df_class_svm_def.loc[(df_class_svm_def['C']==1.0)&
                                                                   (df_class_svm_def['kernel']=='rbf')&
                                                                   (df_class_svm_def['degree']==3)&
                                                                   (df_class_svm_def['gamma']=='scale')&
                                                                   (df_class_svm_def['coef0']==0.0)&
                                                                   (df_class_svm_def['shrinking']==True)&
                                                                   (df_class_svm_def['probability']==False)&
                                                                   (df_class_svm_def['tol']==0.001)&
                                                                   (df_class_svm_def['cache_size']==200.0)&
                                                                   (df_class_svm_def['class_weight']=='None')&
                                                                   (df_class_svm_def['max_iter']==-1)&
                                                                   (df_class_svm_def['decision_function_shape']=='ovr')&
                                                                   (df_class_svm_def['break_ties']==False)]
            if len(trained_before_df_class_svm_def) > 0:
                result_class_svm_def = widgets.Output()
                with result_class_svm_def:
                    display(trained_before_df_class_svm_def)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_svm_def = widgets.HBox([result_class_svm_def], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_svm_def])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf_class_svm_def = SVC()
                clf_class_svm_def.fit(self.X_train, self.y_train)
                train_pred_class_svm_def = clf_class_svm_def.predict(self.X_train)
                test_pred_class_svm_def = clf_class_svm_def.predict(self.X_test)
                train_acc_class_svm_def = accuracy_score(self.y_train, train_pred_class_svm_def)
                test_acc_class_svm_def = accuracy_score(self.y_test, test_pred_class_svm_def)

                result_class_svm_def = f'Training Accuracy: {round(train_acc_class_svm_def,2)*100}%. Test Accuracy:  {round(test_acc_class_svm_def,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_class_svm_def)])
                df_class_svm_def = df_class_svm_def.append({'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'scale',
                                                           'coef0':0.0, 'shrinking': True, 'probability': False,
                                                           'tol': 0.001, 'cache_size': 200.0, 'class_weight': str(None),
                                                           'max_iter': -1, 'decision_function_shape': 'ovr',
                                                           'break_ties': False, 'random_state': None,
                                                           'train_accuracy': train_acc_class_svm_def,'test_accuracy': test_acc_class_svm_def},
                                                          ignore_index = True)
                df_class_svm_def.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_svm_sup':
            filename = 'class_svm.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_svm_sup = pd.read_csv(fullname ,sep=';')

            else:
                df_class_svm_sup = pd.DataFrame(columns = ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking',
                                                           'probability', 'tol', 'cache_size', 'class_weight',
                                                           'max_iter', 'decision_function_shape',
                                                           'break_ties', 'random_state', 'train_accuracy', 'test_accuracy'])

            c = self.current_hp_box.children[0].value
            kernel = self.current_hp_box.children[1].value
            gamma = self.current_hp_box.children[2].value

            trained_before_df_class_svm_sup = df_class_svm_sup.loc[(df_class_svm_sup['C']==c)&
                                                                   (df_class_svm_sup['kernel']==kernel)&
                                                                   (df_class_svm_sup['gamma']==str(gamma))]
            if len(trained_before_df_class_svm_sup) > 0:
                result_class_svm_sup = widgets.Output()
                with result_class_svm_sup:
                    display(trained_before_df_class_svm_sup)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_svm_sup = widgets.HBox([result_class_svm_sup], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_svm_sup])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_class_svm_sup = SVC(C=c, kernel=kernel, gamma=gamma)
                    clf_class_svm_sup.fit(self.X_train, self.y_train)
                    train_pred_class_svm_sup = clf_class_svm_sup.predict(self.X_train)
                    test_pred_class_svm_sup = clf_class_svm_sup.predict(self.X_test)
                    train_acc_class_svm_sup = accuracy_score(self.y_train, train_pred_class_svm_sup)
                    test_acc_class_svm_sup = accuracy_score(self.y_test, test_pred_class_svm_sup)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_class_svm_sup = f'Training Accuracy: {round(train_acc_class_svm_sup,2)*100}%. Test Accuracy:  {round(test_acc_class_svm_sup,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_class_svm_sup)])
                df_class_svm_sup = df_class_svm_sup.append({'C': c, 'kernel': kernel, 'degree': 3, 'gamma': str(gamma),
                                                            'coef0':0.0, 'shrinking': True, 'probability': False,
                                                            'tol': 0.001, 'cache_size': 200.0, 'class_weight': str(None),
                                                            'max_iter': -1, 'decision_function_shape': 'ovr',
                                                            'break_ties': False, 'random_state': None,
                                                            'train_accuracy': train_acc_class_svm_sup,'test_accuracy': test_acc_class_svm_sup},
                                                           ignore_index = True)
                df_class_svm_sup.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_svm_pro':
            filename = 'class_svm.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_class_svm_pro = pd.read_csv(fullname ,sep=';')

            else:
                df_class_svm_pro = pd.DataFrame(columns = ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking',
                                                           'probability', 'tol', 'cache_size', 'class_weight',
                                                           'max_iter', 'decision_function_shape',
                                                           'break_ties', 'random_state', 'train_accuracy', 'test_accuracy'])

            c = self.current_hp_box.children[0].children[0].children[0].children[1].value
            kernel = self.current_hp_box.children[0].children[1].children[0].children[1].value
            degree = self.current_hp_box.children[0].children[0].children[1].children[1].value
            try:
                gamma = float(self.current_hp_box.children[0].children[0].children[2].children[1].value)
            except:
                gamma = self.current_hp_box.children[0].children[0].children[2].children[1].value
            coef0 = self.current_hp_box.children[0].children[0].children[3].children[1].value
            shrinking = self.current_hp_box.children[0].children[1].children[1].children[1].value
            probability = self.current_hp_box.children[0].children[0].children[4].children[1].value
            tol = self.current_hp_box.children[0].children[1].children[2].children[1].value
            cache_size = self.current_hp_box.children[0].children[0].children[5].children[1].value
            class_weight = self.current_hp_box.children[0].children[1].children[3].children[1].value
            max_iter = self.current_hp_box.children[0].children[1].children[4].children[1].value
            verbose = self.current_hp_box.children[0].children[0].children[6].children[1].value
            decision_function_shape = self.current_hp_box.children[0].children[0].children[7].children[1].value
            break_ties = self.current_hp_box.children[0].children[1].children[5].children[1].value
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[6].children[1].value)
            except:
                random_state = None

            trained_before_df_class_svm_pro = df_class_svm_pro.loc[(df_class_svm_pro['C']==c)&
                                                                   (df_class_svm_pro['kernel']==kernel)&
                                                                   (df_class_svm_pro['degree']==degree)&
                                                                   (df_class_svm_pro['gamma']==str(gamma))&
                                                                   (df_class_svm_pro['coef0']==coef0)&
                                                                   (df_class_svm_pro['shrinking']==shrinking)&
                                                                   (df_class_svm_pro['probability']==probability)&
                                                                   (df_class_svm_pro['tol']==tol)&
                                                                   (df_class_svm_pro['cache_size']==cache_size)&
                                                                   (df_class_svm_pro['class_weight']==str(class_weight))&
                                                                   (df_class_svm_pro['max_iter']==max_iter)&
                                                                   (df_class_svm_pro['decision_function_shape']==decision_function_shape)&
                                                                   (df_class_svm_pro['break_ties']==break_ties)]
            if len(trained_before_df_class_svm_pro) > 0:
                result_class_svm_pro = widgets.Output()
                with result_class_svm_pro:
                    display(trained_before_df_class_svm_pro)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_class_svm_pro = widgets.HBox([result_class_svm_pro], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_class_svm_pro])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_class_svm_pro = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                                            probability=probability, shrinking=shrinking, tol=tol,
                                            cache_size=cache_size, class_weight=class_weight, verbose=verbose,
                                            decision_function_shape=decision_function_shape, max_iter=max_iter,
                                            break_ties=break_ties, random_state=random_state)
                    clf_class_svm_pro.fit(self.X_train, self.y_train)
                    train_pred_class_svm_pro = clf_class_svm_pro.predict(self.X_train)
                    test_pred_class_svm_pro = clf_class_svm_pro.predict(self.X_test)
                    train_acc_class_svm_pro = accuracy_score(self.y_train, train_pred_class_svm_pro)
                    test_acc_class_svm_pro = accuracy_score(self.y_test, test_pred_class_svm_pro)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_class_svm_pro = f'Training Accuracy: {round(train_acc_class_svm_pro,2)*100}%. Test Accuracy:  {round(test_acc_class_svm_pro,2)*100}%'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_class_svm_pro)])
                df_class_svm_pro = df_class_svm_pro.append({'C': c, 'kernel': kernel, 'degree': degree, 'gamma': str(gamma),
                                                            'coef0': coef0, 'shrinking': shrinking, 'probability': probability,
                                                            'tol': tol, 'cache_size': cache_size, 'class_weight': str(class_weight),
                                                            'max_iter': max_iter, 'decision_function_shape': decision_function_shape,
                                                            'break_ties': break_ties,'random_state': random_state,
                                                            'train_accuracy': train_acc_class_svm_pro,'test_accuracy': test_acc_class_svm_pro},
                                                           ignore_index = True)
                df_class_svm_pro.to_csv(fullname, sep=';', index=False)

    def reg_lin(self, run):
        if self.current_algo == 'reg_lin_def':
            filename = 'reg_lin.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_lin_def = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_lin_def = pd.DataFrame(columns = ['fit_intercept', 'normalize', 'copy_X', 'positive',
                                                         'train_mse', 'test_mse'])

            trained_before_df_reg_lin_def = df_reg_lin_def.loc[(df_reg_lin_def['fit_intercept']==True)&
                                                               (df_reg_lin_def['normalize']==False)&
                                                               (df_reg_lin_def['copy_X']==True)&
                                                               (df_reg_lin_def['positive']==False)]
            if len(trained_before_df_reg_lin_def) > 0:
                result_reg_lin_def = widgets.Output()
                with result_reg_lin_def:
                    display(trained_before_df_reg_lin_def)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_lin_def = widgets.HBox([result_reg_lin_def], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_lin_def])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf_reg_lin_def = LinearRegression()
                clf_reg_lin_def.fit(self.X_train, self.y_train)
                train_pred_reg_lin_def = clf_reg_lin_def.predict(self.X_train)
                test_pred_reg_lin_def = clf_reg_lin_def.predict(self.X_test)
                train_acc_reg_lin_def = mean_squared_error(self.y_train, train_pred_reg_lin_def)
                test_acc_reg_lin_def = mean_squared_error(self.y_test, test_pred_reg_lin_def)

                result_reg_lin_def = f'Training MSE: {round(train_acc_reg_lin_def,5)}. Test MSE:  {round(test_acc_reg_lin_def,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_reg_lin_def)])
                df_reg_lin_def = df_reg_lin_def.append({'fit_intercept': True, 'normalize': False,
                                                       'copy_X': True, 'positive': False,
                                                       'train_mse': train_acc_reg_lin_def,'test_mse': test_acc_reg_lin_def},
                                                      ignore_index = True)
                df_reg_lin_def.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_lin_sup':
            filename = 'reg_lin.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_lin_sup = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_lin_sup = pd.DataFrame(columns = ['fit_intercept', 'normalize', 'copy_X', 'positive',
                                                         'train_mse', 'test_mse'])

            trained_before_df_reg_lin_sup = df_reg_lin_sup.loc[(df_reg_lin_sup['fit_intercept']==True)&
                                                               (df_reg_lin_sup['normalize']==False)&
                                                               (df_reg_lin_sup['copy_X']==True)&
                                                               (df_reg_lin_sup['positive']==False)]
            if len(trained_before_df_reg_lin_sup) > 0:
                result_reg_lin_sup = widgets.Output()
                with result_reg_lin_sup:
                    display(trained_before_df_reg_lin_sup)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_lin_sup = widgets.HBox([result_reg_lin_sup], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_lin_sup])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf_reg_lin_sup = LinearRegression()
                clf_reg_lin_sup.fit(self.X_train, self.y_train)
                train_pred_reg_lin_sup = clf_reg_lin_sup.predict(self.X_train)
                test_pred_reg_lin_sup = clf_reg_lin_sup.predict(self.X_test)
                train_acc_reg_lin_sup = mean_squared_error(self.y_train, train_pred_reg_lin_sup)
                test_acc_reg_lin_sup = mean_squared_error(self.y_test, test_pred_reg_lin_sup)

                result_reg_lin_sup = f'Training MSE: {round(train_acc_reg_lin_sup,5)}. Test MSE:  {round(test_acc_reg_lin_sup,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_reg_lin_sup)])
                df_reg_lin_sup = df_reg_lin_sup.append({'fit_intercept': True, 'normalize': False,
                                                        'copy_X': True, 'positive': False,
                                                        'train_mse': train_acc_reg_lin_sup,'test_mse': test_acc_reg_lin_sup},
                                                       ignore_index = True)
                df_reg_lin_sup.to_csv(fullname, sep=';', index=False)
        elif self.current_algo == 'reg_lin_pro':
            filename = 'reg_lin.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_lin_pro = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_lin_pro = pd.DataFrame(columns = ['fit_intercept', 'normalize', 'copy_X', 'positive',
                                                         'train_mse', 'test_mse'])

            fit_intercept=self.current_hp_box.children[0].children[0].children[0].children[1].value
            normalize=self.current_hp_box.children[0].children[1].children[0].children[1].value
            copy_X=self.current_hp_box.children[0].children[0].children[1].children[1].value
            n_jobs=self.current_hp_box.children[0].children[1].children[1].children[1].value
            positive = self.current_hp_box.children[0].children[0].children[2].children[1].value

            trained_before_df_reg_lin_pro = df_reg_lin_pro.loc[(df_reg_lin_pro['fit_intercept']==fit_intercept)&
                                                               (df_reg_lin_pro['normalize']==normalize)&
                                                               (df_reg_lin_pro['copy_X']==copy_X)&
                                                               (df_reg_lin_pro['positive']==positive)]
            if len(trained_before_df_reg_lin_pro) > 0:
                result_reg_lin_pro = widgets.Output()
                with result_reg_lin_pro:
                    display(trained_before_df_reg_lin_pro)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_lin_pro = widgets.HBox([result_reg_lin_pro], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_lin_pro])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_reg_lin_pro = LinearRegression(fit_intercept=fit_intercept, normalize=normalize,
                                                       copy_X=copy_X, n_jobs=n_jobs, positive=positive)
                    clf_reg_lin_pro.fit(self.X_train, self.y_train)
                    train_pred_reg_lin_pro = clf_reg_lin_pro.predict(self.X_train)
                    test_pred_reg_lin_pro = clf_reg_lin_pro.predict(self.X_test)
                    train_acc_reg_lin_pro = mean_squared_error(self.y_train, train_pred_reg_lin_pro)
                    test_acc_reg_lin_pro = mean_squared_error(self.y_test, test_pred_reg_lin_pro)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return


                result_reg_lin_pro = f'Training MSE: {round(train_acc_reg_lin_pro,5)}. Test MSE:  {round(test_acc_reg_lin_pro,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_reg_lin_pro)])
                df_reg_lin_pro = df_reg_lin_pro.append({'fit_intercept': fit_intercept, 'normalize': normalize,
                                                        'copy_X': copy_X, 'positive': positive,
                                                        'train_mse': train_acc_reg_lin_pro,'test_mse': test_acc_reg_lin_pro},
                                                       ignore_index = True)
                df_reg_lin_pro.to_csv(fullname, sep=';', index=False)

    def reg_log(self, run):
        if self.current_algo == 'reg_log_def':
            filename = 'reg_log.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_log_def = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_log_def = pd.DataFrame(columns = ['penalty', 'dual', 'tol', 'C', 'fit_intercept',
                                                         'intercept_scaling', 'class_weight', 'random_state','solver',
                                                         'max_iter', 'multi_class', 'warm_start', 'l1_ratio',
                                                         'train_mse', 'test_mse'])

            trained_before_df_reg_log_def = df_reg_log_def.loc[(df_reg_log_def['penalty']=='l2')&
                                                               (df_reg_log_def['dual']==False)&
                                                               (df_reg_log_def['tol']==0.0001)&
                                                               (df_reg_log_def['C']==1.0)&
                                                               (df_reg_log_def['fit_intercept']==True)&
                                                               (df_reg_log_def['intercept_scaling']==1.0)&
                                                               (df_reg_log_def['class_weight']=='None')&
                                                               (df_reg_log_def['solver']=='lbfgs')&
                                                               (df_reg_log_def['max_iter']==100)&
                                                               (df_reg_log_def['multi_class']=='auto')&
                                                               (df_reg_log_def['warm_start']==False)&
                                                               (df_reg_log_def['l1_ratio']=='None')]
            if len(trained_before_df_reg_log_def) > 0:
                result_reg_log_def = widgets.Output()
                with result_reg_log_def:
                    display(trained_before_df_reg_log_def)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_log_def = widgets.HBox([result_reg_log_def], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_log_def])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf_reg_log_def = LogisticRegression()
                clf_reg_log_def.fit(self.X_train, self.y_train)
                train_pred_reg_log_def = clf_reg_log_def.predict(self.X_train)
                test_pred_reg_log_def = clf_reg_log_def.predict(self.X_test)
                train_acc_reg_log_def = mean_squared_error(self.y_train, train_pred_reg_log_def)
                test_acc_reg_log_def = mean_squared_error(self.y_test, test_pred_reg_log_def)

                result_reg_log_def = f'Training MSE: {round(train_acc_reg_log_def,5)}. Test MSE:  {round(test_acc_reg_log_def,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_reg_log_def)])
                df_reg_log_def = df_reg_log_def.append({'penalty': 'l2', 'dual': False, 'tol': 0.0001,
                                                        'C': 1.0, 'fit_intercept': True,
                                                        'intercept_scaling': 1.0, 'class_weight': str(None), 'random_state': None,
                                                        'solver': 'lbfgs', 'max_iter': 100,'multi_class': 'auto',
                                                        'warm_start': False, 'l1_ratio': str(None),
                                                        'train_mse': train_acc_reg_log_def,'test_mse': test_acc_reg_log_def},
                                                       ignore_index = True)
                df_reg_log_def.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_log_sup':
            filename = 'reg_log.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_log_sup = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_log_sup = pd.DataFrame(columns = ['penalty', 'dual', 'tol', 'C', 'fit_intercept',
                                                         'intercept_scaling', 'class_weight', 'random_state', 'solver',
                                                         'max_iter', 'multi_class', 'warm_start', 'l1_ratio',
                                                         'train_mse', 'test_mse'])

            penalty=self.current_hp_box.children[0].value
            c=self.current_hp_box.children[1].value

            trained_before_df_reg_log_sup = df_reg_log_sup.loc[(df_reg_log_sup['penalty']==penalty)&
                                                               (df_reg_log_sup['C']==c)]
            if len(trained_before_df_reg_log_sup) > 0:
                result_reg_log_sup = widgets.Output()
                with result_reg_log_sup:
                    display(trained_before_df_reg_log_sup)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_log_sup = widgets.HBox([result_reg_log_sup], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_log_sup])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_reg_log_sup = LogisticRegression(penalty=penalty, C=c)
                    clf_reg_log_sup.fit(self.X_train, self.y_train)
                    train_pred_reg_log_sup= clf_reg_log_sup.predict(self.X_train)
                    test_pred_reg_log_sup = clf_reg_log_sup.predict(self.X_test)
                    train_acc_reg_log_sup = mean_squared_error(self.y_train, train_pred_reg_log_sup)
                    test_acc_reg_log_sup = mean_squared_error(self.y_test, test_pred_reg_log_sup)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_reg_log_sup = f'Training MSE: {round(train_acc_reg_log_sup,5)}. Test MSE:  {round(test_acc_reg_log_sup,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_reg_log_sup)])
                df_reg_log_sup = df_reg_log_sup.append({'penalty': penalty, 'dual': False, 'tol': 0.0001,
                                                        'C': c, 'fit_intercept': True,
                                                        'intercept_scaling': 1.0, 'class_weight': str(None), 'random_state': None,
                                                        'solver': 'lbfgs', 'max_iter': 100,'multi_class': 'auto',
                                                        'warm_start': False, 'l1_ratio': str(None),
                                                        'train_mse': train_acc_reg_log_sup,'test_mse': test_acc_reg_log_sup},
                                                       ignore_index = True)
                df_reg_log_sup.to_csv(fullname, sep=';', index=False)
        elif self.current_algo == 'reg_log_pro':
            filename = 'reg_log.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                df_reg_log_pro = pd.read_csv(fullname ,sep=';')

            else:
                df_reg_log_pro = pd.DataFrame(columns = ['penalty', 'dual', 'tol', 'C', 'fit_intercept',
                                                         'intercept_scaling', 'class_weight', 'random_state','solver',
                                                         'max_iter', 'multi_class', 'warm_start', 'l1_ratio',
                                                         'train_mse', 'test_mse'])

            penalty=self.current_hp_box.children[0].children[0].children[0].children[1].value
            dual=self.current_hp_box.children[0].children[1].children[0].children[1].value
            tol= self.current_hp_box.children[0].children[1].children[1].children[1].value
            c= self.current_hp_box.children[0].children[1].children[2].children[1].value
            fit_intercept= self.current_hp_box.children[0].children[0].children[1].children[1].value
            intercept_scaling=self.current_hp_box.children[0].children[1].children[3].children[1].value
            class_weight=self.current_hp_box.children[0].children[0].children[2].children[1].value
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[4].children[1].value)
            except ValueError:
                random_state = None
            solver=self.current_hp_box.children[0].children[0].children[3].children[1].value
            max_iter= self.current_hp_box.children[0].children[1].children[5].children[1].value
            multi_class=self.current_hp_box.children[0].children[0].children[4].children[1].value
            verbose=self.current_hp_box.children[0].children[1].children[6].children[1].value
            warm_start=self.current_hp_box.children[0].children[1].children[7].children[1].value
            try:
                n_jobs = int(self.current_hp_box.children[0].children[1].children[8].children[1].value)
            except ValueError:
                n_jobs = None
            try:
                l1_ratio = float(self.current_hp_box.children[0].children[1].children[9].children[1].value)
            except ValueError:
                l1_ratio = None

            trained_before_df_reg_log_pro = df_reg_log_pro.loc[(df_reg_log_pro['penalty']==penalty)&
                                                               (df_reg_log_pro['dual']==dual)&
                                                               (df_reg_log_pro['tol']==tol)&
                                                               (df_reg_log_pro['C']==c)&
                                                               (df_reg_log_pro['fit_intercept']==fit_intercept)&
                                                               (df_reg_log_pro['intercept_scaling']==intercept_scaling)&
                                                               (df_reg_log_pro['class_weight']==str(class_weight))&
                                                               (df_reg_log_pro['solver']==solver)&
                                                               (df_reg_log_pro['max_iter']==max_iter)&
                                                               (df_reg_log_pro['multi_class']==multi_class)&
                                                               (df_reg_log_pro['warm_start']==warm_start)&
                                                               (df_reg_log_pro['l1_ratio']==str(l1_ratio))]
            if len(trained_before_df_reg_log_pro) > 0:
                result_reg_log_pro = widgets.Output()
                with result_reg_log_pro:
                    display(trained_before_df_reg_log_pro)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_df_reg_log_pro = widgets.HBox([result_reg_log_pro], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_df_reg_log_pro])

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf_reg_log_pro = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=c,
                                                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                                         class_weight=class_weight, random_state=random_state, solver=solver,
                                                         max_iter=max_iter, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                                                         multi_class=multi_class, l1_ratio=l1_ratio)
                    clf_reg_log_pro.fit(self.X_train, self.y_train)
                    train_pred_reg_log_pro = clf_reg_log_pro.predict(self.X_train)
                    test_pred_reg_log_pro = clf_reg_log_pro.predict(self.X_test)
                    train_acc_reg_log_pro = mean_squared_error(self.y_train, train_pred_reg_log_pro)
                    test_acc_reg_log_pro = mean_squared_error(self.y_test, test_pred_reg_log_pro)
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                result_reg_log_pro = f'Training MSE: {round(train_acc_reg_log_pro,5)}. Test MSE:  {round(test_acc_reg_log_pro,5)}.'
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result_reg_log_pro)])
                df_reg_log_pro = df_reg_log_pro.append({'penalty': penalty, 'dual': dual, 'tol': tol,
                                                        'C': c, 'fit_intercept': fit_intercept,
                                                        'intercept_scaling': intercept_scaling, 'class_weight': str(class_weight),
                                                        'random_state':random_state, 'solver': solver, 'max_iter': max_iter,'multi_class': multi_class,
                                                        'warm_start': warm_start, 'l1_ratio': str(l1_ratio),
                                                        'train_mse': train_acc_reg_log_pro,'test_mse': test_acc_reg_log_pro},
                                                       ignore_index = True)
                df_reg_log_pro.to_csv(fullname, sep=';', index=False)