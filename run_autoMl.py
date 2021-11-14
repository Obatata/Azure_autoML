from azureml.core import  Workspace
from azureml.core.compute import AmlCompute
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment



"""
Connect the the workspace 
"""

ws =  Workspace.from_config("./config")

# get the data from the workspace
input_ds = ws.datasets.get("Defaults")
print("the input_ds is as follow : ")
print(input_ds)
"""
###############################################################################
"""


"""
Create the compute cluster 
"""
# specify the name
cluster_name = "Cluster-auto-ml"
print("the computer cluster on our workspace are as follow : \n",
      ws.compute_targets)
if cluster_name not in ws.compute_targets:
    print("Let's create the compute cluster : ")
    compute_config = \
        AmlCompute.provisioning_configuration(
                                             vm_size="STANDARD_D11_V2",
                                             max_nodes=2
                                             )
    cluster = AmlCompute.create(ws, cluster_name, compute_config)
    cluster.wait_for_completion()
else:
    cluster = ws.compute_targets[cluster_name]
    print("the cluster {}is founded in the workspace !".format(cluster_name))

print("compute cluster creation : passed !")
"""
###############################################################################
"""


"""
Configure the AutoML run 
"""
automl_config = AutoMLConfig(
                             task="classification",
                             compute_target=cluster,
                             training_data=input_ds,
                             validation_size=0.3,
                             label_column_name="Loan_Status",
                             primary_metric='norm_macro_recall',
                             iterations=2,
                             max_concurrent_iterations=2,
                             experiment_timeout_hours=0.5,
                             featurization='auto'
                            )
print("config automl creattion : passed !")
"""
###############################################################################
"""


"""
Create and submit the experiment
"""
new_experiment = Experiment(ws, "autoMl_exp_01")
new_run = new_experiment.submit(automl_config)
new_run.wait_for_completion(show_output=True)
print("experiment creation : passed !")
"""
###############################################################################
"""


"""
Create and submit the experiment
"""
new_experiment = Experiment(ws, "autoMl_exp_01")
new_run = new_experiment.submit(automl_config)
new_run.wait_for_completion(show_output=True)
print("submit experiment creation : passed !")
"""
###############################################################################
"""


"""
Retrive the best model 
"""
best_model = new_run.get_best_child(metric="accuracy")
print("best model : \n", best_model)
print("---------------------------------------------------------------------")

for run in new_run.get_children():
    print("")
    print("Run ID {} \n accuracy {} \n norm_macro_recall {} \n".format(
                                    run.id,
                                    run.get_metrics("accuracy"),
                                    run.get_metrics("norm_macro_recall")
                                    )
        )

"""
###############################################################################
"""