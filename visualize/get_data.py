import pandas as pd
import wandb
from tqdm import tqdm

api = wandb.Api()
entity, project = "carlosqsw", "component_probing"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
val_test_f1 = {
    "epoch1":[],
    "epoch2":[],
    "epoch3":[],
    "epoch4":[],
    "epoch5":[],
    "test":[]
}
for run in tqdm(runs):
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    hist = run.history()
    for i in range(1,6):
        val_test_f1[f"epoch{i}"].append(hist["val_f1"][i-1])
    val_test_f1["test"].append(hist["test_f1"][5])

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {
        "name": name_list,
        "epoch1": val_test_f1["epoch1"],
        "epoch2": val_test_f1["epoch2"],
        "epoch3": val_test_f1["epoch3"],
        "epoch4": val_test_f1["epoch4"],
        "epoch5": val_test_f1["epoch5"],
        "test": val_test_f1["test"],
        "summary": summary_list, 
        "config": config_list,
    }
)

runs_df.to_csv("project.csv")