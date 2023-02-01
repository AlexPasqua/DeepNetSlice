import wandb
api = wandb.Api()

run = api.run("alexpasqua/Dynamic load/s3pptup5")
run.config["dynamic load range"] = "0-0.8"
run.update()