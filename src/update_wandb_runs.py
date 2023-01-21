import wandb
api = wandb.Api()

run = api.run("alexpasqua/Dynamic connectivity/3o0vtz6x")
run.config["dynamic connectivity"] = False
run.update()