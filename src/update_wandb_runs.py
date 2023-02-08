import wandb
api = wandb.Api()

run = api.run("alexpasqua/Discount factor in accept reward/ybu654op")
run.config["reward type"] = "accumulated"
run.update()