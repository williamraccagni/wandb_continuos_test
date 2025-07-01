import wandb
import os
import json
from dummy_model import DummyModel
import random

# === Config ===
PROJECT = "wandb_continuos_test"
ENTITY = "entity"  # oppure 'tuo_username' se non sei in un team
RUN_NAME = "dummy-run"

RESUME_FILE = "resume.json"

# === Resume logic ===
def load_resume_data():
    if os.path.exists(RESUME_FILE):
        
        with open(RESUME_FILE, "r") as f:
            jdata = json.load(f)
            
            if jdata["completed"] == True:
                return {"run_id": RUN_NAME + '_' + str(random.randint(0, 1000000)), "epoch": 0, "completed": False}
            else:
                return jdata
    
    else:
        return {"run_id": RUN_NAME + '_' + str(random.randint(0, 1000000)), "epoch": 0, "completed": False}

def save_resume_data(run_id, epoch, completed):
    with open(RESUME_FILE, "w") as f:
        json.dump({"run_id": run_id, "epoch": epoch, "completed": completed}, f)

resume_data = load_resume_data()
resume_run_id = resume_data["run_id"]
starting_epoch = resume_data["epoch"]
completed_run = resume_data["completed"]

save_resume_data(resume_run_id, starting_epoch, completed_run)

# === Training loop ===
model = DummyModel()
epochs = 20
number_of_steps_per_epoch = 1000 * 8

for epoch in range(starting_epoch, epochs):
    
    # === wandb init ===
    wandb.init(
        project=PROJECT,
        entity=ENTITY,
        
        id=resume_run_id,    
        name=resume_run_id,
        
        resume="allow",  # "allow" lets it resume if run_id is set
        config={"epochs": epochs},
    )
    
    steps_list = []
    
    print("Training...")
    
    # Training loop (simulated)
    for step in range(number_of_steps_per_epoch):
        loss = model.train_step()        
        global_step = epoch * number_of_steps_per_epoch + step
        
        steps_list.append((loss, global_step))
        
    # when finished save all training logs
    for item in steps_list:
        wandb.log({"train/loss": item[0]}, step=item[1])
    
    # Validation
    val_loss = model.validate()
    wandb.log({"val/loss": val_loss, "epoch": epoch})

    print(f"Epoch {epoch} completata - val_loss: {val_loss:.4f}")
    
    # Save resume data
    save_resume_data(resume_run_id, epoch + 1, False)
    
    wandb.finish()

# save completed to True
save_resume_data(resume_run_id, epoch, True)

# wandb.finish()