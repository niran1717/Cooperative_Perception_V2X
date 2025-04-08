import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from tumtraf_dataset import TUMTrafDataset
from pointpillar import PointPillarModel
from loss import PointPillarLoss
from target_assigner import TargetAssigner
from label_parser import parse_labels_from_json

def train_pointpillar(
    info_path_train,
    info_path_val,
    label_root,
    batch_size=1,
    epochs=1,
    lr=1e-3,
    voxel_size=[0.16, 0.16, 4],
    pc_range=[0, -40, -3, 70.4, 40, 1],
    grid_shape=[500, 440],
    num_classes=10 # set to match number of classes in label_parser
):
    
    train_log_path = "training_log.txt"
    with open(train_log_path, "w") as f:
        f.write("Epoch,Batch,TrainLoss\n")
    
    print("✅ Initializing datasets...")
    train_dataset = TUMTrafDataset(info_path=info_path_train, modality="lidar")
    val_dataset = TUMTrafDataset(info_path=info_path_val, modality="lidar")

    print("✅ Building data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("✅ Initializing model, loss, and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointPillarModel(voxel_size, pc_range, grid_shape, num_classes = num_classes).to(device)
    loss_fn = PointPillarLoss(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("✅ Initializing target assigner...")
    assigner = TargetAssigner(grid_size=grid_shape, pc_range=pc_range, num_classes=num_classes)

    print("✅ Starting training loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n🟡 Epoch {epoch}/{epochs}")
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            print(f"  ▶️ Training batch {batch_idx + 1}/{len(train_loader)}")

            points = batch_data['lidar']    
            labels = batch_data["labels"]  

            gt_boxes = [label['bbox'] for label in labels]
            gt_classes = [label['class'] for label in labels]
            cls_targets, reg_targets = assigner.assign(gt_boxes, gt_classes)

            if len(gt_boxes) == 0:
                print("     ⚠️ Skipping batch with no labels.")
                continue

            cls_targets, reg_targets = assigner.assign(gt_boxes, gt_classes)

            if isinstance(points, list):
                points_gpu = [p.to(device) for p in points]
            else:
                points_gpu = points.to(device)

            cls_targets = cls_targets.unsqueeze(0).to(device)
            reg_targets = reg_targets.unsqueeze(0).to(device)

            print("     ➤ Running forward pass...")
            cls_preds, reg_preds = model(points_gpu)
            
            print("     ➤ Computing loss...")
            loss = loss_fn(cls_preds, reg_preds, cls_targets, reg_targets)
            with open(train_log_path, "a") as f:
                f.write(f"{epoch},{batch_idx + 1},{loss.item():.4f}\n")

            print("     ➤ Backpropagating...")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            print(f"     ✅ Batch {batch_idx + 1} loss: {loss.item():.4f}")

        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch} average training loss: {avg_train_loss:.4f}")

        print("🔍 Running validation...")
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_idx, batch_data in enumerate(val_loader):
                points = batch_data['lidar']    
                labels = batch_data["labels"]  

                gt_boxes = [label['bbox'] for label in labels]
                gt_classes = [label['class'] for label in labels]
                cls_targets, reg_targets = assigner.assign(gt_boxes, gt_classes)

                if len(gt_boxes) == 0:
                    print("     ⚠️ Skipping batch with no labels.")
                    continue
                
                cls_targets, reg_targets = assigner.assign(gt_boxes, gt_classes)

                if isinstance(points, list):
                    points_gpu = [p.to(device) for p in points]
                else:
                    points_gpu = points.to(device)

                cls_targets = cls_targets.unsqueeze(0).to(device)
                reg_targets = reg_targets.unsqueeze(0).to(device)

                cls_preds, reg_preds = model(points_gpu)
                val_loss = loss_fn(cls_preds, reg_preds, cls_targets, reg_targets)

                total_val_loss += val_loss.item()
                print(f"  ✅ Val batch {val_idx + 1} loss: {val_loss.item():.4f}")
                with open(train_log_path, "a") as f:
                    f.write(f"{epoch},-1,{val_loss:.4f}\n")
                

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"✅ Epoch {epoch} average validation loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f"pointpillar_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved model checkpoint to {checkpoint_path}")

    print("\n🎉 Training complete!")

if __name__ == "__main__":
    train_pointpillar(
        info_path_train="/work/10494/niran17/ls6/coopdet3d/data/mini/train/infos_train.pkl",
        info_path_val="/work/10494/niran17/ls6/coopdet3d/data/mini/val/infos_val.pkl",
        label_root="/work/10494/niran17/ls6/coopdet3d/data/mini/train/labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered",
        batch_size=1,
        epochs=10
    )
