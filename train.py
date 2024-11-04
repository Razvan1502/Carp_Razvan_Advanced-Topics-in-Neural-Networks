import torch
from data_loader import get_dataloader
from models import load_model
from utils.early_stopping import EarlyStopping
from utils.optimizer_scheduler import configure_optimizer, configure_scheduler
from utils.logging_utils import initialize_logging

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloader(config['dataset'], config['batch_size'], config['augmentations'])
    model = load_model(config['model'], config['num_classes']).to(device)
    optimizer = configure_optimizer(model.parameters(), config['optimizer'])
    scheduler = configure_scheduler(optimizer, config['scheduler'])


    log = initialize_logging(config['logging'], config_name=config['config_name'])

    early_stopper = EarlyStopping(patience=config['early_stopping']['patience'])

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
           # print(f"Original image shape: {images.shape}")  # Debug line to check the input shape

            #images = images.view(images.size(0), -1)  # Flatten to (batch_size, 3072)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log training loss
        log.add_scalar('train_loss', running_loss / len(train_loader), epoch)

        # Validation
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += torch.nn.functional.cross_entropy(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)

        # Log validation metrics
        log.add_scalar('val_loss', val_loss, epoch)
        log.add_scalar('accuracy', accuracy, epoch)

        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}. Validation loss: {val_loss:.4f}")
            break

            # Scheduler step
        if scheduler:
            # StepLR and other schedulers without a metric should call scheduler.step() only
            if config['scheduler']['name'] != 'ReduceLROnPlateau':
                scheduler.step()
            # ReduceLROnPlateau requires the validation loss as an argument
            else:
                scheduler.step(val_loss)
