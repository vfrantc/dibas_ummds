import argparse
import numpy as np
import evaluate
from datasets import load_dataset
from torchvision.transforms import Resize, CenterCrop, Compose, Normalize, ToTensor
from transformers import AutoImageProcessor, DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

class ImageClassificationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'image': torch.tensor(self.dataset[idx]['pixel_values']),
            'label': torch.tensor(self.dataset[idx]['label'])
        }

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def get_predictions(model, data_loader):
    model = model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)

            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    return predictions, real_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate an image classification model.")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where the dataset is located")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/resnet-18", help="Path to the pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--work_dir", type=str, default="work", help="Directory for saving models and logs")

    args = parser.parse_args()

    dataset = 'UMMDS'
    model_name_or_path = args.model_name_or_path
    data_dir = args.data_dir

    ds = load_dataset("imagefolder", data_dir=f"{data_dir}/train/**", split="train")
    ds = ds.train_test_split(test_size=0.1, shuffle=True)
    test_ds = load_dataset("imagefolder", data_dir=f"{data_dir}/test/**", split="train")

    feature_extractor = AutoImageProcessor.from_pretrained(model_name_or_path, ignore_mismatched_sizes=True)
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    size = (
        feature_extractor.size["shortest_edge"]
        if "shortest_edge" in feature_extractor.size
        else (feature_extractor.size["height"], feature_extractor.size["width"])
    )
    _transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    ds = ds.with_transform(transforms)
    test_ds = test_ds.with_transform(transforms)
    data_collator = DefaultDataCollator()

    accuracy = evaluate.load("accuracy")

    labels = test_ds.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = AutoModelForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    work_dir = args.work_dir
    model_name = model_name_or_path.replace('/', '_')
    epoch = args.epoch
    training_args = TrainingArguments(
        f"{work_dir}/model/{dataset}/{model_name}/{epoch}/",
        save_strategy="no",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=int(epoch),
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        logging_dir=f"{work_dir}/tensorboard/{dataset}/{model_name}/{epoch}/",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(test_ds)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = [key.replace('_', ' ') for key, value in label2id.items()]

    test_dataset = ImageClassificationDataset(test_ds)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    y_pred, y_true = get_predictions(model, test_data_loader)

    report = classification_report(y_true, y_pred, target_names=classes, digits=3)

    result_path = f"{work_dir}/model/{dataset}/{model_name}/{epoch}/result.txt"
    with open(result_path, 'w') as f:
        f.write(report)
