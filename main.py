from data_loader import DataLoader
from rnn_model import LSTM
from train import Trainer
import torch


def evaluate(model, text):
    text = model.data_loader.tokenize(text)
    text = model.data_loader.encode_text(text)
    rating = model(text)
    print("Possible mark:", torch.argmax(rating) + 1)
    return


def main():
    input_size = 32
    hidden_size = 128
    num_layers = 3
    output_size = 5
    data_loader = DataLoader(input_size, "data.csv")
    model = LSTM(input_size, hidden_size, num_layers, output_size, data_loader)
    trainer = Trainer(model)
    trainer.train_model()
    evaluate(model, "This sweater is absolutely perfect")
    return 0


if __name__ == "__main__":
    main()
