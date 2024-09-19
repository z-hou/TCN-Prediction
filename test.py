import numpy as np
import torch
import pandas as pd



model_path = ""


def inference(model, input):
    predcition = model(input)

    return predcition


if __name__ == "__main__":

    model_path = "./checkpoint/model_lastest.pth"
    TCN_model = torch.load(model_path)
    TCN_model.eval()
    TCN_model.to("cpu")

    dummy_input = torch.rand(1,4,96)

    prediction = inference(TCN_model, dummy_input)

    print("output is: ", prediction.size())