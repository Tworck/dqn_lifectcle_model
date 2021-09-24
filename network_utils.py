import os
import torch

def save_model(model_name, net, optimizer):

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    save_path = os.path.join(os.getcwd(), "models", model_name.split(",")[0])
    torch.save(net.state_dict(), save_path)

    with open(save_path + "_information.txt", "a") as f:
        for l in model_name.split(","):
            f.write(l + "\n")


def load_model(net, path):
    net.load_state_dict(torch.load(path))
    net.eval()