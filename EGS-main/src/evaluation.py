import torch
import pandas as pd

@torch.no_grad()
def test(model, x, data, logits=None, evaluator=None, inference_loader=None, device="cuda"):
    if logits is None:
        model.eval()
        logits = (
            inference_sampled(model, x, inference_loader, device)
            if inference_loader
            else inference_full_batch(model, x, data.edge_index)
        )

    accs = []
    # print("data.y[mask]",  data.y)
    # print("x",x)
    # print("data.y[mask]size",data.y.size())
    # print("x.size",x.size())

    #for _, mask in data("test_mask"):
        # print("x", x[mask].size())
        # print("data.y[mask]",data.y[mask].size())
        #pd_x = pd.DataFrame(x[mask].cpu().numpy())
        #pd_y = pd.DataFrame(data.y[mask].cpu().numpy())
        #pd_x.to_csv("photox_1.csv",header=None,index=None)
        #pd_y.to_csv("photoy_1.csv",header=None,index=None)
    
    
    for _, mask in data("train_mask", "val_mask", "test_mask"):

        pred = logits[mask].max(1)[1]
        if evaluator:
            acc = evaluator.eval({"y_true": data.y[mask], "y_pred": pred.unsqueeze(1)})["acc"]
        else:
            acc = pred.eq(data.y[mask].squeeze()).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits


def inference_full_batch(model, x, edge_index):
    out = model(x, edge_index)

    return out


def inference_sampled(model, x, inference_loader, device):
    return model.inference(x, inference_loader, device)
