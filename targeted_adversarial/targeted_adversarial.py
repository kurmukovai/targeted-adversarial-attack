import torch
from tqdm import tqdm
# adopted from https://adversarial-ml-tutorial.org/adversarial_examples/
# TODO average over several training samples

def pgd_linf_targ(model, X, y, epsilon, alpha, num_iter, y_targ):
    """ Construct targeted adversarial examples on the examples X"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model.training:
        model.eval()

    model = model.to(device)
    delta = torch.zeros_like(X).detach().to(device)
    delta.requires_grad_()

    X = X.to(device)
    y = y.to(device)
    y_targ = y_targ.to(device)

    for t in tqdm(range(num_iter)):
        logits = model(X + delta)
        
        loss = (logits[:, y_targ] - logits.gather(1, y[:, None])[:,0]).sum()
        # TODO: try only maximizing logits[:, y_targ]
        # works, but worse
        
        # print(logits[:, y_targ].sum(), logits.gather(1, y[:, None])[:,0].sum())
        # loss = logits[:, y_targ].sum()
        loss.backward()

        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.sign()).clamp(-epsilon, epsilon)

        if delta.grad is not None:
            delta.grad.zero_()

    return delta.detach()
