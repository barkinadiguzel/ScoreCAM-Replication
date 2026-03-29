def forward_score(model, masked_input, target_class=None):
    model.eval()
    with torch.no_grad():
        output = model(masked_input)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[:, target_class]
    return score
