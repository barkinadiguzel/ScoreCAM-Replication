class ActivationExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activation = None
        self._register_hook()

    def _register_hook(self):
        layer = dict([*self.model.named_modules()])[self.layer_name]
        layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activation = output.detach()

    def get_activation(self):
        return self.activation
