Ref:
https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_resnet_example.html#torch-compile-resnet 



```python
traced_model = torch.jit.trace(model, [torch.randn((32, 3, 224, 224)).to("cuda")])

# Print the graph
print(traced_model.graph)
from torchviz import make_dot

input_tensor = torch.randn((32, 3, 224, 224)).cuda()
output = traced_model(input_tensor)

# Generate and save the computation graph
dot = make_dot(output, params=dict(traced_model.named_parameters()))
dot.render("traced_model_graph", format="png")  # Saves as PNG
```
