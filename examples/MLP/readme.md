### muP implementation

#### Identical normal vs kaiming init: 

Relu (non-default)
```python
import torch
import torch.nn as nn

# Step 1: Kaiming from module
linear = nn.Linear(3, 3)
torch.manual_seed(0)
nn.init.kaiming_normal_(linear.weight, a=1, mode="fan_in", nonlinearity="relu")
kaiming_weights = linear.weight.clone()

# Step 2: Manual replication using plain tensor
torch.manual_seed(0)
std = (2 / 3) ** 0.5
with torch.no_grad():
    manual_tensor = torch.empty(3, 3)
    manual_tensor.normal_(0.0, std)
weight = nn.Parameter(manual_tensor)

# Step 3: Compare
print("Manual equal to kaiming?", torch.allclose(weight, kaiming_weights))
print("Manual:\n", weight)
print("Kaiming:\n", kaiming_weights)
```

LeakyRelu (default kaimin)
```python
# Step 1: Kaiming from module
linear = nn.Linear(3, 3)
torch.manual_seed(0)
nn.init.kaiming_normal_(linear.weight, a=1, mode="fan_in")
kaiming_weights = linear.weight.clone()

# Step 2: Manual replication using plain tensor
torch.manual_seed(0)
std = 1 / (3) ** 0.5
with torch.no_grad():
    manual_tensor = torch.empty(3, 3)
    manual_tensor.normal_(0.0, std)
weight = nn.Parameter(manual_tensor)

# Step 3: Compare
print("Manual equal to kaiming?", torch.allclose(weight, kaiming_weights))
print("Manual:\n", weight)
print("Kaiming:\n", kaiming_weights)
```

