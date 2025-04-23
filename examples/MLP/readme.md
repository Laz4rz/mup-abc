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

Correctly sampled
```python
linear = nn.Linear(3, 6)

# 2) Now overwrite its weights with Kaiming-normal
torch.manual_seed(0)


nn.init.kaiming_normal_(
    linear.weight, a=1, mode="fan_in", nonlinearity="leaky_relu"
)
kaiming = linear.weight.clone()

# 3) Reset the seed again, make a fresh contiguous tensor,
#    and draw exactly the same normals
torch.manual_seed(0)
fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
gain = nn.init.calculate_gain("leaky_relu", param=1)
std = gain / math.sqrt(fan_in)   # = 1 / sqrt(3)

with torch.no_grad():
    manual = torch.empty_like(linear.weight)  # shape (6,3), contiguous
    manual.normal_(0.0, std)

# Step 2: Manual replication using plain tensor
torch.manual_seed(0)
std = 1 / (3) ** 0.5
with torch.no_grad():
    manual_tensor = torch.empty(3, 6).T.contiguous()
    manual_tensor.normal_(0.0, std)
weight = nn.Parameter(manual_tensor)

print("Match?", torch.allclose(manual, kaiming))  # → True
print(kaiming)
print(manual)
print(manual_tensor)
```