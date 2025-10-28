import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9'

import torch
import pyro
import pyro.distributions as dist

print(f"PyTorch version: {torch.__version__}")
print(f"Pyro version: {pyro.__version__}")

# 简单的分布功能测试
try:
    test_tensor = torch.randn(10, 3).cuda()
    student_t = dist.MultivariateStudentT(3, torch.zeros(3).cuda(), torch.eye(3).cuda())
    log_prob = student_t.log_prob(test_tensor)
    print("Pyro GPU passed")
except Exception as e:
    print(f"Pyro error: {e}")