import torch

class not_deterministic():
  '''
  removes the requires grad requirement for a list of modules, could be useful for models like GANs or to calculate perceptual losses
  Assumption: Input modules all have requires_grad = True, as it wont be able to return the requires_grad flags for each parameter
  '''
  def __init__(self):
    pass

  def __enter__(self):
    torch.set_deterministic(False)

  def __exit__(self, exc_type, exc_val, exc_tb):
    torch.set_deterministic(True)

m = torch.nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1).cuda()
m2 = torch.nn.Upsample(scale_factor=2, mode='nearest').cuda()
y = torch.rand(2, 10, 6, 6).cuda()
x = m(y)
#with not_deterministic():
#  print(torch.is_deterministic())
  #x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
x = m2(x)
x.mean().backward()
