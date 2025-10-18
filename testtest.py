import torch
if torch._C._GLIBCXX_USE_CXX11_ABI:
    print("安装 cxx11abiTRUE 版本")
else:
    print("安装 cxx11abiFALSE 版本")