import torch
from typing import Any, Dict, Tuple, Union
from src.models.base.mdt_module import MDTLitModule
from src.models.components.xlsr_aasist import XlsrAasist

class XlsrAasistMDTLitModule(MDTLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()
        
    def init_model(self, **kwargs) -> torch.nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XlsrAasistNormalLitModule")
        return XlsrAasist(ssl_pretrained_path)
    
    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x)