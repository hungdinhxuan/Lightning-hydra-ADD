import torch
from typing import Any, Dict, Tuple, Union
from src.models.base.normal_module import NormalLitModule
from src.models.components.xlsr_conformertcm import Model as XLSRConformerTCM

class XLSRConformertcmNormalLitModule(NormalLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()
        
    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x)
    
    def init_model(self, **kwargs) -> torch.nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XLSRConformertcmMDTLitModule")
        return XLSRConformerTCM(
            self.args['conformer'], ssl_pretrained_path
        )