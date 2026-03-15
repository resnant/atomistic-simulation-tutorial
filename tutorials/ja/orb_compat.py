from __future__ import annotations

from types import SimpleNamespace

from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator


pfp_api_client = SimpleNamespace(__version__="orb-models")


class EstimatorCalcMode:
    PBE = "PBE"
    PBE_PLUS_D3 = "PBE_PLUS_D3"
    PBE_U = "PBE_U"
    CRYSTAL_PLUS_D3 = "CRYSTAL_PLUS_D3"
    WB97XD = "WB97XD"


class Estimator:
    def __init__(self, calc_mode: str = EstimatorCalcMode.PBE, model_version: str = "orb-v3", device: str = "cpu") -> None:
        self.calc_mode = calc_mode
        self.model_version = model_version
        self.device = device
        self.orbff, self.atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",
        )
        if calc_mode in (EstimatorCalcMode.PBE_PLUS_D3, EstimatorCalcMode.CRYSTAL_PLUS_D3):
            try:
                from orb_models.forcefield.inference.d3_model import AlchemiDFTD3, D3SumModel

                self.orbff = D3SumModel(
                    self.orbff,
                    AlchemiDFTD3(functional="PBE", damping="BJ", compile=False),
                )
            except Exception as exc:  # pragma: no cover
                print(f"D3補正の初期化に失敗したため、D3なしで続行します: {exc}")


class ASECalculator(ORBCalculator):
    def __init__(self, estimator: Estimator):
        self.estimator = estimator
        super().__init__(
            estimator.orbff,
            atoms_adapter=estimator.atoms_adapter,
            device=estimator.device,
        )
