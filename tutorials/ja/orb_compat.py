from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
import warnings

from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator


try:
    _orb_models_version = version("orb-models")
except PackageNotFoundError:
    _orb_models_version = "0.0.0"

pfp_api_client = SimpleNamespace(__version__=_orb_models_version)


class EstimatorCalcMode:
    PBE = "PBE"
    PBE_PLUS_D3 = "PBE_PLUS_D3"
    PBE_U = "PBE_U"
    CRYSTAL_PLUS_D3 = "CRYSTAL_PLUS_D3"
    WB97XD = "WB97XD"


class Estimator:
    def __init__(self, calc_mode: str = EstimatorCalcMode.PBE, model_version: str = "v8.0.0", device: str = "cpu") -> None:
        self.calc_mode = calc_mode
        self.model_version = model_version
        self.device = device

        model_loader = pretrained.orb_v3_conservative_inf_omat
        if calc_mode == EstimatorCalcMode.WB97XD and hasattr(pretrained, "orb_v3_conservative_omol"):
            model_loader = pretrained.orb_v3_conservative_omol
        elif "omol" in model_version.lower() and hasattr(pretrained, "orb_v3_conservative_omol"):
            model_loader = pretrained.orb_v3_conservative_omol
        elif calc_mode == EstimatorCalcMode.PBE_U:
            warnings.warn("EstimatorCalcMode.PBE_U uses the default Orb model in this compatibility layer.")

        self.orbff, self.atoms_adapter = model_loader(
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
            except (ImportError, RuntimeError) as exc:  # pragma: no cover
                warnings.warn(f"D3 correction initialization failed; continuing without D3: {exc}")


class ASECalculator(ORBCalculator):
    def __init__(self, estimator: Estimator):
        self.estimator = estimator
        super().__init__(
            estimator.orbff,
            atoms_adapter=estimator.atoms_adapter,
            device=estimator.device,
        )
