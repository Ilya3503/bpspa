"""
Контракты данных между слоями. Только структуры, без логики.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PoseResult:
    """Результат оценки позы одного кандидата.
    status: "ok" | "failed" — единственный признак, по которому
    потребитель решает, рисовать позу или пометить как нераспознанную."""
    status: str
    method: str
    position: Optional[List[float]] = None        # [x, y, z], координаты камеры
    orientation: Optional[List[float]] = None     # кватернион [x, y, z, w]
    transformation: Optional[List[List[float]]] = None   # 4x4
    fitness: Optional[float] = None
    inlier_rmse: Optional[float] = None
    reason: Optional[str] = None                  # заполнено при status="failed"
    extra: dict = field(default_factory=dict)     # global_fitness и пр. диагностика