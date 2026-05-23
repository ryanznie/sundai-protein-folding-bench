from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResidueAtom:
    chain_id: str
    residue_id: str
    atom_name: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class AlignmentMetrics:
    coverage: float
    tm_score: float
    lddt: float
    rmsd: float
    ca_rmsd: float
    gdt_ts_like: float
    matched_residues: int
    reference_residues: int


def _parse_pdb_atom_line(line: str) -> ResidueAtom | None:
    if not line.startswith(("ATOM", "HETATM")):
        return None
    atom_name = line[12:16].strip()
    residue_id = line[22:27].strip()
    chain_id = line[21].strip() or "_"
    try:
        return ResidueAtom(
            chain_id=chain_id,
            residue_id=residue_id,
            atom_name=atom_name,
            x=float(line[30:38].strip()),
            y=float(line[38:46].strip()),
            z=float(line[46:54].strip()),
        )
    except ValueError:
        return None


def _parse_mmcif_atom_line(line: str, headers: list[str]) -> ResidueAtom | None:
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) != len(headers):
        return None
    row = dict(zip(headers, parts, strict=False))
    atom_name = row.get("_atom_site.label_atom_id", "").strip().strip('"')
    chain_id = (
        row.get("_atom_site.auth_asym_id")
        or row.get("_atom_site.label_asym_id")
        or "_"
    ).strip().strip('"')
    residue_id = (
        row.get("_atom_site.auth_seq_id")
        or row.get("_atom_site.label_seq_id")
        or "0"
    ).strip().strip('"')
    try:
        return ResidueAtom(
            chain_id=chain_id,
            residue_id=residue_id,
            atom_name=atom_name,
            x=float(row["_atom_site.Cartn_x"]),
            y=float(row["_atom_site.Cartn_y"]),
            z=float(row["_atom_site.Cartn_z"]),
        )
    except (KeyError, ValueError):
        return None


def load_ca_trace(path: Path) -> list[tuple[float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text().splitlines()
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        atoms = []
        for line in lines:
            atom = _parse_pdb_atom_line(line)
            if atom and atom.atom_name == "CA":
                atoms.append(atom)
        return [(atom.x, atom.y, atom.z) for atom in atoms]

    headers: list[str] = []
    atoms: list[ResidueAtom] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            headers = []
            continue
        if stripped.startswith("_atom_site."):
            headers.append(stripped)
            continue
        if headers and stripped.startswith("ATOM"):
            atom = _parse_mmcif_atom_line(stripped, headers)
            if atom and atom.atom_name == "CA":
                atoms.append(atom)
    if atoms:
        return [(atom.x, atom.y, atom.z) for atom in atoms]

    pdb_atoms = []
    for line in lines:
        atom = _parse_pdb_atom_line(line)
        if atom and atom.atom_name == "CA":
            pdb_atoms.append(atom)
    return [(atom.x, atom.y, atom.z) for atom in pdb_atoms]


def _centroid(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _subtract(
    point: tuple[float, float, float], center: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (point[0] - center[0], point[1] - center[1], point[2] - center[2])


def _mat_vec_mul(
    matrix: list[list[float]], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _rotation_from_quaternion(
    q: tuple[float, float, float, float]
) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _normalize_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in q))
    if norm == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(v / norm for v in q)  # type: ignore[return-value]


def _power_iteration(matrix: list[list[float]], iterations: int = 50) -> tuple[float, float, float, float]:
    vector = (1.0, 0.0, 0.0, 0.0)
    for _ in range(iterations):
        next_vector = []
        for row in matrix:
            next_vector.append(sum(value * component for value, component in zip(row, vector, strict=False)))
        vector = _normalize_quaternion(tuple(next_vector))  # type: ignore[arg-type]
    return vector


def _optimal_rotation(
    reference: list[tuple[float, float, float]],
    prediction: list[tuple[float, float, float]],
) -> list[list[float]]:
    sxx = sxy = sxz = 0.0
    syx = syy = syz = 0.0
    szx = szy = szz = 0.0

    for ref_point, pred_point in zip(reference, prediction, strict=False):
        sxx += pred_point[0] * ref_point[0]
        sxy += pred_point[0] * ref_point[1]
        sxz += pred_point[0] * ref_point[2]
        syx += pred_point[1] * ref_point[0]
        syy += pred_point[1] * ref_point[1]
        syz += pred_point[1] * ref_point[2]
        szx += pred_point[2] * ref_point[0]
        szy += pred_point[2] * ref_point[1]
        szz += pred_point[2] * ref_point[2]

    horn = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    quaternion = _power_iteration(horn)
    return _rotation_from_quaternion(quaternion)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def _align_points(
    reference: list[tuple[float, float, float]],
    prediction: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    ref_center = _centroid(reference)
    pred_center = _centroid(prediction)
    centered_ref = [_subtract(point, ref_center) for point in reference]
    centered_pred = [_subtract(point, pred_center) for point in prediction]
    rotation = _optimal_rotation(centered_ref, centered_pred)
    rotated_pred = [_mat_vec_mul(rotation, point) for point in centered_pred]
    return centered_ref, rotated_pred


def _compute_lddt(
    reference: list[tuple[float, float, float]],
    prediction: list[tuple[float, float, float]],
    cutoff: float = 15.0,
) -> float:
    if len(reference) < 2:
        return 0.0
    total = 0.0
    count = 0
    thresholds = (0.5, 1.0, 2.0, 4.0)
    for i in range(len(reference)):
        for j in range(len(reference)):
            if i == j:
                continue
            ref_distance = _distance(reference[i], reference[j])
            if ref_distance > cutoff:
                continue
            pred_distance = _distance(prediction[i], prediction[j])
            delta = abs(ref_distance - pred_distance)
            total += sum(1.0 for threshold in thresholds if delta <= threshold) / len(thresholds)
            count += 1
    if count == 0:
        return 0.0
    return total / count


def _compute_tm_score(
    reference: list[tuple[float, float, float]],
    prediction: list[tuple[float, float, float]],
    reference_length: int,
) -> float:
    if reference_length == 0:
        return 0.0
    d0 = max(0.5, 1.24 * pow(max(reference_length - 15, 1), 1.0 / 3.0) - 1.8)
    total = 0.0
    for ref_point, pred_point in zip(reference, prediction, strict=False):
        distance = _distance(ref_point, pred_point)
        total += 1.0 / (1.0 + (distance / d0) ** 2)
    return total / reference_length


def _compute_gdt_ts(reference: list[tuple[float, float, float]], prediction: list[tuple[float, float, float]]) -> float:
    if not reference:
        return 0.0
    thresholds = (1.0, 2.0, 4.0, 8.0)
    fractions = []
    for threshold in thresholds:
        within = sum(
            1 for ref_point, pred_point in zip(reference, prediction, strict=False)
            if _distance(ref_point, pred_point) <= threshold
        )
        fractions.append(within / len(reference))
    return sum(fractions) / len(fractions)


def compute_alignment_metrics(
    reference: list[tuple[float, float, float]],
    prediction: list[tuple[float, float, float]],
) -> AlignmentMetrics:
    if not reference:
        raise ValueError("reference structure has no CA atoms")
    if not prediction:
        raise ValueError("prediction has no CA atoms")

    matched = min(len(reference), len(prediction))
    reference_subset = reference[:matched]
    prediction_subset = prediction[:matched]
    aligned_reference, aligned_prediction = _align_points(reference_subset, prediction_subset)
    squared_error = sum(
        _distance(ref_point, pred_point) ** 2
        for ref_point, pred_point in zip(aligned_reference, aligned_prediction, strict=False)
    )
    rmsd = math.sqrt(squared_error / matched)
    coverage = matched / len(reference)
    return AlignmentMetrics(
        coverage=coverage,
        tm_score=_compute_tm_score(aligned_reference, aligned_prediction, len(reference)),
        lddt=_compute_lddt(aligned_reference, aligned_prediction),
        rmsd=rmsd,
        ca_rmsd=rmsd,
        gdt_ts_like=_compute_gdt_ts(aligned_reference, aligned_prediction),
        matched_residues=matched,
        reference_residues=len(reference),
    )
