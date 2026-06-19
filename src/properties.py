"""Multi-property molecular profiling.

Beyond the trained LogS model, this module derives a panel of pharmaceutically
relevant properties directly from molecular structure using established
cheminformatics methods:

* Physicochemical descriptors (LogP, MW, TPSA, H-bond donors/acceptors) — RDKit.
* Drug-likeness — QED (Bickerton et al., *Nat. Chem.* 2012) and Lipinski's
  Rule of Five (Lipinski et al., *Adv. Drug Deliv. Rev.* 2001).
* Toxicity risk — count of medicinal-chemistry structural alerts from the
  Brenk and PAINS filter catalogs (Brenk et al., *ChemMedChem* 2008; Baell &
  Holloway, *J. Med. Chem.* 2010).
* Blood-brain-barrier penetration — a probability derived from Clark's logBB
  regression (Clark, *J. Pharm. Sci.* 1999), using TPSA and LogP.

Each property carries a qualitative ``sentiment`` so the UI can colour it
consistently without re-deriving thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Sentiments map to UI colours: good (green), neutral (grey), warn (amber), bad (red).
GOOD, NEUTRAL, WARN, BAD = "good", "neutral", "warn", "bad"


def _build_alert_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


_ALERT_CATALOG = _build_alert_catalog()


@dataclass
class Property:
    name: str
    value: float
    unit: str
    display: str
    category: str
    sentiment: str
    detail: str


@dataclass
class PropertyProfile:
    smiles: str
    properties: list[Property] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def structural_alerts(mol: Chem.Mol) -> list[str]:
    """Return the names of all Brenk/PAINS structural alerts the molecule matches."""
    return [m.GetDescription() for m in _ALERT_CATALOG.GetMatches(mol)]


def toxicity_risk(n_alerts: int) -> tuple[float, str, str]:
    """Map structural-alert count to (score 0-1, category, sentiment)."""
    score = min(1.0, n_alerts / 5.0)
    if n_alerts == 0:
        return score, "Low", GOOD
    if n_alerts <= 2:
        return score, "Moderate", WARN
    return score, "High", BAD


def bbb_penetration_probability(tpsa: float, logp: float) -> float:
    """
    Estimate blood-brain-barrier penetration probability.

    Uses Clark's (1999) regression ``logBB = -0.0148·TPSA + 0.152·LogP + 0.139``
    (TPSA substituted for the original dynamic PSA), then squashes logBB through a
    logistic centred near the CNS-/CNS+ boundary so the result reads as a 0-1
    probability rather than a partition coefficient.
    """
    log_bb = -0.0148 * tpsa + 0.152 * logp + 0.139
    return _sigmoid(2.0 * (log_bb + 0.5))


def _logs_property(logs: float) -> Property:
    if logs > 0:
        category, sentiment = "Highly soluble", GOOD
    elif logs > -4:
        category, sentiment = "Moderately soluble", WARN
    else:
        category, sentiment = "Poorly soluble", BAD
    return Property(
        name="Aqueous Solubility (LogS)",
        value=logs,
        unit="log mol/L",
        display=f"{logs:.2f}",
        category=category,
        sentiment=sentiment,
        detail="Predicted by the trained QSPR model.",
    )


def molecular_property_profile(smiles: str, logs: float | None = None) -> PropertyProfile | None:
    """
    Build the full multi-property profile for a SMILES string.

    ``logs`` is the model-predicted aqueous solubility; pass it in so the
    profile presents every property in one panel. Returns None for SMILES that
    RDKit cannot parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    qed = QED.qed(mol)

    alerts = structural_alerts(mol)
    tox_score, tox_category, tox_sentiment = toxicity_risk(len(alerts))
    bbb_prob = bbb_penetration_probability(tpsa, logp)

    # Lipinski Rule of Five violations.
    ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    properties: list[Property] = []
    if logs is not None:
        properties.append(_logs_property(logs))

    properties.append(
        Property(
            name="Lipophilicity (LogP)",
            value=logp,
            unit="",
            display=f"{logp:.2f}",
            category="Optimal" if 0 <= logp <= 5 else "Out of range",
            sentiment=GOOD if 0 <= logp <= 5 else (WARN if logp <= 6.5 else BAD),
            detail="Octanol-water partition coefficient (Crippen).",
        )
    )
    properties.append(
        Property(
            name="Molecular Weight",
            value=mw,
            unit="g/mol",
            display=f"{mw:.1f}",
            category="Drug-like" if mw <= 500 else "Heavy",
            sentiment=GOOD if mw <= 500 else (WARN if mw <= 700 else BAD),
            detail="Lighter compounds are generally more permeable.",
        )
    )
    properties.append(
        Property(
            name="Polar Surface Area (TPSA)",
            value=tpsa,
            unit="Å²",
            display=f"{tpsa:.1f}",
            category="Permeable" if tpsa <= 140 else "Low permeability",
            sentiment=GOOD if tpsa <= 140 else WARN,
            detail="TPSA ≤ 140 Å² favours oral absorption.",
        )
    )
    properties.append(
        Property(
            name="Drug-Likeness (QED)",
            value=qed,
            unit="",
            display=f"{qed:.2f}",
            category=(
                "Excellent" if qed >= 0.67
                else "Good" if qed >= 0.5
                else "Moderate" if qed >= 0.34
                else "Poor"
            ),
            sentiment=(
                GOOD if qed >= 0.67
                else NEUTRAL if qed >= 0.5
                else WARN if qed >= 0.34
                else BAD
            ),
            detail="Quantitative Estimate of Drug-likeness (0–1).",
        )
    )
    properties.append(
        Property(
            name="Lipinski Violations",
            value=float(ro5_violations),
            unit="of 4",
            display=str(ro5_violations),
            category="Rule of Five pass" if ro5_violations <= 1 else "Fails Ro5",
            sentiment=GOOD if ro5_violations == 0 else (NEUTRAL if ro5_violations == 1 else BAD),
            detail="MW≤500, LogP≤5, HBD≤5, HBA≤10.",
        )
    )
    properties.append(
        Property(
            name="Toxicity Risk",
            value=tox_score,
            unit="",
            display=f"{tox_category} ({len(alerts)} alerts)",
            category=tox_category,
            sentiment=tox_sentiment,
            detail="Structural alerts from Brenk/PAINS catalogs.",
        )
    )
    properties.append(
        Property(
            name="BBB Penetration",
            value=bbb_prob,
            unit="probability",
            display=f"{bbb_prob * 100:.0f}%",
            category="Likely" if bbb_prob >= 0.5 else "Unlikely",
            sentiment=NEUTRAL,
            detail="Estimated from Clark's (1999) logBB regression.",
        )
    )

    return PropertyProfile(smiles=smiles, properties=properties, alerts=alerts)
