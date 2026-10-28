from __future__ import annotations

import os
import json
import urllib.parse
import urllib.request
import re
from pathlib import Path

from flask_cors import CORS
from flask import Flask, jsonify, request, send_from_directory
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://127.0.0.1:5000,http://localhost:5000",
    ).split(",")
    if origin.strip()
]
CORS(
    app,
    resources={r"/api/*": {"origins": allowed_origins}},
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


def molecule_svg(molecule: Chem.Mol, background: tuple[float, float, float], dark: bool = False) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 220)
    drawer.drawOptions().setBackgroundColour(background)
    drawer.drawOptions().bondLineWidth = 2.5
    draw_molecule = Chem.AddHs(molecule) if molecule.GetNumBonds() == 0 else molecule
    drawer.DrawMolecule(draw_molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    if dark:
        svg = re.sub(r"#0000FF", "#69B7FF", svg, flags=re.IGNORECASE)
        svg = re.sub(r"#000000", "#FFFFFF", svg, flags=re.IGNORECASE)
        svg = re.sub(r"#000(?=[^0-9A-Fa-f]|$)", "#FFFFFF", svg, flags=re.IGNORECASE)
    return svg


KNOWN_NAMES = {
    "C": "Methane",
    "N": "Ammonia",
    "CCO": "Ethanol",
    "c1ccccc1": "Benzene",
    "CC(=O)Oc1ccccc1C(=O)O": "Aspirin",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "Caffeine",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O": "Glucose",
}
name_cache: dict[str, str] = {}


def molecule_name(smiles: str, canonical: str) -> str:
    if canonical in KNOWN_NAMES:
        return KNOWN_NAMES[canonical]
    if canonical in name_cache:
        return name_cache[canonical]

    try:
        query = urllib.parse.quote(canonical, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{query}/property/Title/JSON"
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
        title = data["PropertyTable"]["Properties"][0].get("Title")
        if title:
            name_cache[canonical] = str(title)
            return str(title)
    except Exception:
        pass

    return canonical


def analyze_smiles(smiles: str) -> dict[str, object]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string")

    canonical = Chem.MolToSmiles(molecule)
    molblock_3d = None
    molecule_3d = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(molecule_3d, params) == 0:
        AllChem.UFFOptimizeMolecule(molecule_3d, maxIters=200)
        molblock_3d = Chem.MolToMolBlock(molecule_3d)

    return {
        "name": molecule_name(smiles, canonical),
        "smiles": smiles,
        "canonical": canonical,
        "MW": round(Descriptors.MolWt(molecule), 2),
        "logP": round(Descriptors.MolLogP(molecule), 2),
        "HBD": rdMolDescriptors.CalcNumHBD(molecule),
        "HBA": rdMolDescriptors.CalcNumHBA(molecule),
        "TPSA": round(rdMolDescriptors.CalcTPSA(molecule), 2),
        "svg_light": molecule_svg(molecule, (1.0, 1.0, 1.0)),
        "svg_dark": molecule_svg(molecule, (0.11, 0.13, 0.16), dark=True),
        "molblock_3d": molblock_3d,
    }


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("smiles"), list):
        return jsonify(error="Expected a JSON object with a list of SMILES strings."), 400
    if any(not isinstance(smiles, str) for smiles in data["smiles"]):
        return jsonify(error="Every SMILES entry must be a string."), 400

    results = []
    errors = []
    for smiles in data["smiles"]:
        try:
            results.append(analyze_smiles(smiles))
        except (ValueError, RuntimeError) as error:
            errors.append({"smiles": smiles, "message": str(error)})

    return jsonify(results=results, errors=errors)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
