from __future__ import annotations

import os
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


def molecule_svg(molecule: Chem.Mol, background: tuple[float, float, float]) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 220)
    drawer.drawOptions().setBackgroundColour(background)
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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
        "smiles": smiles,
        "canonical": canonical,
        "MW": round(Descriptors.MolWt(molecule), 2),
        "logP": round(Descriptors.MolLogP(molecule), 2),
        "HBD": rdMolDescriptors.CalcNumHBD(molecule),
        "HBA": rdMolDescriptors.CalcNumHBA(molecule),
        "TPSA": round(rdMolDescriptors.CalcTPSA(molecule), 2),
        "svg_light": molecule_svg(molecule, (1.0, 1.0, 1.0)),
        "svg_dark": molecule_svg(molecule, (0.11, 0.13, 0.16)),
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
